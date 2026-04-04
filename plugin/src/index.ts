// index.ts — Nexus Router OpenClaw Plugin
//
// Hooks into `before_model_resolve` to route each chat turn to the best
// available model via the Nexus Router HTTP service.
//
// Flow:
//   1. before_model_resolve fires with the user prompt
//   2. Plugin calls POST http://127.0.0.1:7771/route
//   3. Router returns selected_model + fallbacks + reason
//   4. Plugin returns { modelOverride: selected_model }
//   5. OpenClaw uses that model for this turn

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { appendFile } from "node:fs/promises";

// ── Types ─────────────────────────────────────────────────────────────────────

interface NexusRouterConfig {
  routerUrl?: string;
  enabled?: boolean;
  costProfile?: "cheap" | "balanced" | "premium";
  debugMode?: boolean;
  minConfidence?: number;
  timeoutMs?: number;
}

type RouteMode = "auto" | "balanced" | "fast" | "reasoning" | "off";

interface RouteResponse {
  decision_id?: string;
  task_type: string;
  confidence: number;
  selected_model: string;
  selected_provider: string;
  fallbacks: string[];
  score: number;
  reason: string[];
  classifier_source?: "explicit" | "local" | "heuristic" | "llm" | "fallback";
  reply_context_used?: boolean;
}

interface LastRouteDecision {
  at: number;
  decisionId?: string;
  requestedRouteMode: RouteMode;
  routeMode: RouteMode;
  source: "compiled-prompt" | "raw-user";
  sourceTag: string;
  promptLen: number;
  promptText?: string;
  replyContextUsed: boolean;
  classifierSource: "explicit" | "local" | "heuristic" | "llm" | "fallback";
  costProfile: "cheap" | "balanced" | "premium";
  taskType: string;
  effectiveTaskType?: string;
  confidence: number;
  firstPassModel?: string;
  firstPassProvider?: string;
  selectedModel: string;
  selectedProvider: string;
  actualModel?: string;
  actualProvider?: string;
  usage?: { input?: number; output?: number; total?: number };
  runtimeSuccess?: boolean;
  runtimeDurationMs?: number;
  runtimeError?: string;
  fallbacks: string[];
  score: number;
  reason: string[];
  autoEscalated: boolean;
}

interface PendingOutcome {
  decisionId: string;
  selectedModel: string;
  selectedProvider: string;
  sessionKey?: string;
  shadowMode?: boolean;
  targetSenderId?: string;
}

interface FailureInference {
  httpStatus?: number;
  errorType?: "rate_limit" | "auth" | "server" | "timeout" | "unknown";
  quotaHint?: "low" | "exhausted";
  quotaRemainingRatio?: number;
  shouldCooldownOverride: boolean;
}

interface RouteRequestResult {
  decision: RouteResponse | null;
  error?: "timeout" | "http_error" | "network_error";
  status?: number;
}

interface RouteModePreference {
  pref_key: string;
  scope: "conversation" | "session" | "channel";
  mode: RouteMode;
  updated_at: string;
}

// ── Defaults ──────────────────────────────────────────────────────────────────

const DEFAULT_URL         = "http://127.0.0.1:7771";
const DEFAULT_CONFIDENCE  = 0.60;
const DEFAULT_TIMEOUT_MS  = 10000;
const DEFAULT_COST        = "balanced";
const PLUGIN_VERSION      = "0.1.0";
const RECENT_MESSAGE_TTL_MS = 5 * 60 * 1000;
const AUTO_ESCALATE_CONFIDENCE = 0.70;
const ROUTE_MODES = new Set(["auto", "balanced", "fast", "reasoning", "off"]);
const ROUTE_DEDUPE_WINDOW_MS = 20_000;
const ROUTE_BURST_WINDOW_MS = 5_000;
const ROUTE_BURST_MAX_CALLS = 4;
const ROUTE_BURST_BLOCK_MS = 60_000;
const STARTUP_BYPASS_WINDOW_MS = 30_000;
const COMPILED_RETRY_BYPASS_WINDOW_MS = 15 * 60 * 1000;
const FAILED_OVERRIDE_COOLDOWN_MS = 15 * 60 * 1000;

function splitSelectedModelRef(selectedModel: string): {
  providerOverride?: string;
  modelOverride: string;
} {
  const trimmed = selectedModel.trim();
  const slashIndex = trimmed.indexOf("/");

  if (slashIndex <= 0 || slashIndex === trimmed.length - 1) {
    return { modelOverride: trimmed };
  }

  return {
    providerOverride: trimmed.slice(0, slashIndex),
    modelOverride: trimmed.slice(slashIndex + 1),
  };
}

// ── HTTP helper ───────────────────────────────────────────────────────────────

async function debugLog(line: string): Promise<void> {
  try {
    await appendFile("/tmp/nexus-router-hook.log", `${new Date().toISOString()} ${line}\n`);
  } catch {
    // ignore debug logging failures
  }
}

async function routeRequest(
  url: string,
  prompt: string,
  costProfile: string,
  timeoutMs: number,
  routeMode: RouteMode,
  conversationContext?: string,
  useLlmClassifier?: boolean,
): Promise<RouteRequestResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${url}/route`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Editor-Version": PLUGIN_VERSION,
      },
      body: JSON.stringify({
        message: prompt,
        cost_profile: costProfile,
        route_mode: routeMode,
        conversation_context: conversationContext,
        use_llm_classifier: useLlmClassifier ?? false,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      return { decision: null, error: "http_error", status: res.status };
    }

    return { decision: (await res.json()) as RouteResponse };
  } catch (error: any) {
    if (error?.name === "AbortError") {
      return { decision: null, error: "timeout" };
    }
    return { decision: null, error: "network_error" };
  } finally {
    clearTimeout(timer);
  }
}

function describeRouteRequestFailure(result: RouteRequestResult, timeoutMs: number): string {
  if (result.error === "timeout") {
    return `router_timeout timeout_ms=${timeoutMs}`;
  }
  if (result.error === "http_error") {
    return `router_http_error status=${result.status ?? "?"}`;
  }
  if (result.error === "network_error") {
    return "router_network_error";
  }
  return "router_unavailable";
}

// ── Plugin entry ──────────────────────────────────────────────────────────────

const recentUserMessages = new Map<string, { text: string; at: number }>();
interface RouteModeEntry {
  mode: RouteMode;
  at: number;
  sticky: boolean;
}

interface RouteModeResolution {
  mode: RouteMode;
  source: "session" | "conversation" | "channel" | "default";
  key?: string;
}

const recentRouteModes = new Map<string, RouteModeEntry>();
const recentConversationRouteModes = new Map<string, RouteModeEntry>();
const recentConversationKeyBySession = new Map<string, { conversationKey: string; at: number }>();
const recentSessionKeyByConversation = new Map<string, { sessionKey: string; at: number }>();
const recentConversationContextBySession = new Map<string, { context: string; at: number }>();
const recentLastDecisionBySession = new Map<string, LastRouteDecision>();
const recentLastDecisionByConversation = new Map<string, LastRouteDecision>();
const recentLastDecisionByDecisionId = new Map<string, LastRouteDecision>();
const pendingOutcomeQueueBySessionId = new Map<string, PendingOutcome[]>();
const recentRouteCacheBySession = new Map<string, { text: string; mode: RouteMode; at: number; selectedModel?: string }>();
const recentSenderBySession = new Map<string, { senderId: string; channelId?: string; at: number }>();
const recentFeedbackPromptByDecisionId = new Map<string, { at: number }>();
const routeBurstBySession = new Map<string, { windowStart: number; count: number; blockedUntil?: number }>();
const recentSlashCommandBySession = new Map<string, { at: number; cmd: string }>();
const recentStartupBySession = new Map<string, { at: number; reason: string }>();
const recentFailedOverrides = new Map<string, { at: number; blockedUntil: number; reason: string }>();
const RECENT_COMMAND_GUARD_MS = 15_000;
let recentLastDecisionGlobal: LastRouteDecision | null = null;

function rememberRecentUserMessage(sessionKey: string, text: string): void {
  const trimmed = text.trim();
  if (!sessionKey || !trimmed) return;
  recentUserMessages.set(sessionKey, { text: trimmed, at: Date.now() });
}

// Route mode is an explicit user preference. Keep it sticky for the life of the
// session/conversation instead of silently expiring back to the default.
const STICKY_ROUTE_MODES = new Set<RouteMode>(["auto", "balanced", "fast", "reasoning", "off"]);

export function isShortFollowUpForContextualRouting(text?: string): boolean {
  const trimmed = (text ?? "").trim();
  if (!trimmed) return true;
  const words = trimmed.split(/\s+/).filter(Boolean);
  if (trimmed.length <= 24) return true;
  if (words.length <= 5 && trimmed.length <= 48) return true;
  return false;
}

export function shouldUseContextualLlmClassifier(
  routeMode: RouteMode,
  conversationContext?: string,
  routingText?: string,
): boolean {
  if (!conversationContext?.trim()) return false;
  if (!routingText?.trim()) return false;
  if (routeMode === "auto" || routeMode === "fast") return false;
  if (isShortFollowUpForContextualRouting(routingText)) return false;
  return true;
}

function rememberRouteMode(sessionKey: string, mode: RouteMode): void {
  if (!sessionKey || !ROUTE_MODES.has(mode)) return;
  recentRouteModes.set(sessionKey, {
    mode,
    at: Date.now(),
    sticky: STICKY_ROUTE_MODES.has(mode),
  });
}

function rememberConversationRouteMode(conversationKey: string, mode: RouteMode): void {
  if (!conversationKey || !ROUTE_MODES.has(mode)) return;
  recentConversationRouteModes.set(conversationKey, {
    mode,
    at: Date.now(),
    sticky: STICKY_ROUTE_MODES.has(mode),
  });
}

async function persistRouteModePreference(routerUrl: string, key: string, mode: RouteMode, scope: "conversation" | "session" | "channel" = "conversation"): Promise<void> {
  const trimmedKey = key.trim();
  if (!trimmedKey) return;
  try {
    await fetch(`${routerUrl}/route-mode`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Editor-Version": PLUGIN_VERSION,
      },
      body: JSON.stringify({ key: trimmedKey, mode, scope }),
    });
  } catch {
    // best-effort only; in-memory sticky mode remains as fallback
  }
}

async function loadPersistedRouteModePreference(routerUrl: string, key: string, scope: "conversation" | "session" | "channel" = "conversation"): Promise<RouteModePreference | null> {
  const trimmedKey = key.trim();
  if (!trimmedKey) return null;
  try {
    const res = await fetch(`${routerUrl}/route-mode?key=${encodeURIComponent(trimmedKey)}&scope=${encodeURIComponent(scope)}`, {
      headers: {
        "Editor-Version": PLUGIN_VERSION,
      },
    });
    if (!res.ok) return null;
    const payload = await res.json() as { preference?: RouteModePreference | null };
    return payload.preference ?? null;
  } catch {
    return null;
  }
}

function rememberConversationKeyForSession(sessionKey: string, conversationKey: string): void {
  if (!sessionKey || !conversationKey) return;
  const at = Date.now();
  recentConversationKeyBySession.set(sessionKey, { conversationKey, at });
  recentSessionKeyByConversation.set(conversationKey, { sessionKey, at });
}

function rememberConversationContextForSession(sessionKey: string, context: string): void {
  if (!sessionKey || !context.trim()) return;
  recentConversationContextBySession.set(sessionKey, { context: context.trim(), at: Date.now() });
}

function rememberSenderForSession(sessionKey: string, senderId?: string, channelId?: string): void {
  const trimmedSender = String(senderId ?? "").trim();
  if (!sessionKey || !trimmedSender) return;
  recentSenderBySession.set(sessionKey, { senderId: trimmedSender, channelId: channelId?.trim() || undefined, at: Date.now() });
}

function resolveSenderForSession(sessionKey?: string): { senderId: string; channelId?: string } | null {
  if (!sessionKey) return null;
  const entry = recentSenderBySession.get(sessionKey);
  if (!entry) return null;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentSenderBySession.delete(sessionKey);
    return null;
  }
  return { senderId: entry.senderId, channelId: entry.channelId };
}

function rememberFeedbackPrompt(decisionId?: string): void {
  const trimmed = String(decisionId ?? "").trim();
  if (!trimmed) return;
  recentFeedbackPromptByDecisionId.set(trimmed, { at: Date.now() });
}

function hasRecentFeedbackPrompt(decisionId?: string): boolean {
  const trimmed = String(decisionId ?? "").trim();
  if (!trimmed) return false;
  const entry = recentFeedbackPromptByDecisionId.get(trimmed);
  if (!entry) return false;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentFeedbackPromptByDecisionId.delete(trimmed);
    return false;
  }
  return true;
}

function buildFeedbackKeyboard(decisionId: string): Array<Array<{ text: string; callback_data: string }>> {
  return [
    [
      { text: "✅ Correct", callback_data: `routefb:ok:${decisionId}` },
      { text: "❌ Wrong", callback_data: `routefb:wrong:${decisionId}` },
    ],
    [
      { text: "coding", callback_data: `routefb:fix:${decisionId}:coding` },
      { text: "review", callback_data: `routefb:fix:${decisionId}:code_review` },
    ],
    [
      { text: "reasoning", callback_data: `routefb:fix:${decisionId}:reasoning` },
      { text: "chat", callback_data: `routefb:fix:${decisionId}:general_chat` },
    ],
  ];
}

async function sendTelegramFeedbackCard(
  api: any,
  targetSenderId: string,
  decision: RouteResponse,
  sourceTag: string,
  opts?: { shadowMode?: boolean; actualModel?: string },
): Promise<boolean> {
  const decisionId = String(decision.decision_id || "").trim();
  if (!decisionId || hasRecentFeedbackPrompt(decisionId)) {
    return false;
  }

  const task = String(decision.task_type || "unknown");
  const model = String(decision.selected_model || "unknown");
  const classifierSource = String(decision.classifier_source || "").toLowerCase();
  const confidence = (classifierSource === "fallback" || classifierSource === "heuristic")
    ? "fallback"
    : (Number.isFinite(decision.confidence) ? decision.confidence.toFixed(2) : "?");
  const shadowMode = Boolean(opts?.shadowMode);
  const actualModel = String(opts?.actualModel || "").trim();
  const lines = shadowMode
    ? [
        `🧭 shadow ${task} · proposed ${model} · conf ${confidence}`,
        `Actual reply model: ${actualModel || "unknown"}`,
        `Source: ${sourceTag}`,
        `Feedback?`,
      ]
    : [
        `🧭 ${task} · ${model} · ${confidence}`,
        `Source: ${sourceTag}`,
        `Feedback?`,
      ];
  const text = lines.join("\n");

  try {
    const telegram = api?.runtime?.telegram;
    if (!telegram?.sendMessageTelegram) {
      await debugLog(`[feedback-card] skipped decision=${decisionId} reason=no_telegram_runtime`);
      return false;
    }
    const result = await telegram.sendMessageTelegram(targetSenderId, text, {
      buttons: buildFeedbackKeyboard(decisionId),
      textMode: "markdown",
      cfg: api?.config?.loadConfig?.(),
    });
    rememberFeedbackPrompt(decisionId);
    await debugLog(`[feedback-card] sent decision=${decisionId} to=${targetSenderId} message_id=${result?.messageId ?? "?"}`);
    return true;
  } catch (error: any) {
    await debugLog(`[feedback-card] failed decision=${decisionId} to=${targetSenderId} error=${error?.message ?? String(error)}`);
    return false;
  }
}

function markRecentStartup(sessionKey?: string, reason = "startup"): void {
  if (!sessionKey) return;
  recentStartupBySession.set(sessionKey, { at: Date.now(), reason });
}

function takeRecentStartupReason(sessionKey?: string): string | null {
  if (!sessionKey) return null;
  const entry = recentStartupBySession.get(sessionKey);
  if (!entry) return null;
  if (Date.now() - entry.at > STARTUP_BYPASS_WINDOW_MS) {
    recentStartupBySession.delete(sessionKey);
    return null;
  }
  recentStartupBySession.delete(sessionKey);
  return entry.reason;
}

function classifySessionKind(sessionKey?: string): "cron" | "subagent" | "direct" | "slash" | "main" | "other" {
  const key = (sessionKey ?? "").toLowerCase();
  if (!key) return "other";
  if (key.startsWith("cron:") || key.includes(":cron:")) return "cron";
  if (key.includes(":subagent:")) return "subagent";
  if (key.includes(":direct:")) return "direct";
  if (key.includes(":slash:")) return "slash";
  if (key.endsWith(":main")) return "main";
  return "other";
}

function shouldDefaultRouteOff(sessionKey?: string): boolean {
  const kind = classifySessionKind(sessionKey);
  return kind === "direct" || kind === "slash" || kind === "main";
}

function inferCronLabel(prompt: string): string {
  const lowered = prompt.toLowerCase();
  if (lowered.includes("self_heal_alerts.py") || lowered.includes("self-heal incident")) return "self-heal";
  if (lowered.includes("morning-brief.sh") || lowered.includes("morning brief")) return "morning-brief";
  if (lowered.includes("backup status")) return "backup-status";
  return "job";
}

function buildSourceTag(ctx: any, source: "compiled-prompt" | "raw-user", prompt: string, startupReason?: string | null): string {
  if (ctx?.trigger === "cron" || classifySessionKind(ctx?.sessionKey) === "cron") {
    return `cron:${inferCronLabel(prompt)}`;
  }
  if (startupReason) {
    return "startup";
  }
  if (source === "raw-user") {
    return "user";
  }
  if (ctx?.trigger === "heartbeat") {
    return "heartbeat";
  }
  if (ctx?.trigger === "memory") {
    return "memory";
  }
  return "compiled";
}

function clearRecentRoutingState(sessionKey?: string): void {
  if (!sessionKey) return;
  recentUserMessages.delete(sessionKey);
  recentConversationContextBySession.delete(sessionKey);
  recentRouteCacheBySession.delete(sessionKey);
  recentSlashCommandBySession.delete(sessionKey);
  recentStartupBySession.delete(sessionKey);
  recentLastDecisionBySession.delete(sessionKey);
  recentSenderBySession.delete(sessionKey);
}

function resetInMemoryRoutingState(): void {
  recentUserMessages.clear();
  recentRouteModes.clear();
  recentConversationRouteModes.clear();
  recentConversationKeyBySession.clear();
  recentSessionKeyByConversation.clear();
  recentConversationContextBySession.clear();
  recentLastDecisionBySession.clear();
  recentLastDecisionByConversation.clear();
  recentLastDecisionByDecisionId.clear();
  pendingOutcomeQueueBySessionId.clear();
  recentRouteCacheBySession.clear();
  recentSenderBySession.clear();
  recentFeedbackPromptByDecisionId.clear();
  routeBurstBySession.clear();
  recentSlashCommandBySession.clear();
  recentStartupBySession.clear();
  recentFailedOverrides.clear();
  recentLastDecisionGlobal = null;
}

function buildFailedOverrideKeys(sessionKey: string | undefined, conversationKey: string | null, selectedModel: string, selectedProvider: string): string[] {
  const keys = new Set<string>();
  const model = selectedModel.trim();
  const provider = selectedProvider.trim();
  if (sessionKey) {
    keys.add(`session:${sessionKey}:model:${model}`);
    keys.add(`session:${sessionKey}:provider:${provider}`);
  }
  if (conversationKey) {
    keys.add(`conversation:${conversationKey}:model:${model}`);
    keys.add(`conversation:${conversationKey}:provider:${provider}`);
  }
  return Array.from(keys);
}

function rememberFailedOverride(sessionKey: string | undefined, conversationKey: string | null, selectedModel: string, selectedProvider: string, reason: string): void {
  const now = Date.now();
  const blockedUntil = now + FAILED_OVERRIDE_COOLDOWN_MS;
  for (const key of buildFailedOverrideKeys(sessionKey, conversationKey, selectedModel, selectedProvider)) {
    recentFailedOverrides.set(key, { at: now, blockedUntil, reason });
  }
}

function getFailedOverrideBlock(sessionKey: string | undefined, conversationKey: string | null, selectedModel: string, selectedProvider: string): { at: number; blockedUntil: number; reason: string } | null {
  const now = Date.now();
  for (const key of buildFailedOverrideKeys(sessionKey, conversationKey, selectedModel, selectedProvider)) {
    const entry = recentFailedOverrides.get(key);
    if (!entry) continue;
    if (now >= entry.blockedUntil) {
      recentFailedOverrides.delete(key);
      continue;
    }
    return entry;
  }
  return null;
}

function chooseUnblockedFallback(fallbacks: string[], sessionKey: string | undefined, conversationKey: string | null): string | null {
  for (const fallback of fallbacks) {
    const parsed = splitSelectedModelRef(fallback);
    const provider = parsed.providerOverride;
    if (!provider) {
      return fallback;
    }
    if (!getFailedOverrideBlock(sessionKey, conversationKey, fallback, provider)) {
      return fallback;
    }
  }
  return null;
}

function inferFailureFromRuntime(event: any): FailureInference {
  const errorText = typeof event?.error === "string" ? event.error.trim() : "";
  const lowered = errorText.toLowerCase();

  if (event?.success) {
    return { shouldCooldownOverride: false };
  }

  if (lowered.includes("429") || lowered.includes("rate limit") || lowered.includes("too many requests") || lowered.includes("quota") || lowered.includes("capacity") || lowered.includes("no capacity available")) {
    const exhausted = lowered.includes("quota exceeded")
      || lowered.includes("quota exhausted")
      || lowered.includes("exhausted")
      || lowered.includes("no capacity available")
      || lowered.includes("capacity exhausted");
    return {
      httpStatus: 429,
      errorType: "rate_limit",
      quotaHint: exhausted ? "exhausted" : "low",
      quotaRemainingRatio: exhausted ? 0 : undefined,
      shouldCooldownOverride: true,
    };
  }

  if (lowered.includes("401") || lowered.includes("403") || lowered.includes("unauthorized") || lowered.includes("forbidden") || lowered.includes("auth")) {
    return {
      httpStatus: lowered.includes("403") || lowered.includes("forbidden") ? 403 : 401,
      errorType: "auth",
      shouldCooldownOverride: true,
    };
  }

  if (lowered.includes("timeout") || lowered.includes("timed out")) {
    return {
      errorType: "timeout",
      shouldCooldownOverride: true,
    };
  }

  if (lowered.includes("500") || lowered.includes("502") || lowered.includes("503") || lowered.includes("server error") || lowered.includes("bad gateway") || lowered.includes("service unavailable")) {
    return {
      httpStatus: lowered.includes("503") || lowered.includes("service unavailable") ? 503 : lowered.includes("502") || lowered.includes("bad gateway") ? 502 : 500,
      errorType: "server",
      shouldCooldownOverride: true,
    };
  }

  return {
    errorType: "unknown",
    shouldCooldownOverride: true,
  };
}

function takeRecentConversationContext(sessionKey?: string): string | null {
  if (!sessionKey) return null;
  const entry = recentConversationContextBySession.get(sessionKey);
  if (!entry) return null;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentConversationContextBySession.delete(sessionKey);
    return null;
  }
  recentConversationContextBySession.delete(sessionKey);
  return entry.context;
}

function resolveConversationKeyForSession(sessionKey?: string): string | null {
  if (!sessionKey) return null;
  const entry = recentConversationKeyBySession.get(sessionKey);
  if (!entry) return null;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentConversationKeyBySession.delete(sessionKey);
    return null;
  }
  return entry.conversationKey;
}

function resolveSessionKeyForConversation(conversationKey: string): string | null {
  const entry = recentSessionKeyByConversation.get(conversationKey);
  if (!entry) return null;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentSessionKeyByConversation.delete(conversationKey);
    return null;
  }
  return entry.sessionKey;
}

function shouldBypassCompiledRetryRouting(
  ctx: any,
  source: "compiled-prompt" | "raw-user",
  sessionKey?: string,
): boolean {
  if (source !== "compiled-prompt") return false;
  if (ctx?.trigger !== "user") return false;
  if (!sessionKey) return false;

  const last = recentLastDecisionBySession.get(sessionKey);
  if (!last) return false;
  if (Date.now() - last.at > COMPILED_RETRY_BYPASS_WINDOW_MS) {
    recentLastDecisionBySession.delete(sessionKey);
    return false;
  }

  return true;
}

function shouldBlockRoutingForBurst(sessionRef: string): boolean {
  if (!sessionRef) return false;
  const now = Date.now();
  const current = routeBurstBySession.get(sessionRef) ?? { windowStart: now, count: 0 };

  if (current.blockedUntil && now < current.blockedUntil) {
    routeBurstBySession.set(sessionRef, current);
    return true;
  }

  if (now - current.windowStart > ROUTE_BURST_WINDOW_MS) {
    current.windowStart = now;
    current.count = 0;
    current.blockedUntil = undefined;
  }

  current.count += 1;
  if (current.count > ROUTE_BURST_MAX_CALLS) {
    current.blockedUntil = now + ROUTE_BURST_BLOCK_MS;
    routeBurstBySession.set(sessionRef, current);
    return true;
  }

  routeBurstBySession.set(sessionRef, current);
  return false;
}

function buildConversationKeyFromContext(ctx: any): string | null {
  const channel = ctx?.channelId ?? ctx?.channel;
  const account = ctx?.accountId ?? "default";
  const fromTo = ctx?.from ?? ctx?.to ?? "";
  const thread = ctx?.messageThreadId ?? "";
  if (!channel && !fromTo && !thread) return null;
  return [channel ?? "unknown", account, fromTo, thread].join(":");
}

function takeRecentUserMessage(sessionKey?: string): string | null {
  if (!sessionKey) return null;
  const entry = recentUserMessages.get(sessionKey);
  if (!entry) return null;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentUserMessages.delete(sessionKey);
    return null;
  }
  recentUserMessages.delete(sessionKey);
  return entry.text;
}

function getRecentRouteModeEntry(sessionKey?: string): RouteModeEntry | null {
  if (!sessionKey) return null;
  const entry = recentRouteModes.get(sessionKey);
  if (!entry) return null;
  if (!entry.sticky && Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentRouteModes.delete(sessionKey);
    return null;
  }
  return entry;
}

function takeRecentRouteMode(sessionKey?: string): RouteMode | null {
  return getRecentRouteModeEntry(sessionKey)?.mode ?? null;
}

function getRecentConversationRouteModeEntry(conversationKey?: string): RouteModeEntry | null {
  if (!conversationKey) return null;
  const entry = recentConversationRouteModes.get(conversationKey);
  if (!entry) return null;
  if (!entry.sticky && Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentConversationRouteModes.delete(conversationKey);
    return null;
  }
  return entry;
}

function takeRecentConversationRouteMode(conversationKey?: string): RouteMode | null {
  return getRecentConversationRouteModeEntry(conversationKey)?.mode ?? null;
}

function parseRouteModeFromText(text: string): RouteMode | null {
  const lowered = text.trim().toLowerCase();
  const match = lowered.match(/^(?:⚙️\s*)?routing mode(?:\s*(?:set to|:|=)\s*|\s+)(auto|balanced|fast|reasoning|off)(?:\s*\([^)]*\))?\.?$/i);
  if (match?.[1]) {
    const mode = match[1].toLowerCase();
    if (ROUTE_MODES.has(mode)) {
      return mode as RouteMode;
    }
  }
  return null;
}

function rememberLastDecision(sessionKey: string | undefined, decision: LastRouteDecision): void {
  if (sessionKey) {
    recentLastDecisionBySession.set(sessionKey, decision);
    const conversationKey = resolveConversationKeyForSession(sessionKey);
    if (conversationKey) {
      recentLastDecisionByConversation.set(conversationKey, decision);
    }
  }
  if (decision.decisionId) {
    recentLastDecisionByDecisionId.set(decision.decisionId, decision);
  }
  recentLastDecisionGlobal = decision;
}

function enqueuePendingOutcome(sessionId: string, pending: PendingOutcome): void {
  const queue = pendingOutcomeQueueBySessionId.get(sessionId) ?? [];
  queue.push(pending);
  pendingOutcomeQueueBySessionId.set(sessionId, queue);
}

function peekPendingOutcome(sessionId: string): PendingOutcome | undefined {
  const queue = pendingOutcomeQueueBySessionId.get(sessionId);
  if (!queue?.length) return undefined;
  return queue[0];
}

function shiftPendingOutcome(sessionId: string): PendingOutcome | undefined {
  const queue = pendingOutcomeQueueBySessionId.get(sessionId);
  if (!queue?.length) return undefined;
  const first = queue.shift();
  if (queue.length === 0) {
    pendingOutcomeQueueBySessionId.delete(sessionId);
  } else {
    pendingOutcomeQueueBySessionId.set(sessionId, queue);
  }
  return first;
}

function takeLastDecisionForConversation(conversationKey: string): LastRouteDecision | null {
  const entry = recentLastDecisionByConversation.get(conversationKey);
  if (!entry) return null;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentLastDecisionByConversation.delete(conversationKey);
    return null;
  }
  return entry;
}

function extractEffectiveTaskType(reason: string[]): string | undefined {
  for (const item of reason) {
    const m = item.match(/adapted to '([^']+)'/i);
    if (m) return m[1];
  }
  return undefined;
}

function buildRouteLastReply(last: LastRouteDecision | null): { text: string } {
  if (!last) {
    return {
      text: "No recent routing decision found yet for this conversation.",
    };
  }

  const fallbacks = last.fallbacks.length ? last.fallbacks.join(", ") : "none";
  const reason = last.reason.length ? last.reason.slice(0, 3).join("; ") : "n/a";
  const contextLabel = last.replyContextUsed ? "reply-context used: yes" : "reply-context used: no";
  const escalationLabel = last.autoEscalated ? `${last.requestedRouteMode} → ${last.routeMode}` : last.routeMode;

  const lines = [
    `Routing: ${escalationLabel}`,
    `Input: ${last.source}`,
    `Source tag: ${last.sourceTag}`,
    `Context: ${contextLabel}`,
    `Classifier: ${last.classifierSource}`,
    `Classifier task: ${last.taskType}`,
    `Effective route task: ${last.effectiveTaskType ?? last.taskType}`,
    `Confidence: ${last.confidence.toFixed(2)}`,
    `First pass: ${last.firstPassModel ?? last.selectedModel}`,
    `Selected: ${last.selectedModel}`,
    `Fallbacks: ${fallbacks}`,
    `Reason: ${reason}`,
  ];

  return { text: lines.join("\n") };
}

function buildRouteExplainReply(last: LastRouteDecision | null): { text: string } {
  if (!last) {
    return {
      text: "No recent routing decision found yet for this conversation.",
    };
  }

  const fallbacks = last.fallbacks.length ? last.fallbacks.join(", ") : "none";
  const reason = last.reason.length ? last.reason.join("; ") : "n/a";
  const actual = last.actualModel ? `${last.actualProvider ?? "unknown"}/${last.actualModel}` : "not recorded yet";
  const usageBits = last.usage
    ? ` in=${last.usage.input ?? "?"} out=${last.usage.output ?? "?"} total=${last.usage.total ?? "?"}`
    : "";
  const contextLabel = last.replyContextUsed ? "reply-context used: yes" : "reply-context used: no";
  const escalationLabel = last.autoEscalated ? `${last.requestedRouteMode} → ${last.routeMode}` : last.routeMode;
  const overrideDetected = !!(last.actualModel && last.actualModel !== last.selectedModel);
  const runtimeStatus = last.runtimeSuccess === undefined
    ? "not recorded yet"
    : (last.runtimeSuccess ? "success" : "error");
  const runtimeDuration = last.runtimeDurationMs === undefined ? "" : ` (${last.runtimeDurationMs}ms)`;

  const lines = [
    `Routing: ${escalationLabel}`,
    `Input: ${last.source}`,
    `Source tag: ${last.sourceTag}`,
    `Context: ${contextLabel}`,
    `Classifier: ${last.classifierSource}`,
    `Classifier task: ${last.taskType}`,
    `Effective route task: ${last.effectiveTaskType ?? last.taskType}`,
    `Confidence: ${last.confidence.toFixed(2)}`,
    `First pass: ${last.firstPassModel ?? last.selectedModel}`,
    `Selected: ${last.selectedModel}`,
    `Inspected turn model: ${actual}${usageBits}`,
    `Execution override detected: ${overrideDetected ? "yes" : "no"}`,
    `Runtime status: ${runtimeStatus}${runtimeDuration}`,
    `Fallbacks: ${fallbacks}`,
    `Reason: ${reason}`,
  ];

  lines.push("Note: the message footer may still show the current session/base model for the /route command itself, not the inspected turn model above.");

  if (last.runtimeError?.trim()) {
    lines.push(`Runtime error: ${last.runtimeError.trim()}`);
  }

  return { text: lines.join("\n") };
}

async function buildRouteCompareReply(
  routerUrl: string,
  last: LastRouteDecision | null,
  modes: RouteMode[],
  timeoutMs: number,
): Promise<{ text: string }> {
  if (!last?.promptText?.trim()) {
    return {
      text: "No recent prompt found to compare. Send a normal message first, then run /route compare.",
    };
  }

  const uniqueModes = Array.from(new Set(modes.filter((mode) => ROUTE_MODES.has(mode))));
  const results = await Promise.all(
    uniqueModes.map(async (mode) => {
      const costProfile = resolveCostProfileForRouteMode(mode, last.costProfile);
      const result = await routeRequest(routerUrl, last.promptText ?? "", costProfile, timeoutMs, mode);
      return { mode, costProfile, decision: result.decision, error: result.error, status: result.status };
    }),
  );

  const lines = [
    `Prompt: ${last.promptText.slice(0, 120)}${last.promptText.length > 120 ? "…" : ""}`,
    `Task hint: ${last.taskType} · source=${last.source}`,
    "",
  ];

  for (const row of results) {
    if (!row.decision) {
      if (row.error === "timeout") {
        lines.push(`${row.mode}: timeout (${timeoutMs}ms)`);
      } else if (row.error === "http_error") {
        lines.push(`${row.mode}: http error (${row.status ?? "?"})`);
      } else if (row.error === "network_error") {
        lines.push(`${row.mode}: network error`);
      } else {
        lines.push(`${row.mode}: unavailable`);
      }
      continue;
    }
    lines.push(
      `${row.mode}: ${row.decision.selected_model} ` +
      `(task=${row.decision.task_type}, conf=${row.decision.confidence.toFixed(2)}, score=${row.decision.score.toFixed(3)}, profile=${row.costProfile})`,
    );
  }

  return { text: lines.join("\n") };
}

function resolveCostProfileForRouteMode(
  mode: RouteMode,
  defaultProfile: "cheap" | "balanced" | "premium",
): "cheap" | "balanced" | "premium" {
  switch (mode) {
    case "auto":
      return "cheap";
    case "balanced":
      return defaultProfile;
    case "fast":
      return "cheap";
    case "reasoning":
      return "premium";
    case "off":
      return defaultProfile;
  }
}

function buildRouteInteractiveReply(mode?: RouteMode, scopeLabel = "this conversation"): {
  text: string;
  interactive: { blocks: Array<{ type: "text"; text: string } | { type: "buttons"; buttons: Array<{ label: string; value: string; style?: "primary" | "secondary" | "success" | "danger" }> }> };
} {
  const label = mode ?? "auto";
  return {
    text: `⚙️ Routing mode: ${label} (${scopeLabel}).`,
    interactive: {
      blocks: [
        { type: "text", text: `Choose a routing mode (current: ${label}, persisted in router state):` },
        {
          type: "buttons",
          buttons: [
            { label: "Auto", value: "/route auto", style: "primary" },
            { label: "Balanced", value: "/route balanced", style: "secondary" },
            { label: "Fast", value: "/route fast", style: "success" },
            { label: "Reasoning", value: "/route reasoning", style: "secondary" },
            { label: "Off", value: "/route off", style: "danger" },
          ],
        },
      ],
    },
  };
}

function resolveLastDecisionForContext(ctx: any, conversationKey: string): LastRouteDecision | null {
  const bySession = ctx?.sessionKey ? recentLastDecisionBySession.get(ctx.sessionKey) ?? null : null;
  return bySession ?? takeLastDecisionForConversation(conversationKey) ?? recentLastDecisionGlobal;
}

function buildRouteHelpText(currentMode: RouteMode): string {
  return [
    `⚙️ Nexus Router help`,
    `Current session mode: ${currentMode} (persisted until changed)`,
    ``,
    `Modes:`,
    `- auto: cheap-first routing; escalates to balanced when confidence is weak`,
    `- balanced: quality-first default`,
    `- fast: stronger cost bias; prefers cheaper/faster models`,
    `- reasoning: stronger-model bias for planning/trade-off tasks`,
    `- off: bypass router overrides`,
    ``,
    `Commands:`,
    `- /route status → show current session mode`,
    `- /route last → show last routing decision (short form)`,
    `- /route explain → show richer diagnostics (context + escalation + classifier source)`,
    `- /route compare [fast balanced reasoning] → compare modes on the last prompt`,
    ``,
    `Examples:`,
    `- /route fast`,
    `- /route reasoning`,
    `- /route compare`,
  ].join("\n");
}

function collectTextFragments(value: unknown, output: string[], seen = new WeakSet<object>()): void {
  if (typeof value === "string") {
    output.push(value);
    return;
  }
  if (!value || typeof value !== "object") {
    return;
  }
  const obj = value as Record<string, unknown>;
  if (seen.has(obj)) return;
  seen.add(obj);
  if (Array.isArray(value)) {
    for (const item of value) collectTextFragments(item, output, seen);
    return;
  }
  for (const entry of Object.values(obj)) {
    collectTextFragments(entry, output, seen);
  }
}

function buildConversationContextFromMessages(messages: unknown[]): string {
  const fragments: string[] = [];
  for (const message of messages.slice(-8)) {
    collectTextFragments(message, fragments);
  }
  return fragments
    .map((frag) => frag.trim())
    .filter(Boolean)
    .slice(-12)
    .join("\n")
    .slice(0, 2000);
}

async function resolveRouteModeFromSession(api: any, sessionKey?: string): Promise<RouteMode> {
  return (await resolveRouteModeDetailsFromContext(api, { sessionKey })).mode;
}

async function resolveRouteModeDetailsFromContext(api: any, ctx: any): Promise<RouteModeResolution> {
  const candidates: Array<RouteModeResolution & { at: number }> = [];
  const seen = new Set<string>();

  const routerUrl = (api?.config ?? {}).routerUrl ?? DEFAULT_URL;

  const addConversationCandidate = (
    key: string | null | undefined,
    source: "conversation" | "channel",
  ): void => {
    if (!key || seen.has(`${source}:${key}`)) return;
    seen.add(`${source}:${key}`);
    const entry = getRecentConversationRouteModeEntry(key);
    if (!entry) return;
    candidates.push({ mode: entry.mode, source, key, at: entry.at });
  };

  const addSessionCandidate = (key?: string): void => {
    if (!key || seen.has(`session:${key}`)) return;
    seen.add(`session:${key}`);
    const entry = getRecentRouteModeEntry(key);
    if (!entry) return;
    candidates.push({ mode: entry.mode, source: "session", key, at: entry.at });
  };

  addSessionCandidate(ctx?.sessionKey);

  const conversationKey = buildConversationKeyFromContext(ctx);
  addConversationCandidate(conversationKey, "conversation");

  const mappedConversationKey = resolveConversationKeyForSession(ctx?.sessionKey);
  if (mappedConversationKey && mappedConversationKey !== conversationKey) {
    addConversationCandidate(mappedConversationKey, "conversation");
  }

  const conversationIdKey = (ctx?.channelId ?? ctx?.channel)
    ? [ctx.channelId ?? ctx.channel, ctx.accountId ?? "default", ctx.conversationId ?? "", ""].join(":")
    : null;
  if (conversationIdKey && conversationIdKey !== conversationKey) {
    addConversationCandidate(conversationIdKey, "conversation");
  }

  const channelKey = ctx?.channelId ?? ctx?.channel ?? "";
  if (channelKey) {
    addConversationCandidate(channelKey, "channel");
  }

  if (conversationKey) {
    const persistedConversation = await loadPersistedRouteModePreference(routerUrl, conversationKey, "conversation");
    if (persistedConversation) {
      candidates.push({ mode: persistedConversation.mode, source: "conversation", key: conversationKey, at: Date.parse(persistedConversation.updated_at) || 0 });
    }
  }
  if (ctx?.sessionKey) {
    const persistedSession = await loadPersistedRouteModePreference(routerUrl, ctx.sessionKey, "session");
    if (persistedSession) {
      candidates.push({ mode: persistedSession.mode, source: "session", key: ctx.sessionKey, at: Date.parse(persistedSession.updated_at) || 0 });
    }
  }
  if (channelKey) {
    const persistedChannel = await loadPersistedRouteModePreference(routerUrl, channelKey, "channel");
    if (persistedChannel) {
      candidates.push({ mode: persistedChannel.mode, source: "channel", key: channelKey, at: Date.parse(persistedChannel.updated_at) || 0 });
    }
  }

  candidates.sort((a, b) => b.at - a.at);
  const resolved = candidates[0];
  if (!resolved) {
    return { mode: "auto", source: "default" };
  }

  if (ctx?.sessionKey && resolved.source !== "session") {
    rememberRouteMode(ctx.sessionKey, resolved.mode);
  }

  return {
    mode: resolved.mode,
    source: resolved.source,
    key: resolved.key,
  };
}

async function resolveRouteModeFromContext(api: any, ctx: any): Promise<RouteMode> {
  return (await resolveRouteModeDetailsFromContext(api, ctx)).mode;
}

function shouldAutoEscalate(confidence: number): boolean {
  return confidence < AUTO_ESCALATE_CONFIDENCE;
}

export const __testHelpers = {
  rememberRouteMode,
  rememberLastDecision,
  rememberFailedOverride,
  resolveRouteModeDetailsFromContext,
  shouldUseContextualLlmClassifier,
  shouldBypassCompiledRetryRouting,
  isShortFollowUpForContextualRouting,
  inferFailureFromRuntime,
  getFailedOverrideBlock,
  chooseUnblockedFallback,
  shouldAutoEscalate,
  resetInMemoryRoutingState,
};

export default definePluginEntry({
  id: "nexus-router",
  name: "Nexus Router",
  description: "Routes each chat turn to the best model via the Nexus Router service",

  register(api: any) {
    const cfg = (api.config ?? {}) as NexusRouterConfig;

    const routerUrl      = cfg.routerUrl     ?? DEFAULT_URL;
    const enabled        = cfg.enabled       ?? true;
    const costProfile    = cfg.costProfile   ?? DEFAULT_COST;
    const debugMode      = cfg.debugMode     ?? false;
    const minConfidence  = cfg.minConfidence ?? DEFAULT_CONFIDENCE;
    const timeoutMs      = cfg.timeoutMs     ?? DEFAULT_TIMEOUT_MS;

    if (!enabled) {
      if (debugMode) console.log("[nexus-router] disabled via config");
      return;
    }

    api.on("session_start", (event: any, ctx: any) => {
      markRecentStartup(event.sessionKey ?? ctx.sessionKey, event.resumedFrom ? "resume" : "session-start");
    });

    api.on("before_reset", async (event: any, ctx: any) => {
      const sessionKey = ctx.sessionKey;
      clearRecentRoutingState(sessionKey);
      markRecentStartup(sessionKey, `reset:${event.reason ?? "unknown"}`);
      await debugLog(`[route-reset] session=${sessionKey ?? "unknown"} cleared-ephemeral-state reason=${event.reason ?? "unknown"}`);
    });

    api.registerCommand({
      name: "route",
      description: "Choose routing mode for model selection.",
      acceptsArgs: true,
      requireAuth: true,
      handler: async (ctx: any) => {
        const rawArgs = ctx.args?.trim() ?? "";
        const loweredArgs = rawArgs.toLowerCase();
        const arg = loweredArgs;
        const normalized = arg && ROUTE_MODES.has(arg) ? (arg as RouteMode) : undefined;
        const conversationKey = buildConversationKeyFromContext(ctx) ?? [ctx.channelId ?? ctx.channel, ctx.accountId ?? "default", ctx.from ?? ctx.to ?? "", ctx.messageThreadId ?? ""].join(":");
        const sessionKeyForConversation = conversationKey ? resolveSessionKeyForConversation(conversationKey) ?? undefined : undefined;
        const modeResolution = await resolveRouteModeDetailsFromContext(api, ctx);
        const currentMode = modeResolution.mode;

        if (!arg || arg === "help" || arg === "?") {
          return {
            text: buildRouteHelpText(currentMode),
            interactive: buildRouteInteractiveReply(currentMode, "this conversation").interactive,
          };
        }

        if (arg === "status") {
          const scopeLabel = modeResolution.source === "default"
            ? "default"
            : `resolved from ${modeResolution.source}`;
          return buildRouteInteractiveReply(currentMode, scopeLabel);
        }

        if (arg === "last") {
          const last = resolveLastDecisionForContext(ctx, conversationKey);
          return buildRouteLastReply(last);
        }

        if (arg === "explain") {
          const last = resolveLastDecisionForContext(ctx, conversationKey);
          return buildRouteExplainReply(last);
        }

        if (arg.startsWith("compare")) {
          const last = resolveLastDecisionForContext(ctx, conversationKey);
          const modeList = rawArgs.split(/\s+/).slice(1).map((m: string) => m.toLowerCase()).filter(Boolean) as RouteMode[];
          const compareModes: RouteMode[] = modeList.length ? modeList : ["fast", "balanced", "reasoning"] as RouteMode[];
          return buildRouteCompareReply(routerUrl, last, compareModes, timeoutMs);
        }

        if (!normalized) {
          return {
            text: buildRouteHelpText(currentMode),
            interactive: buildRouteInteractiveReply("auto").interactive,
          };
        }

        const routeModeSessionKey = sessionKeyForConversation ?? ctx.conversationId ?? ctx.sessionKey ?? ctx.channelId ?? ctx.senderId ?? "";
        rememberRouteMode(routeModeSessionKey, normalized);
        if (ctx.sessionKey && ctx.sessionKey !== routeModeSessionKey) rememberRouteMode(ctx.sessionKey, normalized);
        rememberConversationRouteMode(conversationKey, normalized);
        // Also store with a channel-only key so before_model_resolve can find the mode
        // even when its hook ctx does not carry the full from/to/thread fields.
        const channelOnlyKey = ctx.channelId ?? ctx.channel ?? "";
        if (channelOnlyKey && channelOnlyKey !== conversationKey) {
          rememberConversationRouteMode(channelOnlyKey, normalized);
        }
        if (conversationKey) await persistRouteModePreference(routerUrl, conversationKey, normalized, "conversation");
        if (ctx.sessionKey) await persistRouteModePreference(routerUrl, ctx.sessionKey, normalized, "session");
        if (channelOnlyKey) await persistRouteModePreference(routerUrl, channelOnlyKey, normalized, "channel");
        return buildRouteInteractiveReply(normalized, "this conversation");
      },
    });

    api.on("before_dispatch", (event: any, ctx: any) => {
      const rawText = (event.body ?? event.content ?? "").trim();
      const sessionKey = ctx.sessionKey ?? event.sessionKey;
      const channel = ctx.channelId ?? event.channel ?? "unknown";
      const conversationKey = [channel, ctx.accountId ?? "default", ctx.conversationId ?? "", ""].join(":");

      // Skip slash commands — they are not user prompts and should not be routed.
      const isSlashCommand = rawText.startsWith("/");
      if (sessionKey && rawText && isSlashCommand) {
        recentSlashCommandBySession.set(sessionKey, { at: Date.now(), cmd: rawText });
      }
      if (sessionKey && rawText && !isSlashCommand) {
        rememberRecentUserMessage(sessionKey, rawText);
      }
      if (sessionKey && ctx.conversationId) {
        rememberConversationKeyForSession(sessionKey, conversationKey);
      }
      if (sessionKey && (ctx.senderId || event?.senderId)) {
        rememberSenderForSession(sessionKey, String(ctx.senderId ?? event.senderId ?? ""), String(ctx.channelId ?? event.channel ?? ""));
      }
      if (debugMode && sessionKey && rawText) {
        console.log(`[nexus-router] captured inbound text for ${sessionKey} (${rawText.length} chars)`);
      }
      return;
    });

    api.on("before_prompt_build", async (event: any, ctx: any) => {
      const sessionKey = ctx.sessionKey;
      if (!sessionKey) return;
      const context = buildConversationContextFromMessages(Array.isArray(event.messages) ? event.messages : []);
      if (context) {
        rememberConversationContextForSession(sessionKey, context);
      }
    });

    // Register the pre-model-resolve hook
    // Classifier prompt sentinels — used to detect recursive classifier calls.
    // We match multiple markers because prompt compilation can reformat text.
    const CLASSIFIER_SENTINELS = [
      "return only a json object",
      "message to classify",
      "task_type",
      "needs_tools",
      "needs_vision",
      "needs_long_context",
    ];

    api.on("before_model_resolve", async (event: any, ctx: any) => {
      const prompt = String(event.prompt ?? "");
      const normalizedPrompt = prompt.toLowerCase();

      // Guard: skip routing when this turn IS the classifier call itself.
      // Without this guard, codex exec triggers before_model_resolve → recursion loop.
      if (CLASSIFIER_SENTINELS.every((m) => normalizedPrompt.includes(m))) {
        return;
      }

      const sessionForCommandGuard = String(ctx?.sessionKey ?? "");
      if (sessionForCommandGuard) {
        const cmd = recentSlashCommandBySession.get(sessionForCommandGuard);
        if (cmd && Date.now() - cmd.at < RECENT_COMMAND_GUARD_MS) {
          await debugLog(`[hook-result] source=command route=skipped cmd=${cmd.cmd.slice(0, 80)}`);
          recentSlashCommandBySession.delete(sessionForCommandGuard);
          return;
        }
      }

      const rawUserText = takeRecentUserMessage(ctx.sessionKey) ?? "";
      const conversationContext = takeRecentConversationContext(ctx.sessionKey) ?? "";
      const routingText = rawUserText || prompt;
      const source = rawUserText ? "raw-user" : "compiled-prompt";
      const startupReason = source === "compiled-prompt" ? takeRecentStartupReason(ctx.sessionKey) : null;
      const sourceTag = buildSourceTag(ctx, source, routingText, startupReason);
      const routeMode = await resolveRouteModeFromContext(api, ctx);
      const firstPassCostProfile = resolveCostProfileForRouteMode(routeMode, costProfile);
      const sessionRef = String(ctx?.sessionKey ?? ctx?.sessionId ?? ctx?.conversationId ?? "");
      const dedupeText = routingText.trim();
      const shouldUseLlmClassifier = shouldUseContextualLlmClassifier(
        routeMode,
        conversationContext,
        routingText,
      );
      await debugLog(
        `[hook-enter] source=${source} source_tag=${sourceTag} trigger=${ctx?.trigger ?? "unknown"} route=${routeMode} prompt_len=${routingText.length} profile=${firstPassCostProfile}`,
      );

      if (ctx?.trigger === "cron" || classifySessionKind(ctx?.sessionKey) === "cron") {
        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
        }
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} bypassed reason=cron`);
        return;
      }

      if (ctx?.trigger === "heartbeat" || ctx?.trigger === "memory") {
        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
        }
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} bypassed reason=${ctx?.trigger}`);
        return;
      }

      if (routeMode === "off") {
        const shadowResult = await routeRequest(
          routerUrl,
          routingText,
          firstPassCostProfile,
          timeoutMs,
          routeMode,
          conversationContext,
          shouldUseLlmClassifier,
        );
        const shadowDecision = shadowResult.decision;
        if (!shadowDecision) {
          const failure = describeRouteRequestFailure(shadowResult, timeoutMs);
          await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} shadow_failed ${failure}`);
          return;
        }

        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, {
            text: dedupeText,
            mode: routeMode,
            at: Date.now(),
            selectedModel: shadowDecision.selected_model,
          });
        }

        const lastDecision: LastRouteDecision = {
          at: Date.now(),
          decisionId: shadowDecision.decision_id,
          requestedRouteMode: routeMode,
          routeMode,
          source,
          sourceTag,
          promptLen: routingText.length,
          promptText: routingText,
          replyContextUsed: shadowDecision.reply_context_used ?? Boolean(conversationContext.trim()),
          classifierSource: (shadowDecision.classifier_source ?? (shouldUseLlmClassifier ? "llm" : "heuristic")) as LastRouteDecision["classifierSource"],
          costProfile: firstPassCostProfile,
          taskType: shadowDecision.task_type,
          effectiveTaskType: extractEffectiveTaskType(shadowDecision.reason),
          confidence: shadowDecision.confidence,
          firstPassModel: shadowDecision.selected_model,
          firstPassProvider: shadowDecision.selected_provider,
          selectedModel: shadowDecision.selected_model,
          selectedProvider: shadowDecision.selected_provider,
          fallbacks: shadowDecision.fallbacks,
          score: shadowDecision.score,
          reason: [...shadowDecision.reason, "shadow_mode:route_off"],
          autoEscalated: false,
        };
        rememberLastDecision(ctx.sessionKey, lastDecision);

        if (ctx.sessionId && shadowDecision.decision_id) {
          enqueuePendingOutcome(ctx.sessionId, {
            decisionId: shadowDecision.decision_id,
            selectedModel: shadowDecision.selected_model,
            selectedProvider: shadowDecision.selected_provider,
            sessionKey: ctx.sessionKey,
            shadowMode: true,
            targetSenderId: resolveSenderForSession(ctx.sessionKey ?? sessionRef)?.senderId,
          });
        }

        await debugLog(
          `[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} shadow decision=${shadowDecision.selected_model} task=${shadowDecision.task_type} confidence=${shadowDecision.confidence.toFixed(2)} score=${shadowDecision.score.toFixed(3)}`,
        );
        return;
      }

      // Bypass routing on startup only when no explicit mode was set (auto = default).
      // If the user persisted an explicit mode (fast/reasoning/balanced), honor it even on the first post-startup turn.
      if (startupReason && routeMode === "auto") {
        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
        }
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} bypassed reason=${startupReason}`);
        return;
      }
      if (shouldBypassCompiledRetryRouting(ctx, source, ctx.sessionKey ?? sessionRef)) {
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} bypassed reason=compiled-retry`);
        return;
      }
      if (sessionRef && shouldBlockRoutingForBurst(sessionRef)) {
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=blocked burst session=${sessionRef}`);
        return;
      }

      const conversationKey = buildConversationKeyFromContext(ctx) ?? resolveConversationKeyForSession(ctx.sessionKey);
      const cached = sessionRef ? recentRouteCacheBySession.get(sessionRef) : undefined;
      if (
        sessionRef &&
        cached &&
        Date.now() - cached.at < ROUTE_DEDUPE_WINDOW_MS &&
        cached.mode === routeMode &&
        cached.text === dedupeText
      ) {
        if (cached.selectedModel) {
          const cachedOverride = splitSelectedModelRef(cached.selectedModel);
          const cachedProvider = cachedOverride.providerOverride;
          if (cachedProvider && getFailedOverrideBlock(ctx.sessionKey ?? sessionRef, conversationKey, cached.selectedModel, cachedProvider)) {
            recentRouteCacheBySession.delete(sessionRef);
            await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} dedupe-bypass blocked_model=${cached.selectedModel}`);
          } else {
            await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} dedupe-hit model=${cached.selectedModel}`);
            return cachedOverride;
          }
        } else {
          await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} dedupe-hit bypass`);
          return;
        }
      }

      let decision: RouteResponse | null = null;
      let autoDecision: RouteResponse | null = null;
      let finalMode: RouteMode = routeMode;
      let finalCostProfile = firstPassCostProfile;

      if (routeMode === "auto") {
        const autoResult = await routeRequest(
          routerUrl,
          routingText,
          firstPassCostProfile,
          timeoutMs,
          routeMode,
          conversationContext,
          shouldUseLlmClassifier,
        );
        autoDecision = autoResult.decision;
        if (!autoDecision) {
          const failure = describeRouteRequestFailure(autoResult, timeoutMs);
          await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} ${failure}`);
          if (debugMode) console.warn(`[nexus-router] ${failure}, using default model`);
          return;
        }

        if (shouldAutoEscalate(autoDecision.confidence)) {
          const balancedCostProfile = resolveCostProfileForRouteMode("balanced", costProfile);
          const balancedResult = await routeRequest(
            routerUrl,
            routingText,
            balancedCostProfile,
            timeoutMs,
            "balanced",
            conversationContext,
            shouldUseLlmClassifier,
          );
          const balancedDecision = balancedResult.decision;
          if (balancedDecision) {
            decision = balancedDecision;
            finalMode = "balanced";
            finalCostProfile = balancedCostProfile;
            await debugLog(
              `[hook-auto-escalate] source=${source} source_tag=${sourceTag} from=auto confidence=${autoDecision.confidence.toFixed(2)} to=balanced selected=${balancedDecision.selected_model}`,
            );
          } else {
            if (balancedResult.error) {
              const failure = describeRouteRequestFailure(balancedResult, timeoutMs);
              await debugLog(`[hook-auto-escalate] source=${source} source_tag=${sourceTag} balanced_fallback_failed ${failure}`);
            }
            decision = autoDecision;
          }
        } else {
          decision = autoDecision;
        }
      } else {
        const directResult = await routeRequest(
          routerUrl,
          routingText,
          firstPassCostProfile,
          timeoutMs,
          routeMode,
          conversationContext,
          shouldUseLlmClassifier,
        );
        decision = directResult.decision;
        if (!decision) {
          const failure = describeRouteRequestFailure(directResult, timeoutMs);
          await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} ${failure}`);
          if (debugMode) console.warn(`[nexus-router] ${failure}, using default model`);
          return;
        }
      }

      if (!decision) {
        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
        }
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} router_unavailable`);
        if (debugMode) console.warn("[nexus-router] router unavailable, using default model");
        return;
      }

      const blockedOverride = getFailedOverrideBlock(
        ctx.sessionKey ?? sessionRef,
        conversationKey,
        decision.selected_model,
        decision.selected_provider,
      );
      if (blockedOverride) {
        const fallbackModel = chooseUnblockedFallback(decision.fallbacks, ctx.sessionKey ?? sessionRef, conversationKey);
        if (fallbackModel) {
          const fallbackOverride = splitSelectedModelRef(fallbackModel);
          await debugLog(
            `[hook-result] source=${source} source_tag=${sourceTag} route=${finalMode} blocked model=${decision.selected_model} fallback=${fallbackModel} reason=${blockedOverride.reason}`,
          );
          decision = {
            ...decision,
            selected_model: fallbackModel,
            selected_provider: fallbackOverride.providerOverride ?? decision.selected_provider,
            reason: [...decision.reason, `blocked recent failed override ${blockedOverride.reason}`],
          };
        } else {
          if (sessionRef) {
            recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
          }
          await debugLog(
            `[hook-result] source=${source} source_tag=${sourceTag} route=${finalMode} blocked model=${decision.selected_model} fallback=none reason=${blockedOverride.reason}`,
          );
          return;
        }
      }

      if (sessionRef) {
        recentRouteCacheBySession.set(sessionRef, {
          text: dedupeText,
          mode: routeMode,
          at: Date.now(),
          selectedModel: decision.selected_model,
        });
      }

      const sender = resolveSenderForSession(ctx.sessionKey ?? sessionRef);
      if (decision.decision_id) {
        if (sender) {
          await sendTelegramFeedbackCard(api, sender.senderId, decision, sourceTag);
        } else {
          await debugLog(
            `[feedback-card] skipped decision=${decision.decision_id} reason=missing_sender session=${ctx.sessionKey ?? sessionRef ?? ""}`,
          );
        }
      }

      if (decision.confidence < minConfidence) {
        await debugLog(
          `[hook-result] source=${source} source_tag=${sourceTag} skipped confidence=${decision.confidence.toFixed(2)} threshold=${minConfidence}`,
        );
        if (debugMode) {
          console.log(
            `[nexus-router] confidence ${decision.confidence.toFixed(2)} < ${minConfidence}, skipping`,
          );
        }
        return;
      }

      const lastDecision: LastRouteDecision = {
        at: Date.now(),
        decisionId: decision.decision_id,
        requestedRouteMode: routeMode,
        routeMode: finalMode,
        source,
        sourceTag,
        promptLen: routingText.length,
        promptText: routingText,
        replyContextUsed: decision.reply_context_used ?? Boolean(conversationContext.trim()),
        classifierSource: (decision.classifier_source ?? (shouldUseLlmClassifier ? "llm" : "heuristic")) as LastRouteDecision["classifierSource"],
        costProfile: finalCostProfile,
        taskType: decision.task_type,
        effectiveTaskType: extractEffectiveTaskType(decision.reason),
        confidence: decision.confidence,
        firstPassModel: routeMode === "auto" && autoDecision ? autoDecision.selected_model : decision.selected_model,
        firstPassProvider: routeMode === "auto" && autoDecision ? autoDecision.selected_provider : decision.selected_provider,
        selectedModel: decision.selected_model,
        selectedProvider: decision.selected_provider,
        fallbacks: decision.fallbacks,
        score: decision.score,
        reason: decision.reason,
        autoEscalated: routeMode === "auto" && finalMode === "balanced",
      };
      rememberLastDecision(ctx.sessionKey, lastDecision);

      if (ctx.sessionId && decision.decision_id) {
        enqueuePendingOutcome(ctx.sessionId, {
          decisionId: decision.decision_id,
          selectedModel: decision.selected_model,
          selectedProvider: decision.selected_provider,
          sessionKey: ctx.sessionKey,
        });
      }

      await debugLog(
        `[hook-result] source=${source} source_tag=${sourceTag} route=${finalMode} override model=${decision.selected_model} task=${decision.task_type} confidence=${decision.confidence.toFixed(2)} score=${decision.score.toFixed(3)}`,
      );

      if (debugMode) {
        console.log(
          `[nexus-router] ${decision.task_type} → ${decision.selected_model}` +
          ` (score=${decision.score.toFixed(3)}, confidence=${decision.confidence.toFixed(2)}, mode=${finalMode})`,
        );
        console.log(`[nexus-router] reason: ${decision.reason.join("; ")}`);
      }

      return splitSelectedModelRef(decision.selected_model);
    });

    api.on("llm_output", async (event: any, ctx: any) => {
      const pending = peekPendingOutcome(event.sessionId);

      const byDecisionId = pending?.decisionId
        ? recentLastDecisionByDecisionId.get(pending.decisionId)
        : undefined;
      const bySessionKey = (pending?.sessionKey ?? ctx.sessionKey)
        ? recentLastDecisionBySession.get((pending?.sessionKey ?? ctx.sessionKey) as string)
        : undefined;
      const last = byDecisionId ?? bySessionKey ?? recentLastDecisionGlobal;
      if (!last) return;

      last.actualProvider = event.provider;
      last.actualModel = event.model;
      last.usage = {
        input: event.usage?.input,
        output: event.usage?.output,
        total: event.usage?.total,
      };
    });

    api.on("agent_end", async (event: any, ctx: any) => {
      const pending = ctx.sessionId ? shiftPendingOutcome(ctx.sessionId) : undefined;
      if (!pending) return;

      try {
        const byDecisionId = recentLastDecisionByDecisionId.get(pending.decisionId);
        const bySessionKey = (pending.sessionKey ?? ctx.sessionKey)
          ? recentLastDecisionBySession.get((pending.sessionKey ?? ctx.sessionKey) as string)
          : undefined;
        const last = byDecisionId ?? bySessionKey ?? recentLastDecisionGlobal;
        if (last) {
          last.runtimeSuccess = Boolean(event.success);
          last.runtimeDurationMs = event.durationMs;
          last.runtimeError = typeof event.error === "string" ? event.error : undefined;
        }
        const actualModel = last?.actualModel;
        const actualProvider = last?.actualProvider;
        const fallbackUsed = !!actualModel && actualModel !== pending.selectedModel;
        const failure = inferFailureFromRuntime(event);
        const outcomeProvider = actualProvider ?? pending.selectedProvider;
        const outcomeModel = actualModel ?? pending.selectedModel;
        const failedConversationKey = resolveConversationKeyForSession(pending.sessionKey)
          ?? (pending.sessionKey ? buildConversationKeyFromContext({ sessionKey: pending.sessionKey }) : null);

        if (failure.shouldCooldownOverride && !event.success) {
          rememberFailedOverride(
            pending.sessionKey ?? ctx.sessionKey,
            failedConversationKey,
            outcomeModel,
            outcomeProvider,
            failure.errorType ?? "unknown",
          );
        }

        await fetch(`${routerUrl}/outcome`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            decision_id: pending.decisionId,
            success: event.success,
            latency_ms: event.durationMs,
            fallback_used: fallbackUsed,
            fallback_model: fallbackUsed ? actualModel : undefined,
            provider: outcomeProvider,
            http_status: failure.httpStatus,
            error_type: failure.errorType,
            quota_hint: failure.quotaHint,
            quota_remaining_ratio: failure.quotaRemainingRatio,
          }),
        });

        if (pending.shadowMode) {
          const decision = last;
          const sender = pending.targetSenderId
            ? { senderId: pending.targetSenderId }
            : resolveSenderForSession(pending.sessionKey ?? ctx.sessionKey);
          if (decision?.decisionId && sender?.senderId) {
            await sendTelegramFeedbackCard(
              api,
              sender.senderId,
              {
                decision_id: decision.decisionId,
                task_type: decision.taskType,
                confidence: decision.confidence,
                selected_model: decision.selectedModel,
                selected_provider: decision.selectedProvider,
                fallbacks: decision.fallbacks,
                score: decision.score,
                reason: decision.reason,
                classifier_source: decision.classifierSource,
                reply_context_used: decision.replyContextUsed,
              },
              decision.sourceTag,
              { shadowMode: true, actualModel: outcomeModel },
            );
          } else {
            await debugLog(
              `[feedback-card] shadow skipped decision=${pending.decisionId} reason=${decision ? 'missing_sender' : 'missing_decision'}`,
            );
          }
        }
      } catch {
        // best effort; keep routing path non-blocking
      }
    });

    if (debugMode) {
      console.log(`[nexus-router] registered, router=${routerUrl}, profile=${costProfile}`);
    }
  },
});
