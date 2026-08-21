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
  bridgeUrl?: string;
  bridgeBearerToken?: string;
  enabled?: boolean;
  costProfile?: "cheap" | "balanced" | "premium";
  debugMode?: boolean;
  minConfidence?: number;
  timeoutMs?: number;
}

type RouteMode = "auto" | "balanced" | "fast" | "reasoning" | "eco" | "free" | "off";

interface RouteResponse {
  decision_id?: string;
  task_type: string;
  confidence: number;
  selected_model: string;
  selected_provider: string;
  fallbacks: string[];
  excluded_models?: RouteExcludedModel[];
  score: number;
  reason: string[];
  classifier_source?: "explicit" | "local" | "heuristic" | "llm" | "fallback";
  reply_context_used?: boolean;
}

interface RouteExcludedModel {
  model: string;
  reason?: string;
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
  applied?: boolean;
  skippedReason?: string;
  fallbacks: string[];
  excludedModels: RouteExcludedModel[];
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
  free_filter?: boolean;
  updated_at: string;
}

// ── Defaults ──────────────────────────────────────────────────────────────────

const DEFAULT_URL         = "http://127.0.0.1:7771";
const DEFAULT_BRIDGE_URL  = "http://127.0.0.1:8091";
const DEFAULT_CONFIDENCE  = 0.60;
const DEFAULT_TIMEOUT_MS  = 10000;
const SHADOW_TIMEOUT_MS   = 60000;
const DEFAULT_COST        = "balanced";
const PLUGIN_VERSION      = "0.1.0";
const RECENT_MESSAGE_TTL_MS = 5 * 60 * 1000;
const AUTO_ESCALATE_CONFIDENCE = 0.70;
const ROUTE_MODES = new Set(["auto", "balanced", "fast", "reasoning", "eco", "off"]);
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
  sourceType?: "compiled-prompt" | "raw-user",
  sourceTag?: string,
  provenanceMode?: "route" | "shadow",
  freeFilter?: boolean,
): Promise<RouteRequestResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${url}/route`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Editor-Version": PLUGIN_VERSION,
        "Connection": "close",
      },
      body: JSON.stringify({
        message: prompt,
        cost_profile: costProfile,
        route_mode: routeMode,
        mode: provenanceMode ?? "route",
        provenance_mode: provenanceMode ?? "route",
        source_type: sourceType,
        source_tag: sourceTag,
        conversation_context: conversationContext,
        use_llm_classifier: useLlmClassifier ?? false,
        free_only: freeFilter ?? false,
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
  freeFilter?: boolean;
}

interface RouteModeResolution {
  mode: RouteMode;
  source: "session" | "conversation" | "channel" | "default";
  key?: string;
  freeFilter?: boolean;
}

const recentRouteModes = new Map<string, RouteModeEntry>();
const recentConversationRouteModes = new Map<string, RouteModeEntry>();
const recentConversationKeyBySession = new Map<string, { conversationKey: string; at: number }>();
const recentSessionKeyByConversation = new Map<string, { sessionKey: string; at: number }>();
const recentConversationContextBySession = new Map<string, { context: string; at: number }>();
const recentConversationTextsByConversation = new Map<string, { texts: string[]; at: number }>();
const recentLastDecisionBySession = new Map<string, LastRouteDecision>();
const recentLastDecisionByConversation = new Map<string, LastRouteDecision>();
const recentLastDecisionByDecisionId = new Map<string, LastRouteDecision>();
const pendingOutcomeQueueBySessionId = new Map<string, PendingOutcome[]>();
const recentRouteCacheBySession = new Map<string, { text: string; mode: RouteMode; at: number; selectedModel?: string }>();
const recentSenderBySession = new Map<string, { senderId: string; channelId?: string; at: number }>();
const recentFeedbackPromptByDecisionId = new Map<string, { at: number }>();
const recentFeedbackSuppressedBySession = new Map<string, { at: number; suppressed: boolean }>();

function rememberFeedbackSuppressed(sessionKey: string, suppressed: boolean): void {
  if (!sessionKey) return;
  recentFeedbackSuppressedBySession.set(sessionKey, { at: Date.now(), suppressed });
}

function isFeedbackSuppressed(sessionKey?: string): boolean {
  if (!sessionKey) return false;
  const entry = recentFeedbackSuppressedBySession.get(sessionKey);
  if (!entry) return false;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentFeedbackSuppressedBySession.delete(sessionKey);
    return false;
  }
  return entry.suppressed;
}

const routeBurstBySession = new Map<string, { windowStart: number; count: number; blockedUntil?: number }>();
const recentSlashCommandBySession = new Map<string, { at: number; cmd: string }>();
const recentStartupBySession = new Map<string, { at: number; reason: string }>();
const recentFailedOverrides = new Map<string, { at: number; blockedUntil: number; reason: string }>();
const RECENT_COMMAND_GUARD_MS = 15_000;
let recentLastDecisionGlobal: LastRouteDecision | null = null;

function rememberRecentUserMessage(sessionKey: string, text: string): void {
  const trimmed = stripOpenClawMetadataEnvelope(text);
  if (!sessionKey || !trimmed) return;
  recentUserMessages.set(sessionKey, { text: trimmed, at: Date.now() });
}

function rememberRecentConversationText(conversationKey: string, text: string): string | null {
  const trimmed = stripOpenClawMetadataEnvelope(text).trim();
  if (!conversationKey || !trimmed) return null;

  const now = Date.now();
  const existing = recentConversationTextsByConversation.get(conversationKey);
  const previousTexts = existing && now - existing.at <= RECENT_MESSAGE_TTL_MS
    ? existing.texts
    : [];
  const previousContext = previousTexts.join("\n").slice(-2000) || null;
  const nextTexts = [...previousTexts, trimmed].slice(-8);
  recentConversationTextsByConversation.set(conversationKey, { texts: nextTexts, at: now });
  return previousContext;
}

// Route mode is an explicit user preference. Keep it sticky for the life of the
// session/conversation instead of silently expiring back to the default.
const STICKY_ROUTE_MODES = new Set<RouteMode>(["auto", "balanced", "fast", "reasoning", "eco", "off"]);

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
  if (routeMode === "fast") return false;
  if (isShortFollowUpForContextualRouting(routingText)) return false;
  return true;
}

function rememberRouteMode(sessionKey: string, mode: RouteMode, freeFilter?: boolean): void {
  if (!sessionKey || !ROUTE_MODES.has(mode)) return;
  recentRouteModes.set(sessionKey, {
    mode,
    at: Date.now(),
    sticky: STICKY_ROUTE_MODES.has(mode),
    freeFilter: freeFilter ?? false,
  });
}

function rememberConversationRouteMode(conversationKey: string, mode: RouteMode, freeFilter?: boolean): void {
  if (!conversationKey || !ROUTE_MODES.has(mode)) return;
  recentConversationRouteModes.set(conversationKey, {
    mode,
    at: Date.now(),
    sticky: STICKY_ROUTE_MODES.has(mode),
    freeFilter: freeFilter ?? false,
  });
}

async function persistRouteModePreference(routerUrl: string, key: string, mode: RouteMode, scope: "conversation" | "session" | "channel" = "conversation", freeFilter = false): Promise<void> {
  const trimmedKey = key.trim();
  if (!trimmedKey) return;
  try {
    await fetch(`${routerUrl}/route-mode`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Editor-Version": PLUGIN_VERSION,
      },
      body: JSON.stringify({ key: trimmedKey, mode, scope, free_filter: freeFilter }),
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
  // Initial card only asks Correct/Wrong — Wrong triggers a follow-up prompt
  // asking for the correct task and model verdict (matches nexus-arc service).
  return [
    [
      { text: "✅ Correct", callback_data: `routefb:ok:${decisionId}` },
      { text: "❌ Wrong", callback_data: `routefb:wrong:${decisionId}` },
    ],
  ];
}

const FEEDBACK_TASK_LABELS = ["coding", "code_review", "reasoning", "general_chat"] as const;
const FEEDBACK_MODEL_VERDICTS: Record<string, string> = {
  too_cheap: "🔼 Too cheap/fast",
  ok: "✅ Model OK",
  too_powerful: "🔽 Too powerful/slow",
};

function buildWrongTaskKeyboard(decisionId: string): Array<Array<{ text: string; callback_data: string }>> {
  const buttons: Array<Array<{ text: string; callback_data: string }>> = [];
  let row: Array<{ text: string; callback_data: string }> = [];
  for (const label of FEEDBACK_TASK_LABELS) {
    row.push({ text: label, callback_data: `routefb:wrong_task:${decisionId}:${label}` });
    if (row.length === 2) {
      buttons.push(row);
      row = [];
    }
  }
  if (row.length) buttons.push(row);
  buttons.push([
    { text: "⬅️ Back", callback_data: `routefb:back:${decisionId}:initial` },
    { text: "⏭ Skip", callback_data: `routefb:wrong_task:${decisionId}:skip` },
  ]);
  return buttons;
}

function buildWrongModelKeyboard(decisionId: string, taskSlot: string): Array<Array<{ text: string; callback_data: string }>> {
  const slot = taskSlot || "skip";
  return [
    Object.entries(FEEDBACK_MODEL_VERDICTS).map(([key, label]) => ({
      text: label,
      callback_data: `routefb:wrong_model:${decisionId}:${slot}:${key}`,
    })),
    [
      { text: "⬅️ Back", callback_data: `routefb:back:${decisionId}:wrong_task` },
      { text: "⏭ Skip", callback_data: `routefb:wrong_model:${decisionId}:${slot}:skip` },
    ],
  ];
}

function summarizeExcludedModels(excludedModels?: RouteExcludedModel[], limit = 3): string {
  const items = (excludedModels ?? [])
    .filter((item) => String(item?.model ?? "").trim())
    .map((item) => ({
      model: String(item.model).trim(),
      reason: String(item.reason ?? "").trim(),
    }));
  if (!items.length) return "none";

  const codexItems = items.filter((item) => /\b(?:codex|openai)\//i.test(item.model));
  const selected = (codexItems.length ? codexItems : items).slice(0, limit);
  return selected
    .map((item) => item.reason ? `${item.model} (${item.reason})` : item.model)
    .join(", ");
}

/**
 * Strip OpenClaw injected metadata envelopes from a raw user message before
 * routing. These blocks are injected by the runtime and should never reach
 * the classifier or appear in feedback-card previews.
 *
 * Strips:
 *   - Conversation info (untrusted metadata): ```json ... ```
 *   - Sender (untrusted metadata): ```json ... ```
 *   - Replied message (untrusted, for context): ```json ... ```
 *   - Any leading/trailing whitespace after stripping.
 */
export function stripOpenClawMetadataEnvelope(text: string): string {
  // Remove all fenced-json blocks that follow an OpenClaw envelope header
  // Pattern: optional header line + ```json ... ``` block
  let stripped = text
    .replace(
      /(?:^|\n)(?:Conversation info|Sender|Replied message|Inbound Context|Group Chat Context)\s*\([^)]*\):[^\n]*\n```json[\s\S]*?```/gi,
      "",
    )
    // Also strip bare ```json blocks that look like openclaw metadata (heuristic: contain "message_id", "sender_id", "schema")
    .replace(
      /```json\s*\{[\s\S]*?(?:"message_id"|"sender_id"|"schema":\s*"openclaw)[\s\S]*?```/gi,
      "",
    );
  // Collapse runs of blank lines
  stripped = stripped.replace(/\n{3,}/g, "\n\n").trim();
  return stripped;
}

function shouldSuppressFeedbackCardForText(messagePreview?: string): boolean {
  const text = String(messagePreview ?? "").trim();
  if (!text) return false;

  const lowered = text.toLowerCase();
  const internalMarkers = [
    "<<<begin_openclaw_internal_context>>>",
    "<<<end_openclaw_internal_context>>>",
    "[subagent context]",
    "[subagent task]",
    "requester session:",
    "requester channel:",
    "results auto-announce to your requester",
    "completion is push-based",
    "do not busy-poll for status",
    // OpenClaw injected metadata envelopes
    "conversation info (untrusted metadata)",
    "sender (untrusted metadata)",
    "replied message (untrusted",
    "inbound context (trusted metadata)",
  ];
  if (internalMarkers.some((marker) => lowered.includes(marker))) {
    return true;
  }

  const looksLikeApprovalRelay =
    lowered.includes("/approve")
    || lowered.includes("approval-pending")
    || lowered.includes("approval required")
    || lowered.includes("allow-once")
    || lowered.includes("approve what will actually run")
    || lowered.includes("native approval card")
    || lowered.includes("chat approvals are unavailable");

  if (looksLikeApprovalRelay) {
    return true;
  }

  return false;
}

async function sendTelegramFeedbackCard(
  api: any,
  targetSenderId: string,
  decision: RouteResponse,
  sourceTag: string,
  opts?: { shadowMode?: boolean; actualModel?: string; messagePreview?: string; sessionKey?: string },
): Promise<boolean> {
  const sessionKey = opts?.sessionKey;
  if (sessionKey && isFeedbackSuppressed(sessionKey)) {
    await debugLog('[feedback-card] suppressed by /route feedback off');
    return false;
  }
  const decisionId = String(decision.decision_id || "").trim();
  if (!decisionId || hasRecentFeedbackPrompt(decisionId)) {
    return false;
  }

  if (shouldSuppressFeedbackCardForText(opts?.messagePreview)) {
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
  const excludedSummary = summarizeExcludedModels(decision.excluded_models, 2);
  const lines = shadowMode
    ? [
        `🧭 shadow ${task} · proposed ${model} · conf ${confidence}`,
        `Actual reply model: ${actualModel || "unknown"}`,
        ...(excludedSummary !== "none" ? [`Excluded: ${excludedSummary}`] : []),
        `Source: ${sourceTag}`,
        `Feedback?`,
      ]
    : [
        `🧭 ${task} · ${model} · ${confidence}`,
        ...(excludedSummary !== "none" ? [`Excluded: ${excludedSummary}`] : []),
        `Source: ${sourceTag}`,
        `Feedback?`,
      ];
  const text = lines.join("\n");

  try {
    const telegram = api?.runtime?.telegram;
    if (telegram?.sendMessageTelegram) {
      try {
        const result = await telegram.sendMessageTelegram(targetSenderId, text, {
          buttons: buildFeedbackKeyboard(decisionId),
          textMode: "plain",
          cfg: api?.config?.loadConfig?.(),
        });
        rememberFeedbackPrompt(decisionId);
        await debugLog(`[feedback-card] sent decision=${decisionId} to=${targetSenderId} message_id=${result?.messageId ?? "?"} via=telegram_runtime`);
        return true;
      } catch (err) {
        await debugLog(`[feedback-card] telegram_runtime failed decision=${decisionId} to=${targetSenderId} error=${(err as any)?.message ?? String(err)}; falling back to bridge`);
        // fall through to bridge path
      }
    }

    const bridgePayload = {
      telegram_user_id: targetSenderId,
      decision_id: decisionId,
      task_type: task,
      selected_model: model,
      confidence: Number.isFinite(decision.confidence) ? decision.confidence : undefined,
      classifier_source: decision.classifier_source,
      excluded_models: decision.excluded_models,
      provenance_mode: shadowMode ? "shadow" : "route",
      actual_model: actualModel || undefined,
      source_channel: "telegram",
      source_message_preview: opts?.messagePreview,
    };

    const runtimePluginConfig = ((api as any)?.config?.plugins?.entries?.["nexus-router"]?.config ?? {}) as NexusRouterConfig;
    const loadedConfig = (api as any)?.config?.loadConfig?.();
    const livePluginConfig = (loadedConfig?.plugins?.entries?.["nexus-router"]?.config ?? {}) as NexusRouterConfig;
    const bridgeBearerToken = livePluginConfig.bridgeBearerToken ?? runtimePluginConfig.bridgeBearerToken;
    const bridgeHeaders: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (bridgeBearerToken) {
      bridgeHeaders["Authorization"] = `Bearer ${bridgeBearerToken}`;
    }

    const bridgeBaseUrl = livePluginConfig.bridgeUrl ?? runtimePluginConfig.bridgeUrl ?? DEFAULT_BRIDGE_URL;
    const bridgeRes = await fetch(`${bridgeBaseUrl}/api/v1/router/feedback-card`, {
      method: "POST",
      headers: bridgeHeaders,
      body: JSON.stringify(bridgePayload),
    });

    if (!bridgeRes.ok) {
      await debugLog(`[feedback-card] failed decision=${decisionId} to=${targetSenderId} error=bridge_http_${bridgeRes.status}`);
      return false;
    }

    const bridgeJson = await bridgeRes.json().catch(() => ({} as any));
    if (!bridgeJson?.ok) {
      await debugLog(`[feedback-card] failed decision=${decisionId} to=${targetSenderId} error=bridge_not_ok`);
      return false;
    }

    rememberFeedbackPrompt(decisionId);
    await debugLog(`[feedback-card] sent decision=${decisionId} to=${targetSenderId} via=bridge`);
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

function classifySessionKind(sessionKey?: string): "cron" | "dreaming" | "subagent" | "direct" | "slash" | "main" | "other" {
  const key = (sessionKey ?? "").toLowerCase();
  if (!key) return "other";
  if (key.startsWith("cron:") || key.includes(":cron:")) return "cron";
  if (key.startsWith("dreaming-narrative-") || key.includes(":dreaming-narrative-")) return "dreaming";
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
  if (classifySessionKind(ctx?.sessionKey) === "dreaming") {
    return "memory:dreaming";
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
  recentConversationTextsByConversation.clear();
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

type FailedOverrideScope = "model" | "provider" | "both";

function buildFailedOverrideKeys(
  sessionKey: string | undefined,
  conversationKey: string | null,
  selectedModel: string,
  selectedProvider: string,
  scope: FailedOverrideScope = "both",
): string[] {
  const keys = new Set<string>();
  const model = selectedModel.trim();
  const provider = selectedProvider.trim();
  if (sessionKey) {
    if (scope === "model" || scope === "both") keys.add(`session:${sessionKey}:model:${model}`);
    if (scope === "provider" || scope === "both") keys.add(`session:${sessionKey}:provider:${provider}`);
  }
  if (conversationKey) {
    if (scope === "model" || scope === "both") keys.add(`conversation:${conversationKey}:model:${model}`);
    if (scope === "provider" || scope === "both") keys.add(`conversation:${conversationKey}:provider:${provider}`);
  }
  return Array.from(keys);
}

function failedOverrideScopeForReason(reason: string): FailedOverrideScope {
  return reason === "timeout" ? "model" : "both";
}

function rememberFailedOverride(sessionKey: string | undefined, conversationKey: string | null, selectedModel: string, selectedProvider: string, reason: string): void {
  const now = Date.now();
  const blockedUntil = now + FAILED_OVERRIDE_COOLDOWN_MS;
  const scope = failedOverrideScopeForReason(reason);
  for (const key of buildFailedOverrideKeys(sessionKey, conversationKey, selectedModel, selectedProvider, scope)) {
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
  const match = lowered.match(/^(?:⚙️\s*)?routing mode(?:\s*(?:set to|:|=)\s*|\s+)(auto|balanced|fast|reasoning|eco|free|off)(?:\s*\([^)]*\))?\.?$/i);
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
  const excluded = summarizeExcludedModels(last.excludedModels);
  const reason = last.reason.length ? last.reason.slice(0, 3).join("; ") : "n/a";
  const contextLabel = last.replyContextUsed ? "reply-context used: yes" : "reply-context used: no";
  const escalationLabel = last.autoEscalated ? `${last.requestedRouteMode} → ${last.routeMode}` : last.routeMode;

  const lines = [
    `Routing: ${escalationLabel}`,
    `Applied: ${last.applied === false ? "no" : "yes"}`,
    ...(last.applied === false && last.skippedReason ? [`Skip reason: ${last.skippedReason}`] : []),
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
    `Excluded: ${excluded}`,
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
  const excluded = summarizeExcludedModels(last.excludedModels, 6);
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
    `Applied: ${last.applied === false ? "no" : "yes"}`,
    ...(last.applied === false && last.skippedReason ? [`Skip reason: ${last.skippedReason}`] : []),
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
    `Excluded: ${excluded}`,
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
      return defaultProfile;
    case "balanced":
      return defaultProfile;
    case "fast":
      return "cheap";
    case "reasoning":
      return "premium";
    case "eco":
      return "cheap";
    case "free":
      return "cheap";
    case "off":
      return defaultProfile;
  }
}

function buildRouteInteractiveReply(
  mode?: RouteMode,
  scopeLabel = "this conversation",
  freeFilter?: boolean,
  includeInteractive = true,
): {
  text: string;
  interactive?: { blocks: Array<{ type: "text"; text: string } | { type: "buttons"; buttons: Array<{ label: string; value: string; style?: "primary" | "secondary" | "success" | "danger" }> }> };
} {
  const label = `${mode ?? "auto"}${freeFilter && mode !== "free" ? " free" : ""}`;
  return {
    text: `⚙️ Routing mode: ${label} (${scopeLabel}).`,
    ...(includeInteractive
      ? {
          interactive: {
            blocks: [
              { type: "text", text: `Choose a routing mode (current: ${label}, persisted in router state):` },
              {
                type: "buttons",
                buttons: [
                  { label: "Auto", value: "tgcmd:/route auto", style: "primary" },
                  { label: "Balanced", value: "tgcmd:/route balanced", style: "secondary" },
                  { label: "Fast", value: "tgcmd:/route fast", style: "success" },
                  { label: "Reasoning", value: "tgcmd:/route reasoning", style: "secondary" },
                  { label: "Eco", value: "tgcmd:/route eco", style: "secondary" },
                  { label: "Free", value: "tgcmd:/route free", style: "secondary" },
                  { label: "Off", value: "tgcmd:/route off", style: "danger" },
                ],
              },
            ],
          },
        }
      : {}),
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
    `- eco: bias toward more efficient/lower-footprint models`,
    `Modifiers:`,
    `- free: only consider models marked is_free=true (combine with any mode)`,
    `- off: bypass router overrides`,
    ``,
    `Commands:`,
    `- /route status → show current session mode`,
    `- /route last → show last routing decision (short form)`,
    `- /route explain → show richer diagnostics (context + escalation + classifier source)`,
    `- /route feedback → request a feedback card for the last routing decision`,
`- /route feedback on|off → enable/disable feedback cards for this session`,
    `- /route compare [fast balanced reasoning eco] → compare modes on the last prompt`,
    ``,
    `Examples:`,
    `- /route fast`,
    `- /route reasoning`,
    `- /route eco`,
    `- /route auto free`,
    `- /route free`,
    `  (shorthand for /route auto free)`,
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
    candidates.push({ mode: entry.mode, source, key, at: entry.at, freeFilter: entry.freeFilter });
  };

  const addSessionCandidate = (key?: string): void => {
    if (!key || seen.has(`session:${key}`)) return;
    seen.add(`session:${key}`);
    const entry = getRecentRouteModeEntry(key);
    if (!entry) return;
    candidates.push({ mode: entry.mode, source: "session", key, at: entry.at, freeFilter: entry.freeFilter });
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
      candidates.push({ mode: persistedConversation.mode, source: "conversation", key: conversationKey, at: Date.parse(persistedConversation.updated_at) || 0, freeFilter: Boolean(persistedConversation.free_filter) });
    }
  }
  if (ctx?.sessionKey) {
    const persistedSession = await loadPersistedRouteModePreference(routerUrl, ctx.sessionKey, "session");
    if (persistedSession) {
      candidates.push({ mode: persistedSession.mode, source: "session", key: ctx.sessionKey, at: Date.parse(persistedSession.updated_at) || 0, freeFilter: Boolean(persistedSession.free_filter) });
    }
  }
  if (channelKey) {
    const persistedChannel = await loadPersistedRouteModePreference(routerUrl, channelKey, "channel");
    if (persistedChannel) {
      candidates.push({ mode: persistedChannel.mode, source: "channel", key: channelKey, at: Date.parse(persistedChannel.updated_at) || 0, freeFilter: Boolean(persistedChannel.free_filter) });
    }
  }

  candidates.sort((a, b) => b.at - a.at);
  const resolved = candidates[0];
  if (!resolved) {
    return { mode: "auto", source: "default" };
  }

  if (ctx?.sessionKey && resolved.source !== "session") {
    rememberRouteMode(ctx.sessionKey, resolved.mode, resolved.freeFilter);
  }

  return {
    mode: resolved.mode,
    source: resolved.source,
    key: resolved.key,
    freeFilter: resolved.freeFilter,
  };
}

function resolveFreeFilterFromContext(ctx: any): boolean {
  const sessionEntry = ctx?.sessionKey ? getRecentRouteModeEntry(ctx.sessionKey) : null;
  if (sessionEntry?.freeFilter) return true;
  const conversationKey = buildConversationKeyFromContext(ctx);
  if (conversationKey) {
    const convEntry = getRecentConversationRouteModeEntry(conversationKey);
    if (convEntry?.freeFilter) return true;
  }
  return false;
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
  rememberRecentConversationText,
  takeRecentConversationContext,
  rememberConversationContextForSession,
  resolveRouteModeDetailsFromContext,
  shouldUseContextualLlmClassifier,
  shouldBypassCompiledRetryRouting,
  classifySessionKind,
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
      const tokens = loweredArgs.split(/\s+/).filter(Boolean);
      
      const firstToken = tokens[0] ?? "";
      const secondToken = tokens[1] ?? "";
      
      // Parse: /route [mode] [free] or /route feedback on|off
      const hasFreeModifier = tokens.includes('free');
      const modeToken = tokens.find((t: string) => ROUTE_MODES.has(t)) ?? '';
      const arg = firstToken;  // Keep for compatibility with existing checks
      
      const conversationKey = buildConversationKeyFromContext(ctx) ?? [ctx.channelId ?? ctx.channel, ctx.accountId ?? "default", ctx.from ?? ctx.to ?? "", ctx.messageThreadId ?? ""].join(":");
      const sessionKeyForConversation = conversationKey ? resolveSessionKeyForConversation(conversationKey) ?? undefined : undefined;
      const modeResolution = await resolveRouteModeDetailsFromContext(api, ctx);
      const currentMode = modeResolution.mode;
      
      // Handle /route feedback on|off FIRST
      if (firstToken === 'feedback' && (secondToken === 'on' || secondToken === 'off')) {
        const suppressed = secondToken === 'off';
        rememberFeedbackSuppressed(ctx.sessionKey, suppressed);
        const currentState = isFeedbackSuppressed(ctx.sessionKey) ? 'OFF' : 'ON';
        return { text: (suppressed ? '⛔ Feedback cards disabled for this session.' : '✅ Feedback cards enabled for this session.') + ` Current: ${currentState}.` };
      }
      
      if (!firstToken || firstToken === "help" || firstToken === "?") {
        return { text: buildRouteHelpText(currentMode), interactive: buildRouteInteractiveReply(currentMode, "this conversation", modeResolution.freeFilter).interactive };
      }
      if (firstToken === "status") {
        const scopeLabel = modeResolution.source === "default" ? "default" : `resolved from ${modeResolution.source}`;
        return buildRouteInteractiveReply(currentMode, scopeLabel, modeResolution.freeFilter);
      }

      // Handle /route free as shorthand for /route auto free
      const effectiveMode = hasFreeModifier && !modeToken ? 'auto' : modeToken;
      const normalized = effectiveMode && ROUTE_MODES.has(effectiveMode) ? (effectiveMode as RouteMode) : undefined;

      if (arg === "last") {
          const last = resolveLastDecisionForContext(ctx, conversationKey);
          return buildRouteLastReply(last);
        }

    if (arg === "feedback") {
      const last = resolveLastDecisionForContext(ctx, conversationKey);
      const suppressed = isFeedbackSuppressed(ctx.sessionKey);
      const statusText = suppressed ? 'Feedback cards are currently OFF for this session.' : 'Feedback cards are currently ON for this session.';
      if (!last || !last.decisionId) {
        return { text: `No recent routing decision found. ${statusText} Send a message first, then use /route feedback to request a feedback card (or /route feedback on|off to toggle).` };
      }
      const sender = resolveSenderForSession(ctx.sessionKey) ?? (ctx.senderId ? { senderId: String(ctx.senderId) } : null);
      if (!sender?.senderId) {
        return { text: "Cannot send feedback card: sender not resolved for this session." };
      }
      const syntheticDecision: RouteResponse = {
        decision_id: last.decisionId,
        task_type: last.taskType,
        confidence: last.confidence,
        selected_model: last.selectedModel,
        selected_provider: last.selectedProvider,
        fallbacks: last.fallbacks,
        score: last.score,
        reason: last.reason,
        classifier_source: last.classifierSource as RouteResponse["classifier_source"],
        reply_context_used: last.replyContextUsed,
      };
      const sent = await sendTelegramFeedbackCard(api, sender.senderId, syntheticDecision, last.sourceTag, { messagePreview: last.promptText, sessionKey: ctx.sessionKey });
      const currentState = suppressed ? 'OFF' : 'ON';
      return { text: (sent ? `Feedback card sent.` : `Failed to send feedback card (rate limited or bridge error).`) + ` Current: ${currentState}.` };
    }


        if (arg === "explain") {
          const last = resolveLastDecisionForContext(ctx, conversationKey);
          return buildRouteExplainReply(last);
        }

        if (arg.startsWith("compare")) {
          const last = resolveLastDecisionForContext(ctx, conversationKey);
          const modeList = rawArgs.split(/\s+/).slice(1).map((m: string) => m.toLowerCase()).filter(Boolean) as RouteMode[];
          const compareModes: RouteMode[] = modeList.length ? modeList : ["fast", "balanced", "reasoning", "eco", "free"] as RouteMode[];
          return buildRouteCompareReply(routerUrl, last, compareModes, timeoutMs);
        }

        if (!normalized) {
          return {
            text: buildRouteHelpText(currentMode),
            interactive: buildRouteInteractiveReply("auto").interactive,
          };
        }

        const routeModeSessionKey = sessionKeyForConversation ?? ctx.conversationId ?? ctx.sessionKey ?? ctx.channelId ?? ctx.senderId ?? "";
        rememberRouteMode(routeModeSessionKey, normalized, hasFreeModifier);
        if (ctx.sessionKey && ctx.sessionKey !== routeModeSessionKey) rememberRouteMode(ctx.sessionKey, normalized, hasFreeModifier);
        rememberConversationRouteMode(conversationKey, normalized, hasFreeModifier);
        // Also store with a channel-only key so before_model_resolve can find the mode
        // even when its hook ctx does not carry the full from/to/thread fields.
        const channelOnlyKey = ctx.channelId ?? ctx.channel ?? "";
        if (channelOnlyKey && channelOnlyKey !== conversationKey) {
          rememberConversationRouteMode(channelOnlyKey, normalized, hasFreeModifier);
        }
        if (conversationKey) await persistRouteModePreference(routerUrl, conversationKey, normalized, "conversation", hasFreeModifier);
        if (ctx.sessionKey) await persistRouteModePreference(routerUrl, ctx.sessionKey, normalized, "session", hasFreeModifier);
        if (channelOnlyKey) await persistRouteModePreference(routerUrl, channelOnlyKey, normalized, "channel", hasFreeModifier);
        return buildRouteInteractiveReply(normalized, "this conversation", hasFreeModifier, false);
      },
    });

    // ── Feedback card interactive handler ─────────────────────────────────────
    // Handles `routefb:*` callbacks so that clicking "Wrong" does NOT
    // immediately record `wrong` but instead asks for the correct task and
    // model verdict (Step 1/2 → Step 2/2). This matches the expected UX:
    // Wrong → Which task was it? → Was the model right? → recorded.
    try {
      const registerInteractive = (api as any).registerInteractiveHandler?.bind(api);
      if (typeof registerInteractive === "function") {
        registerInteractive({
          channel: "telegram",
          namespace: "routefb",
          handler: async (ctx: any) => {
            const rawPayload: string = String(ctx?.callback?.payload ?? ctx?.callback?.data ?? "").trim();
            let payload = rawPayload;
            // Gateway may pass full `routefb:...` string as payload in some versions;
            // our registration is for namespace `routefb`, so payload should be the
            // suffix after `routefb:`. Handle both forms defensively.
            if (payload.startsWith("routefb:")) {
              payload = payload.slice("routefb:".length);
            }
            const parts = payload.split(":");
            const action = (parts[0] ?? "").trim();
            const decisionId = (parts[1] ?? "").trim();
            const extra1 = parts[2] != null ? String(parts[2]).trim() : undefined;
            const extra2 = parts[3] != null ? String(parts[3]).trim() : undefined;

            const routerUrlResolved = (api as any)?.config?.routerUrl ?? (cfg as any).routerUrl ?? DEFAULT_URL;

            const postFeedback = async (body: Record<string, unknown>): Promise<boolean> => {
              try {
                const res = await fetch(`${routerUrlResolved}/feedback`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(body),
                });
                return res.ok;
              } catch {
                return false;
              }
            };

            // Helper to safely edit the original feedback message
            const safeEdit = async (text: string, buttons: Array<Array<{ text: string; callback_data: string }>>): Promise<void> => {
              try {
                if (typeof ctx?.respond?.editMessage === "function") {
                  await ctx.respond.editMessage({ text, buttons });
                } else if (typeof ctx?.editMessage === "function") {
                  await ctx.editMessage({ text, buttons });
                }
              } catch {
                // best-effort; Telegram edit may fail if message was deleted
              }
            };

            if (action === "ok" && decisionId) {
              const ok = await postFeedback({
                decision_id: decisionId,
                verdict: "correct",
                source_surface: "telegram_feedback_card",
                source_channel: "telegram",
                source_user_id: ctx?.senderId ? String(ctx.senderId) : undefined,
                source_message_id: ctx?.callback?.messageId ? String(ctx.callback.messageId) : undefined,
              });
              if (ok) {
                await safeEdit("✅ Feedback recorded.", []);
              } else {
                await safeEdit("⚠️ Could not record feedback. Please try again.", buildFeedbackKeyboard(decisionId));
              }
              return { handled: true };
            }

            if (action === "wrong" && decisionId) {
              await safeEdit("❌ Step 1/2 — Which task was it?", buildWrongTaskKeyboard(decisionId));
              return { handled: true };
            }

            if (action === "wrong_task" && decisionId) {
              const taskSlot = (extra1 ?? "skip").trim();
              await safeEdit("❌ Step 2/2 — Was the model right?", buildWrongModelKeyboard(decisionId, taskSlot));
              return { handled: true };
            }

            if (action === "wrong_model" && decisionId) {
              const taskSlot = (extra1 ?? "skip").trim();
              const modelVerdictRaw = (extra2 ?? "skip").trim();
              const correctedTask =
                taskSlot && taskSlot !== "skip" && (FEEDBACK_TASK_LABELS as readonly string[]).includes(taskSlot as any)
                  ? taskSlot
                  : null;
              const modelVerdict =
                modelVerdictRaw && modelVerdictRaw !== "skip" && Object.prototype.hasOwnProperty.call(FEEDBACK_MODEL_VERDICTS, modelVerdictRaw)
                  ? modelVerdictRaw
                  : null;
              const ok = await postFeedback({
                decision_id: decisionId,
                verdict: "wrong",
                corrected_task: correctedTask,
                model_verdict: modelVerdict,
                source_surface: "telegram_feedback_card",
                source_channel: "telegram",
                source_user_id: ctx?.senderId ? String(ctx.senderId) : undefined,
                source_message_id: ctx?.callback?.messageId ? String(ctx.callback.messageId) : undefined,
              });
              if (ok) {
                const bits: string[] = [];
                if (correctedTask) bits.push(`task→${correctedTask}`);
                if (modelVerdict) bits.push(`model→${modelVerdict}`);
                const summary = bits.length ? `✅ Marked wrong (${bits.join(", ")}).` : "✅ Marked wrong.";
                await safeEdit(summary, []);
              } else {
                await safeEdit("⚠️ Feedback service unreachable. Please try again.", []);
              }
              return { handled: true };
            }

            if (action === "fix" && decisionId) {
              const rawTask = extra1 ?? "";
              // `skip` means user chose to mark wrong without specifying task
              if (!rawTask || rawTask === "skip") {
                const ok = await postFeedback({
                  decision_id: decisionId,
                  verdict: "wrong",
                  corrected_task: null,
                  source_surface: "telegram_feedback_card",
                  source_channel: "telegram",
                  source_user_id: ctx?.senderId ? String(ctx.senderId) : undefined,
                  source_message_id: ctx?.callback?.messageId ? String(ctx.callback.messageId) : undefined,
                });
                await safeEdit(ok ? "✅ Marked wrong." : "⚠️ Could not record feedback.", []);
                return { handled: true };
              }
              const correctedTask = (FEEDBACK_TASK_LABELS as readonly string[]).includes(rawTask as any) ? rawTask : null;
              if (!correctedTask) {
                await safeEdit("⚠️ Invalid task.", buildWrongTaskKeyboard(decisionId));
                return { handled: true };
              }
              const ok = await postFeedback({
                decision_id: decisionId,
                verdict: "wrong",
                corrected_task: correctedTask,
                source_surface: "telegram_feedback_card",
                source_channel: "telegram",
                source_user_id: ctx?.senderId ? String(ctx.senderId) : undefined,
                source_message_id: ctx?.callback?.messageId ? String(ctx.callback.messageId) : undefined,
              });
              await safeEdit(ok ? `✅ Marked wrong → ${correctedTask}.` : "⚠️ Could not record feedback.", []);
              return { handled: true };
            }

            if (action === "back" && decisionId) {
              const target = (extra1 ?? "initial").trim();
              if (target === "initial") {
                await safeEdit("Feedback?", buildFeedbackKeyboard(decisionId));
              } else if (target === "wrong_task") {
                await safeEdit("❌ Step 1/2 — Which task was it?", buildWrongTaskKeyboard(decisionId));
              } else {
                await safeEdit("Feedback?", buildFeedbackKeyboard(decisionId));
              }
              return { handled: true };
            }

            if (action === "cancel" && decisionId) {
              await safeEdit("❌ Cancelled.", []);
              return { handled: true };
            }

            return { handled: false };
          },
        });
        void debugLog("[feedback-handler] registered interactive handler for routefb:telegram");
      } else {
        void debugLog("[feedback-handler] registerInteractiveHandler not available");
      }
    } catch (e: any) {
      void debugLog(`[feedback-handler] failed to register: ${e?.message ?? String(e)}`);
    }

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
        const previousConversationContext = rememberRecentConversationText(conversationKey, rawText);
        if (previousConversationContext) {
          rememberConversationContextForSession(sessionKey, previousConversationContext);
        }
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
const modeResolution = await resolveRouteModeDetailsFromContext(api, ctx);
      const routeMode = modeResolution.mode;
      const freeFilter = Boolean(modeResolution.freeFilter);
      const shouldEnforceConfidenceGate = modeResolution.source === "default" && routeMode === "auto" && !freeFilter;
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

      const sessionKind = classifySessionKind(ctx?.sessionKey);
      if (ctx?.trigger === "cron" || sessionKind === "cron") {
        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
        }
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} bypassed reason=cron`);
        return;
      }

      if (sessionKind === "dreaming") {
        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
        }
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} bypassed reason=dreaming`);
        return;
      }

      if (ctx?.trigger === "heartbeat" || (ctx?.trigger === "memory" && source !== "raw-user")) {
        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
        }
        await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} bypassed reason=${ctx?.trigger}`);
        return;
      }

      if (routeMode === "off") {
        const shadowTimeoutMs = Math.max(timeoutMs, SHADOW_TIMEOUT_MS);
        const shadowResult = await routeRequest(routerUrl, routingText, firstPassCostProfile, shadowTimeoutMs, routeMode, conversationContext, shouldUseLlmClassifier, source, sourceTag, "shadow", freeFilter);
        const shadowDecision = shadowResult.decision;
        if (!shadowDecision) {
          const failure = describeRouteRequestFailure(shadowResult, shadowTimeoutMs);
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
          excludedModels: shadowDecision.excluded_models ?? [],
          score: shadowDecision.score,
          reason: [...shadowDecision.reason, "route_off"],
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
            targetSenderId: (resolveSenderForSession(ctx.sessionKey ?? sessionRef) ?? (ctx.senderId ? { senderId: String(ctx.senderId) } : undefined))?.senderId,
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

            // Ensure feedback card is still offered for cached overrides when possible.
            // In queued/busy session cases the main routing decision may have already
            // been recorded earlier; try to reuse the last recorded decision for this
            // session or conversation to send a feedback card. This addresses cases
            // where before_model_resolve returns early due to dedupe but a feedback
            // card was expected for the follow-up message.
            try {
              const lastDecision = (ctx.sessionKey && recentLastDecisionBySession.get(ctx.sessionKey))
                || (conversationKey && takeLastDecisionForConversation(conversationKey))
                || recentLastDecisionGlobal;
              const sender = resolveSenderForSession(ctx.sessionKey ?? sessionRef) ?? (ctx.senderId ? { senderId: String(ctx.senderId) } : null);
              if (lastDecision && lastDecision.decisionId && sender?.senderId) {
                // Build a minimal RouteResponse-like object from LastRouteDecision
                const syntheticDecision: RouteResponse = {
                  decision_id: lastDecision.decisionId,
                  task_type: lastDecision.taskType,
                  confidence: lastDecision.confidence,
                  selected_model: lastDecision.selectedModel,
                  selected_provider: lastDecision.selectedProvider,
                  fallbacks: lastDecision.fallbacks,
                  excluded_models: lastDecision.excludedModels,
                  score: lastDecision.score,
                  reason: lastDecision.reason,
                  classifier_source: lastDecision.classifierSource as RouteResponse['classifier_source'],
                  reply_context_used: lastDecision.replyContextUsed,
                };
                // Fire-and-forget; do not block routing on notification success
                void sendTelegramFeedbackCard(api, sender.senderId, syntheticDecision, sourceTag, { messagePreview: routingText });
              }
            } catch (err) {
              // best-effort only
            }

            recentRouteCacheBySession.set(sessionRef, {
              text: dedupeText,
              mode: routeMode,
              at: Date.now(),
            });
            return cachedOverride;
          }
        } else {
          await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} dedupe-hit bypass`);
          return;
        }
      }

      let decision: RouteResponse | null = null;
      let finalMode: RouteMode = routeMode;
      let finalCostProfile = firstPassCostProfile;

      if (routeMode === "auto") {
        const autoResult = await routeRequest(routerUrl, routingText, firstPassCostProfile, timeoutMs, routeMode, conversationContext, shouldUseLlmClassifier, source, sourceTag, "route", freeFilter);
        decision = autoResult.decision;
        if (!decision) {
          const failure = describeRouteRequestFailure(autoResult, timeoutMs);
          await debugLog(`[hook-result] source=${source} source_tag=${sourceTag} route=${routeMode} ${failure}`);
          if (debugMode) console.warn(`[nexus-router] ${failure}, using default model`);
          return;
        }
      } else {
        const directResult = await routeRequest(routerUrl, routingText, firstPassCostProfile, timeoutMs, routeMode, conversationContext, shouldUseLlmClassifier, source, sourceTag, "route", freeFilter);
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

      const sender = resolveSenderForSession(ctx.sessionKey ?? sessionRef) ?? (ctx.senderId ? { senderId: String(ctx.senderId), channelId: ctx.channelId ? String(ctx.channelId) : undefined } : null);
      if (decision.decision_id) {
        if (sender) {
          await sendTelegramFeedbackCard(api, sender.senderId, decision, sourceTag, { messagePreview: routingText, sessionKey: ctx.sessionKey });
        } else {
          await debugLog(
            `[feedback-card] skipped decision=${decision.decision_id} reason=missing_sender session=${ctx.sessionKey ?? sessionRef ?? ""}`,
          );
        }
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
        firstPassModel: routeMode === "auto" ? (decision?.selected_model) : decision.selected_model,
        firstPassProvider: routeMode === "auto" ? (decision?.selected_provider) : decision.selected_provider,
        selectedModel: decision.selected_model,
        selectedProvider: decision.selected_provider,
        fallbacks: decision.fallbacks,
        excludedModels: decision.excluded_models ?? [],
        score: decision.score,
        reason: decision.reason,
        autoEscalated: routeMode === "auto" && finalMode === "balanced",
        applied: !shouldEnforceConfidenceGate || decision.confidence >= minConfidence,
      };
      rememberLastDecision(ctx.sessionKey, lastDecision);

      if (shouldEnforceConfidenceGate && decision.confidence < minConfidence) {
        lastDecision.skippedReason = `confidence ${decision.confidence.toFixed(2)} below threshold ${minConfidence} in default auto mode`;
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
              { shadowMode: true, actualModel: outcomeModel, messagePreview: decision.promptText },
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
