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
  classifier_source?: "explicit" | "heuristic" | "llm" | "fallback";
  reply_context_used?: boolean;
}

interface LastRouteDecision {
  at: number;
  decisionId?: string;
  requestedRouteMode: RouteMode;
  routeMode: RouteMode;
  source: "compiled-prompt" | "raw-user";
  promptLen: number;
  promptText?: string;
  replyContextUsed: boolean;
  classifierSource: "explicit" | "heuristic" | "llm" | "fallback";
  costProfile: "cheap" | "balanced" | "premium";
  taskType: string;
  confidence: number;
  selectedModel: string;
  selectedProvider: string;
  actualModel?: string;
  actualProvider?: string;
  usage?: { input?: number; output?: number; total?: number };
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
}

// ── Defaults ──────────────────────────────────────────────────────────────────

const DEFAULT_URL         = "http://127.0.0.1:7771";
const DEFAULT_CONFIDENCE  = 0.60;
const DEFAULT_TIMEOUT_MS  = 3000;
const DEFAULT_COST        = "balanced";
const PLUGIN_VERSION      = "0.1.0";
const RECENT_MESSAGE_TTL_MS = 5 * 60 * 1000;
const AUTO_ESCALATE_CONFIDENCE = 0.76;
const ROUTE_MODES = new Set(["auto", "balanced", "fast", "reasoning", "off"]);
const ROUTE_DEDUPE_WINDOW_MS = 20_000;
const ROUTE_BURST_WINDOW_MS = 5_000;
const ROUTE_BURST_MAX_CALLS = 4;
const ROUTE_BURST_BLOCK_MS = 60_000;

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
): Promise<RouteResponse | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

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

    clearTimeout(timer);

    if (!res.ok) return null;
    return (await res.json()) as RouteResponse;
  } catch {
    return null;
  }
}

// ── Plugin entry ──────────────────────────────────────────────────────────────

const recentUserMessages = new Map<string, { text: string; at: number }>();
const recentRouteModes = new Map<string, { mode: RouteMode; at: number }>();
const recentConversationRouteModes = new Map<string, { mode: RouteMode; at: number }>();
const recentConversationKeyBySession = new Map<string, { conversationKey: string; at: number }>();
const recentSessionKeyByConversation = new Map<string, { sessionKey: string; at: number }>();
const recentConversationContextBySession = new Map<string, { context: string; at: number }>();
const recentLastDecisionBySession = new Map<string, LastRouteDecision>();
const recentLastDecisionByConversation = new Map<string, LastRouteDecision>();
const recentLastDecisionByDecisionId = new Map<string, LastRouteDecision>();
const pendingOutcomeQueueBySessionId = new Map<string, PendingOutcome[]>();
const recentRouteCacheBySession = new Map<string, { text: string; mode: RouteMode; at: number; selectedModel?: string }>();
const routeBurstBySession = new Map<string, { windowStart: number; count: number; blockedUntil?: number }>();
let recentLastDecisionGlobal: LastRouteDecision | null = null;

function rememberRecentUserMessage(sessionKey: string, text: string): void {
  const trimmed = text.trim();
  if (!sessionKey || !trimmed) return;
  recentUserMessages.set(sessionKey, { text: trimmed, at: Date.now() });
}

// Modes that should never expire — set explicitly by the user.
const STICKY_ROUTE_MODES = new Set<RouteMode>(["off"]);

function rememberRouteMode(sessionKey: string, mode: RouteMode): void {
  if (!sessionKey || !ROUTE_MODES.has(mode)) return;
  // Sticky modes use a far-future timestamp so they never expire.
  const at = STICKY_ROUTE_MODES.has(mode) ? Number.MAX_SAFE_INTEGER : Date.now();
  recentRouteModes.set(sessionKey, { mode, at });
}

function rememberConversationRouteMode(conversationKey: string, mode: RouteMode): void {
  if (!conversationKey || !ROUTE_MODES.has(mode)) return;
  const at = STICKY_ROUTE_MODES.has(mode) ? Number.MAX_SAFE_INTEGER : Date.now();
  recentConversationRouteModes.set(conversationKey, { mode, at });
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

function takeRecentRouteMode(sessionKey?: string): RouteMode | null {
  if (!sessionKey) return null;
  const entry = recentRouteModes.get(sessionKey);
  if (!entry) return null;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentRouteModes.delete(sessionKey);
    return null;
  }
  return entry.mode;
}

function takeRecentConversationRouteMode(conversationKey?: string): RouteMode | null {
  if (!conversationKey) return null;
  const entry = recentConversationRouteModes.get(conversationKey);
  if (!entry) return null;
  if (Date.now() - entry.at > RECENT_MESSAGE_TTL_MS) {
    recentConversationRouteModes.delete(conversationKey);
    return null;
  }
  return entry.mode;
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

function buildRouteDecisionText(last: LastRouteDecision | null, detailed: boolean): { text: string } {
  if (!last) {
    return {
      text: "No recent routing decision found yet for this conversation.",
    };
  }

  const fallbacks = last.fallbacks.length ? last.fallbacks.join(", ") : "none";
  const reason = detailed ? last.reason.join("; ") : (last.reason.length ? last.reason.slice(0, 3).join("; ") : "n/a");
  const actual = last.actualModel ? `${last.actualProvider ?? "unknown"}/${last.actualModel}` : "not recorded yet";
  const usageBits = last.usage
    ? ` in=${last.usage.input ?? "?"} out=${last.usage.output ?? "?"} total=${last.usage.total ?? "?"}`
    : "";
  const contextLabel = last.replyContextUsed ? "reply-context used: yes" : "reply-context used: no";
  const escalationLabel = last.autoEscalated ? `${last.requestedRouteMode} → ${last.routeMode}` : last.routeMode;

  const lines = [
    `Routing: ${escalationLabel}`,
    `Input: ${last.source}`,
    `Context: ${contextLabel}`,
    `Classifier: ${last.classifierSource}`,
    `Task: ${last.taskType}`,
    `Confidence: ${last.confidence.toFixed(2)}`,
    `Selected: ${last.selectedModel}`,
    `Actual: ${actual}${usageBits}`,
    `Fallbacks: ${fallbacks}`,
    `Reason: ${reason}`,
  ];

  return { text: lines.join("\n") };
}

function buildRouteLastReply(last: LastRouteDecision | null): { text: string } {
  return buildRouteDecisionText(last, false);
}

function buildRouteExplainReply(last: LastRouteDecision | null): { text: string } {
  return buildRouteDecisionText(last, true);
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
      const decision = await routeRequest(routerUrl, last.promptText ?? "", costProfile, timeoutMs, mode);
      return { mode, costProfile, decision };
    }),
  );

  const lines = [
    `Prompt: ${last.promptText.slice(0, 120)}${last.promptText.length > 120 ? "…" : ""}`,
    `Task hint: ${last.taskType} · source=${last.source}`,
    "",
  ];

  for (const row of results) {
    if (!row.decision) {
      lines.push(`${row.mode}: unavailable`);
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

function buildRouteInteractiveReply(mode?: RouteMode): {
  text: string;
  interactive: { blocks: Array<{ type: "text"; text: string } | { type: "buttons"; buttons: Array<{ label: string; value: string; style?: "primary" | "secondary" | "success" | "danger" }> }> };
} {
  const label = mode ?? "auto";
  return {
    text: `⚙️ Routing mode: ${label} (this session).`,
    interactive: {
      blocks: [
        { type: "text", text: `Choose a routing mode (current: ${label}, sticky for this session):` },
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

function buildRouteHelpText(currentMode: RouteMode): string {
  return [
    `⚙️ Nexus Router help`,
    `Current session mode: ${currentMode} (sticky until changed)`,
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
  const cached = takeRecentRouteMode(sessionKey);
  if (cached) return cached;

  const conversationKey = resolveConversationKeyForSession(sessionKey);
  const conversationMode = takeRecentConversationRouteMode(conversationKey ?? undefined);
  if (conversationMode) {
    if (sessionKey) rememberRouteMode(sessionKey, conversationMode);
    return conversationMode;
  }

  // Do not recover mode from historical session messages.
  // That can resurrect stale modes (e.g. old `/route off`) when command/session keys diverge.
  return "auto";
}

async function resolveRouteModeFromContext(api: any, ctx: any): Promise<RouteMode> {
  const conversationKey = buildConversationKeyFromContext(ctx);
  if (conversationKey) {
    const conversationMode = takeRecentConversationRouteMode(conversationKey);
    if (conversationMode) {
      if (ctx?.sessionKey) rememberRouteMode(ctx.sessionKey, conversationMode);
      return conversationMode;
    }
  }

  const sessionMode = await resolveRouteModeFromSession(api, ctx?.sessionKey);
  if (sessionMode !== "auto") return sessionMode;

  if (conversationKey) {
    const fallbackConversationMode = takeRecentConversationRouteMode(conversationKey);
    if (fallbackConversationMode) {
      if (ctx?.sessionKey) rememberRouteMode(ctx.sessionKey, fallbackConversationMode);
      return fallbackConversationMode;
    }
  }

  // Fallback: check channel-only key — covers the case where the hook ctx lacks
  // from/to/thread fields that the command ctx used when building conversationKey.
  const channelKey = ctx?.channelId ?? ctx?.channel ?? "";
  if (channelKey && channelKey !== conversationKey) {
    const channelMode = takeRecentConversationRouteMode(channelKey);
    if (channelMode) {
      if (ctx?.sessionKey) rememberRouteMode(ctx.sessionKey, channelMode);
      return channelMode;
    }
  }

  return sessionMode;
}

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
        const inferredMode = await resolveRouteModeFromContext(api, ctx);
        const currentMode = inferredMode ?? takeRecentConversationRouteMode(conversationKey) ?? "auto";

        if (!arg || arg === "help" || arg === "?") {
          return {
            text: buildRouteHelpText(currentMode),
            interactive: buildRouteInteractiveReply(currentMode).interactive,
          };
        }

        if (arg === "status") {
          return buildRouteInteractiveReply(currentMode);
        }

        if (arg === "last") {
          const last = takeLastDecisionForConversation(conversationKey) ?? recentLastDecisionGlobal;
          return buildRouteLastReply(last);
        }

        if (arg === "explain") {
          const last = takeLastDecisionForConversation(conversationKey) ?? recentLastDecisionGlobal;
          return buildRouteExplainReply(last);
        }

        if (arg.startsWith("compare")) {
          const last = takeLastDecisionForConversation(conversationKey) ?? recentLastDecisionGlobal;
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
        return buildRouteInteractiveReply(normalized);
      },
    });

    api.on("before_dispatch", (event: any, ctx: any) => {
      const rawText = (event.body ?? event.content ?? "").trim();
      const sessionKey = ctx.sessionKey ?? event.sessionKey;
      const channel = ctx.channelId ?? event.channel ?? "unknown";
      const conversationKey = [channel, ctx.accountId ?? "default", ctx.conversationId ?? "", ""].join(":");

      // Skip slash commands — they are not user prompts and should not be routed.
      const isSlashCommand = rawText.startsWith("/");
      if (sessionKey && rawText && !isSlashCommand) {
        rememberRecentUserMessage(sessionKey, rawText);
      }
      if (sessionKey && ctx.conversationId) {
        rememberConversationKeyForSession(sessionKey, conversationKey);
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

      const rawUserText = takeRecentUserMessage(ctx.sessionKey) ?? "";
      const conversationContext = takeRecentConversationContext(ctx.sessionKey) ?? "";
      const routingText = rawUserText || prompt;
      const source = rawUserText ? "raw-user" : "compiled-prompt";
      const routeMode = await resolveRouteModeFromContext(api, ctx);
      const firstPassCostProfile = resolveCostProfileForRouteMode(routeMode, costProfile);
      await debugLog(
        `[hook-enter] source=${source} route=${routeMode} prompt_len=${routingText.length} profile=${firstPassCostProfile}`,
      );

      if (routeMode === "off") {
        await debugLog(`[hook-result] source=${source} route=${routeMode} bypassed`);
        return;
      }

      const sessionRef = String(ctx?.sessionKey ?? ctx?.sessionId ?? ctx?.conversationId ?? "");
      if (sessionRef && shouldBlockRoutingForBurst(sessionRef)) {
        await debugLog(`[hook-result] source=${source} route=blocked burst session=${sessionRef}`);
        return;
      }

      const dedupeText = routingText.trim();
      const cached = sessionRef ? recentRouteCacheBySession.get(sessionRef) : undefined;
      if (
        sessionRef &&
        cached &&
        Date.now() - cached.at < ROUTE_DEDUPE_WINDOW_MS &&
        cached.mode === routeMode &&
        cached.text === dedupeText
      ) {
        if (cached.selectedModel) {
          await debugLog(`[hook-result] source=${source} route=${routeMode} dedupe-hit model=${cached.selectedModel}`);
          return splitSelectedModelRef(cached.selectedModel);
        }
        await debugLog(`[hook-result] source=${source} route=${routeMode} dedupe-hit bypass`);
        return;
      }

      let decision: RouteResponse | null = null;
      let finalMode: RouteMode = routeMode;
      let finalCostProfile = firstPassCostProfile;

      const shouldUseLlmClassifier = Boolean(conversationContext.trim());

      if (routeMode === "auto") {
        const autoDecision = await routeRequest(
          routerUrl,
          routingText,
          firstPassCostProfile,
          timeoutMs,
          routeMode,
          conversationContext,
          shouldUseLlmClassifier,
        );
        if (!autoDecision) {
          await debugLog(`[hook-result] source=${source} route=${routeMode} router_unavailable`);
          if (debugMode) console.warn("[nexus-router] router unavailable, using default model");
          return;
        }

        if (autoDecision.confidence < AUTO_ESCALATE_CONFIDENCE) {
          const balancedCostProfile = resolveCostProfileForRouteMode("balanced", costProfile);
          const balancedDecision = await routeRequest(
            routerUrl,
            routingText,
            balancedCostProfile,
            timeoutMs,
            "balanced",
            conversationContext,
            shouldUseLlmClassifier,
          );
          if (balancedDecision) {
            decision = balancedDecision;
            finalMode = "balanced";
            finalCostProfile = balancedCostProfile;
            await debugLog(
              `[hook-auto-escalate] source=${source} from=auto confidence=${autoDecision.confidence.toFixed(2)} to=balanced selected=${balancedDecision.selected_model}`,
            );
          } else {
            decision = autoDecision;
          }
        } else {
          decision = autoDecision;
        }
      } else {
        decision = await routeRequest(
          routerUrl,
          routingText,
          firstPassCostProfile,
          timeoutMs,
          routeMode,
          conversationContext,
          shouldUseLlmClassifier,
        );
        if (!decision) {
          await debugLog(`[hook-result] source=${source} router_unavailable`);
          if (debugMode) console.warn("[nexus-router] router unavailable, using default model");
          return;
        }
      }

      if (!decision) {
        if (sessionRef) {
          recentRouteCacheBySession.set(sessionRef, { text: dedupeText, mode: routeMode, at: Date.now() });
        }
        await debugLog(`[hook-result] source=${source} router_unavailable`);
        if (debugMode) console.warn("[nexus-router] router unavailable, using default model");
        return;
      }

      if (sessionRef) {
        recentRouteCacheBySession.set(sessionRef, {
          text: dedupeText,
          mode: routeMode,
          at: Date.now(),
          selectedModel: decision.selected_model,
        });
      }

      if (decision.confidence < minConfidence) {
        await debugLog(
          `[hook-result] source=${source} skipped confidence=${decision.confidence.toFixed(2)} threshold=${minConfidence}`,
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
        promptLen: routingText.length,
        promptText: routingText,
        replyContextUsed: decision.reply_context_used ?? Boolean(conversationContext.trim()),
        classifierSource: (decision.classifier_source ?? (shouldUseLlmClassifier ? "llm" : "heuristic")) as LastRouteDecision["classifierSource"],
        costProfile: finalCostProfile,
        taskType: decision.task_type,
        confidence: decision.confidence,
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
        `[hook-result] source=${source} route=${finalMode} override model=${decision.selected_model} task=${decision.task_type} confidence=${decision.confidence.toFixed(2)} score=${decision.score.toFixed(3)}`,
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
        const actualModel = last?.actualModel;
        const actualProvider = last?.actualProvider;
        const fallbackUsed = !!actualModel && actualModel !== pending.selectedModel;

        await fetch(`${routerUrl}/outcome`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            decision_id: pending.decisionId,
            success: event.success,
            latency_ms: event.durationMs,
            fallback_used: fallbackUsed,
            fallback_model: fallbackUsed ? actualModel : undefined,
            provider: actualProvider ?? pending.selectedProvider,
          }),
        });
      } catch {
        // best effort; keep routing path non-blocking
      }
    });

    if (debugMode) {
      console.log(`[nexus-router] registered, router=${routerUrl}, profile=${costProfile}`);
    }
  },
});
