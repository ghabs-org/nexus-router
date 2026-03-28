/**
 * Nexus Router OpenClaw Plugin
 *
 * Routes each chat turn to the best model via the Nexus Router HTTP server.
 *
 * Architecture:
 *   OpenClaw turn arrives
 *     → plugin calls Nexus Router server (POST /route)
 *     → server returns selected_model + fallbacks
 *     → plugin sets model for the turn via context.setModel()
 *     → after turn completes, plugin reports outcome (POST /outcome)
 *
 * Requirements:
 *   - Nexus Router server must be running: python -m src.server
 *   - Default: http://127.0.0.1:7771
 *
 * Config (openclaw.json):
 *   plugins.entries.nexus-router.config:
 *     routerUrl:        "http://127.0.0.1:7771"  (default)
 *     enabled:          true                      (default)
 *     costProfile:      "balanced"                (cheap|balanced|premium)
 *     useLlmClassifier: false                     (use LLM for ambiguous messages)
 *     debugMode:        false                     (log routing decisions to console)
 *     minConfidence:    0.60                      (skip routing below this confidence)
 */

const DEFAULT_ROUTER_URL = "http://127.0.0.1:7771";
const PLUGIN_VERSION = "0.1.0";

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

interface RouterConfig {
  routerUrl?: string;
  enabled?: boolean;
  costProfile?: "cheap" | "balanced" | "premium";
  useLlmClassifier?: boolean;
  debugMode?: boolean;
  minConfidence?: number;
}

interface RouteRequest {
  message: string;
  has_image?: boolean;
  cost_profile?: string;
  use_llm_classifier?: boolean;
  classifier?: ClassifierHint;
  nexus_context?: NexusContext;
}

interface ClassifierHint {
  task_type: string;
  subtype?: string | null;
  complexity?: string;
  needs_tools?: boolean;
  needs_vision?: boolean;
  needs_long_context?: boolean;
  cost_profile?: string;
  confidence?: number;
  detected_language?: string | null;
}

interface NexusContext {
  nexus_workflow_id?: string;
  nexus_step_id?: string;
  nexus_issue_id?: string;
  nexus_project?: string;
}

interface RouteResponse {
  task_type: string;
  confidence: number;
  selected_model: string;
  selected_provider: string;
  fallbacks: string[];
  score: number;
  reason: string[];
  pre_signals?: Record<string, unknown>;
  _decision_id?: string;
}

interface OutcomeRequest {
  decision_id: string;
  success: boolean;
  provider: string;
  latency_ms?: number;
  http_status?: number;
  fallback_used?: boolean;
  fallback_model?: string;
  user_override?: boolean;
  user_override_model?: string;
  error_type?: string;
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

async function post<T>(url: string, body: unknown, timeoutMs = 3000): Promise<T | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Editor-Version": PLUGIN_VERSION,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

async function get<T>(url: string, timeoutMs = 2000): Promise<T | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, {
      signal: controller.signal,
      headers: {
        "Editor-Version": PLUGIN_VERSION,
      },
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

// ── Plugin state ──────────────────────────────────────────────────────────────

let _config: RouterConfig = {};
let _serverAvailable: boolean | null = null;  // null = not yet checked
let _lastHealthCheck = 0;
const HEALTH_CHECK_INTERVAL_MS = 60_000; // re-check every 60s

async function isServerAvailable(routerUrl: string): Promise<boolean> {
  const now = Date.now();
  if (_serverAvailable !== null && (now - _lastHealthCheck) < HEALTH_CHECK_INTERVAL_MS) {
    return _serverAvailable;
  }
  const result = await get<{ status: string }>(`${routerUrl}/health`, 1500);
  _serverAvailable = result?.status === "ok";
  _lastHealthCheck = now;
  return _serverAvailable;
}

// ── Plugin exports ────────────────────────────────────────────────────────────

/**
 * Plugin install/init hook.
 * Called once by OpenClaw when the plugin is loaded.
 */
export function install(config: RouterConfig = {}): void {
  _config = {
    routerUrl: config.routerUrl ?? DEFAULT_ROUTER_URL,
    enabled: config.enabled ?? true,
    costProfile: config.costProfile ?? "balanced",
    useLlmClassifier: config.useLlmClassifier ?? false,
    debugMode: config.debugMode ?? false,
    minConfidence: config.minConfidence ?? 0.60,
  };

  if (_config.debugMode) {
    console.log("[nexus-router] plugin installed, router:", _config.routerUrl);
  }
}

/**
 * Route a single turn.
 *
 * Returns the routing decision or null if the router is unavailable.
 * The caller is responsible for applying the decision to the turn.
 */
export async function routeTurn(options: {
  message: string;
  hasImage?: boolean;
  classifierHint?: ClassifierHint;
  nexusContext?: NexusContext;
}): Promise<RouteResponse | null> {
  const cfg = _config;
  if (!cfg.enabled) return null;

  const routerUrl = cfg.routerUrl ?? DEFAULT_ROUTER_URL;
  const available = await isServerAvailable(routerUrl);
  if (!available) {
    if (cfg.debugMode) {
      console.warn("[nexus-router] server unavailable, skipping routing");
    }
    return null;
  }

  const payload: RouteRequest = {
    message: options.message,
    has_image: options.hasImage ?? false,
    cost_profile: cfg.costProfile,
    use_llm_classifier: cfg.useLlmClassifier,
  };

  if (options.classifierHint) {
    payload.classifier = options.classifierHint;
  }
  if (options.nexusContext) {
    payload.nexus_context = options.nexusContext;
  }

  const decision = await post<RouteResponse>(`${routerUrl}/route`, payload, 3000);

  if (!decision) {
    if (cfg.debugMode) console.warn("[nexus-router] route request failed");
    return null;
  }

  if (decision.confidence < (cfg.minConfidence ?? 0.60)) {
    if (cfg.debugMode) {
      console.log(`[nexus-router] confidence ${decision.confidence} below threshold, skipping`);
    }
    return null;
  }

  if (cfg.debugMode) {
    console.log(
      `[nexus-router] ${decision.task_type} → ${decision.selected_model} ` +
      `(score=${decision.score.toFixed(3)}, confidence=${decision.confidence.toFixed(2)})`
    );
  }

  return decision;
}

/**
 * Record the outcome of a routed turn.
 * Call this after the turn completes.
 */
export async function recordOutcome(outcome: OutcomeRequest): Promise<void> {
  const routerUrl = _config.routerUrl ?? DEFAULT_ROUTER_URL;
  await post(`${routerUrl}/outcome`, outcome, 2000);
}

/**
 * Get a health snapshot from the router.
 * Useful for diagnostics or periodic checks.
 */
export async function getHealth(): Promise<Record<string, unknown> | null> {
  const routerUrl = _config.routerUrl ?? DEFAULT_ROUTER_URL;
  return get(routerUrl + "/health");
}

/**
 * Probe provider auth status.
 * Triggers active probes on the router side.
 */
export async function probeProviders(providers?: string[]): Promise<unknown> {
  const routerUrl = _config.routerUrl ?? DEFAULT_ROUTER_URL;
  return post(`${routerUrl}/probe`, { providers }, 15_000);
}

/**
 * Format a routing decision as a short debug line.
 */
export function formatDecision(d: RouteResponse): string {
  return `[router] ${d.task_type} → ${d.selected_model} (score=${d.score.toFixed(3)})`;
}
