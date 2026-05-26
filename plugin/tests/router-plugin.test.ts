import test from "node:test";
import assert from "node:assert/strict";

import { __testHelpers } from "../src/index.ts";

const {
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
} = __testHelpers;

test("explicit auto mode stays sticky beyond the recent-message TTL", async () => {
  const realNow = Date.now;
  let now = 1_700_000_000_000;

  try {
    Date.now = () => now;
    resetInMemoryRoutingState();
    rememberRouteMode("session:test", "auto");

    now += 10 * 60 * 1000;

    const resolved = await resolveRouteModeDetailsFromContext(undefined, {
      sessionKey: "session:test",
    });

    assert.equal(resolved.mode, "auto");
    assert.equal(resolved.source, "session");
  } finally {
    Date.now = realNow;
    resetInMemoryRoutingState();
  }
});

test("short contextual follow-ups do not trigger the contextual LLM classifier", () => {
  assert.equal(isShortFollowUpForContextualRouting("Did you read this?"), true);
  assert.equal(
    shouldUseContextualLlmClassifier(
      "balanced",
      "We were discussing why the router was timing out.",
      "Did you read this?",
    ),
    false,
  );
  assert.equal(
    shouldUseContextualLlmClassifier(
      "auto",
      "We were discussing why the router was timing out.",
      "Can you compare the trade-offs between these two routing strategies?",
    ),
    true,
  );
});

test("longer ambiguous follow-ups can still opt into contextual LLM classification outside auto/fast", () => {
  assert.equal(
    shouldUseContextualLlmClassifier(
      "balanced",
      "We were discussing why the router was timing out.",
      "Can you compare the trade-offs between these two routing strategies?",
    ),
    true,
  );
  assert.equal(
    shouldUseContextualLlmClassifier(
      "reasoning",
      "We were discussing why the router was timing out.",
      "What architecture would you choose and why?",
    ),
    true,
  );
});

test("recent conversation text provides previous-turn context without current message", () => {
  resetInMemoryRoutingState();
  try {
    const key = "telegram:default:direct:";

    assert.equal(
      rememberRecentConversationText(key, "Please check Biome Backoffice improvements."),
      null,
    );

    const previous = rememberRecentConversationText(key, "Can you complete the BO tasks?");
    assert.equal(previous, "Please check Biome Backoffice improvements.");

    rememberConversationContextForSession("session:test", previous ?? "");
    assert.equal(takeRecentConversationContext("session:test"), "Please check Biome Backoffice improvements.");
  } finally {
    resetInMemoryRoutingState();
  }
});

test("compiled retry prompts for user-triggered retries are bypassed after a recent route decision", () => {
  resetInMemoryRoutingState();
  try {
    rememberLastDecision("session:test", {
      at: Date.now(),
      requestedRouteMode: "auto",
      routeMode: "balanced",
      source: "raw-user",
      sourceTag: "user",
      promptLen: 67,
      promptText: "Yes please, if not maybe gemini 3 flash",
      replyContextUsed: false,
      classifierSource: "heuristic",
      costProfile: "cheap",
      taskType: "fast_utility",
      confidence: 0.72,
      selectedModel: "google-gemini-cli/gemini-3-flash-preview",
      selectedProvider: "google-gemini-cli",
      fallbacks: [],
      score: 0.93,
      reason: [],
      autoEscalated: true,
    });

    assert.equal(
      shouldBypassCompiledRetryRouting({ trigger: "user" }, "compiled-prompt", "session:test"),
      true,
    );
    assert.equal(
      shouldBypassCompiledRetryRouting({ trigger: "heartbeat" }, "compiled-prompt", "session:test"),
      false,
    );
    assert.equal(
      shouldBypassCompiledRetryRouting({ trigger: "user" }, "raw-user", "session:test"),
      false,
    );
  } finally {
    resetInMemoryRoutingState();
  }
});

test("dreaming narrative sessions are classified as background memory work", () => {
  assert.equal(classifySessionKind("dreaming-narrative-rem-4aec231fb4d0"), "dreaming");
  assert.equal(classifySessionKind("agent:main:dreaming-narrative-light-4aec231fb4d0"), "dreaming");
  assert.equal(classifySessionKind("agent:main:telegram:direct:47168736"), "direct");
});


test("runtime capacity failures are inferred as rate-limit exhaustion", () => {
  const failure = inferFailureFromRuntime({
    success: false,
    error: "google-gemini-cli failed: HTTP 429 No capacity available",
  });

  assert.equal(failure.httpStatus, 429);
  assert.equal(failure.errorType, "rate_limit");
  assert.equal(failure.quotaHint, "exhausted");
  assert.equal(failure.quotaRemainingRatio, 0);
  assert.equal(failure.shouldCooldownOverride, true);
});

test("recent failed overrides block immediate re-selection and permit unblocked fallbacks", () => {
  resetInMemoryRoutingState();
  try {
    rememberFailedOverride(
      "session:test",
      "conversation:test",
      "google-gemini-cli/gemini-3-flash-preview",
      "google-gemini-cli",
      "rate_limit",
    );

    const blocked = getFailedOverrideBlock(
      "session:test",
      "conversation:test",
      "google-gemini-cli/gemini-3-flash-preview",
      "google-gemini-cli",
    );
    assert.ok(blocked);

    const fallback = chooseUnblockedFallback(
      [
        "google-gemini-cli/gemini-3-flash-preview",
        "github-copilot/gpt-5.4",
      ],
      "session:test",
      "conversation:test",
    );

    assert.equal(fallback, "github-copilot/gpt-5.4");
  } finally {
    resetInMemoryRoutingState();
  }
});

test("auto escalation threshold is calibrated to avoid escalating ordinary 0.72 confidence routes", () => {
  assert.equal(shouldAutoEscalate(0.60), true);
  assert.equal(shouldAutoEscalate(0.69), true);
  assert.equal(shouldAutoEscalate(0.70), false);
  assert.equal(shouldAutoEscalate(0.72), false);
});
