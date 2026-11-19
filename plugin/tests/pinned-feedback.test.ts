import test, { after } from "node:test";
import assert from "node:assert/strict";
import fs, { appendFile } from "node:fs/promises";
import { syncBuiltinESMExports } from "node:module";

import plugin, { __testHelpers } from "../src/index.ts";

// This test worker exercises real hooks. Never put synthetic hook entries in the
// live plugin log used by the host's route checker.
const realAppend = fs.appendFile;
fs.appendFile = async (file, ...opts) => {
  if (file !== "/tmp/nexus-router-hook.log") await realAppend(file, ...opts);
};
syncBuiltinESMExports();
assert.notEqual(appendFile, realAppend);
after(() => { fs.appendFile = realAppend; syncBuiltinESMExports(); });

const pinned = {
  sessionKey: "agent:test:telegram:direct:123456",
  sessionId: "test-session",
  agentId: "test",
  trigger: "user",
  modelProviderId: "openai",
  modelId: "pinned-model",
};
const route = {
  decision_id: "test-decision",
  task_type: "reasoning",
  selected_model: "nvidia/suggested-model",
  selected_provider: "nvidia",
  confidence: 0.91,
  fallbacks: [],
  score: 0.9,
  reason: [],
  classifier_source: "local",
};

function fixture(t, routeMode?: string) {
  __testHelpers.resetInMemoryRoutingState();
  const hooks = new Map();
  const calls: { path: string; body: any }[] = [];
  let command: any;
  t.mock.method(globalThis, "fetch", async (url, init) => {
    const path = new URL(String(url)).pathname;
    if (init?.method !== "POST") {
      return path === "/route-mode" && routeMode
        ? Response.json({ preference: { mode: routeMode, updated_at: new Date().toISOString() } })
        : Response.json({}, { status: 404 });
    }
    calls.push({ path, body: JSON.parse(String(init.body)) });
    return Response.json(path === "/route" ? route : { ok: true });
  });
  plugin.register({
    config: { agents: { defaults: { model: "nvidia/default" } } },
    on: (name, cb) => hooks.set(name, cb),
    registerCommand: (registered) => { command = registered.handler; },
  });
  return {
    calls,
    invoke: (name: string, event: any, ctx = pinned) => hooks.get(name)(event, ctx),
    command: (args: string, sessionKey = pinned.sessionKey) => command({ args, sessionKey, channelId: "telegram" }),
    dispatch: async (senderId = "telegram:123456", ctx = pinned) => {
      await hooks.get("before_dispatch")(
        { body: "Compare the two approaches." },
        { channelId: "telegram", sessionKey: ctx.sessionKey, senderId, conversationId: senderId },
      );
    },
  };
}

const successfulReply = { success: true, messages: [{ role: "assistant", stopReason: "stop" }] };

test("pinned Telegram replies collect shadow feedback after completion without overriding or recording a fallback", async (t) => {
  const f = fixture(t);
  await f.dispatch();
  assert.equal(await f.invoke("before_model_resolve", { prompt: "compiled" }), undefined);
  // Retries may fire model resolution again, without a fresh Telegram message.
  assert.equal(await f.invoke("before_model_resolve", { prompt: "retry" }), undefined);
  assert.deepEqual(f.calls, []);
  await f.invoke("llm_output", { sessionId: pinned.sessionId, model: "actual-model", provider: "openai" });
  await f.invoke("agent_end", successfulReply);
  await f.invoke("agent_end", successfulReply);
  assert.deepEqual(f.calls.map(({ path }) => path), ["/route", "/api/v1/router/feedback-card"]);
  assert.equal(f.calls[0].body.message, "Compare the two approaches.");
  assert.equal(f.calls[0].body.mode, "shadow");
  assert.equal(f.calls[0].body.route_mode, "auto");
  assert.equal(f.calls[1].body.provenance_mode, "shadow");
  assert.equal(f.calls[1].body.telegram_user_id, "123456");
  assert.equal(f.calls[1].body.actual_model, "openai/actual-model");
  assert.match((await f.command("last")).text, /Applied: no\nSkip reason: session model pin/);
});

test("pinned probe does not send for compiled-only turns, another sender, or a failed reply", async (t) => {
  const f = fixture(t);
  await f.invoke("before_model_resolve", { prompt: "compiled only" });
  await f.invoke("agent_end", successfulReply);
  await f.dispatch("123457");
  await f.invoke("before_model_resolve", { prompt: "mismatched sender" });
  await f.invoke("agent_end", successfulReply);
  await f.dispatch();
  await f.invoke("before_model_resolve", { prompt: "failed reply" });
  await f.invoke("agent_end", { success: false });
  assert.deepEqual(f.calls, []);
});

test("pinned shadow feedback honors session feedback off", async (t) => {
  const f = fixture(t);
  try {
    await f.command("feedback off");
    await f.dispatch();
    assert.equal(await f.invoke("before_model_resolve", { prompt: "compiled" }), undefined);
    await f.invoke("agent_end", successfulReply);
    assert.deepEqual(f.calls, []);
  } finally {
    await f.command("feedback on");
  }
});

test("ordinary auto routing still overrides and logs an actual outcome", async (t) => {
  const f = fixture(t);
  const unpinned = { ...pinned, modelProviderId: "nvidia", modelId: "default" };
  await f.dispatch();
  assert.deepEqual(await f.invoke("before_model_resolve", { prompt: "compiled" }, unpinned), {
    providerOverride: "nvidia", modelOverride: "suggested-model",
  });
  await f.invoke("agent_end", successfulReply, unpinned);
  assert.deepEqual(f.calls.map(({ path }) => path), ["/route", "/api/v1/router/feedback-card", "/outcome"]);
  assert.equal(f.calls[0].body.mode, "route");
});

test("route off still records a shadow decision and does not override the model", async (t) => {
  const f = fixture(t, "off");
  await f.dispatch();
  assert.equal(await f.invoke("before_model_resolve", { prompt: "compiled" }), undefined);
  assert.deepEqual(f.calls.map(({ path }) => path), ["/route"]);
  await f.invoke("agent_end", successfulReply);
  assert.deepEqual(f.calls.map(({ path }) => path), ["/route", "/outcome", "/api/v1/router/feedback-card"]);
  assert.equal(f.calls[1].body.success, true);
  assert.equal(f.calls[2].body.provenance_mode, "shadow");
});
