/**
 * WebSocket bridge — serializes the live node-tree, streams training metrics
 * (loss, tokens/sec, VRAM), and accepts execution/edit commands from clients.
 *
 *   npm run server            → ws://localhost:8081
 *
 * The same Engine instance backs both this server and the CLI, so a pipeline
 * started headlessly is mirrored on every connected canvas in real time.
 */
import { readFileSync } from "node:fs";
import { WebSocketServer, WebSocket } from "ws";
import "../nodes/index.js"; // register built-in node types
import { Engine } from "../core/Engine.js";
import { LibraryManager } from "../core/LibraryManager.js";
import { listDescriptors } from "../core/Registry.js";
import { cudaAvailable, cudaDeviceName } from "../ml/backend.js";
import type { PipelineSpec } from "../core/types.js";
import { createMcpHandler } from "./mcp.js";
import type { ClientMessage, ServerMessage } from "./protocol.js";

const PORT = Number(process.env.LLMDEV_WS_PORT ?? 8081);
const engine = new Engine();
const library = new LibraryManager(engine.artifactsDir);
const mcp = createMcpHandler(library);

// Active chat generations, cancellable per chatId.
const chatAborts = new Map<string, AbortController>();

// Optionally preload a pipeline: `npm run server -- pipelines/poc-js-1m.json`
const preload = process.argv[2];
if (preload) {
  engine.load(JSON.parse(readFileSync(preload, "utf8")) as PipelineSpec);
  console.log(`Preloaded pipeline: ${preload}`);
}

const wss = new WebSocketServer({ port: PORT });
const clients = new Set<WebSocket>();

function broadcast(msg: ServerMessage): void {
  const data = JSON.stringify(msg);
  for (const ws of clients) {
    if (ws.readyState === WebSocket.OPEN) ws.send(data);
  }
}

// Engine events → all clients.
engine.on("state", (state) => broadcast({ ev: "state", state }));
engine.on("metric", (metric) => broadcast({ ev: "metric", metric }));
engine.on("log", ({ nodeId, message }) => {
  console.log(`[${nodeId}] ${message}`);
  broadcast({ ev: "log", nodeId, message });
});

// Library events → all clients (variant list changes + live benchmark points).
library.on("library", (variants) => broadcast({ ev: "library", variants }));
library.on("metric", (metric) => broadcast({ ev: "variant_metric", metric }));

// ── Token-by-token local chat inference ──────────────────────────────
async function runChat(
  chatId: string, variantId: string, prompt: string,
  maxTokens: number, temperature: number
): Promise<void> {
  const abort = new AbortController();
  chatAborts.set(chatId, abort);
  try {
    const { model, tokenizer } = await library.loadForInference(variantId);
    const ids: number[] = Array.from(tokenizer.encode(prompt));
    for (let n = 0; n < maxTokens; n++) {
      if (abort.signal.aborted) {
        broadcast({ ev: "chat_done", chatId, reason: "stopped" });
        return;
      }
      const token = model.nextToken(ids, temperature);
      ids.push(token);
      broadcast({ ev: "chat_token", chatId, token, text: tokenizer.decode([token]) });
      await new Promise((r) => setImmediate(r)); // stream frames between tokens
    }
    broadcast({ ev: "chat_done", chatId, reason: "complete" });
  } catch (err) {
    broadcast({
      ev: "chat_done", chatId, reason: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  } finally {
    chatAborts.delete(chatId);
  }
}

wss.on("connection", (ws) => {
  clients.add(ws);
  const send = (msg: ServerMessage) => ws.send(JSON.stringify(msg));
  // Sync new clients immediately.
  send({ ev: "catalog", descriptors: listDescriptors() });
  send({ ev: "state", state: engine.snapshot() });
  send({ ev: "library", variants: library.list() });

  ws.on("message", (raw) => {
    let msg: ClientMessage;
    try {
      msg = JSON.parse(String(raw)) as ClientMessage;
    } catch {
      return send({ ev: "error", message: "Invalid JSON" });
    }
    try {
      switch (msg.op) {
        case "get_state":
          return send({ ev: "state", state: engine.snapshot() });
        case "get_catalog":
          return send({ ev: "catalog", descriptors: listDescriptors() });
        case "load_pipeline":
          engine.load(msg.spec);
          return;
        case "run":
          if (!engine.running) {
            engine.run().catch((err) =>
              broadcast({ ev: "error", message: err instanceof Error ? err.message : String(err) })
            );
          }
          return;
        case "stop":
          return engine.stop();
        case "update_params":
          return engine.updateNodeParams(msg.nodeId, msg.params);
        case "move_node":
          return engine.updateNodePosition(msg.nodeId, msg.position);
        case "library_list":
          return send({ ev: "library", variants: library.list() });
        case "library_clone":
          library.clone(msg.sourceId, msg.name, msg.overrides ?? {});
          return;
        case "library_train":
          library
            .train(msg.variantId, { steps: msg.steps, batchSize: msg.batchSize, lr: msg.lr })
            .catch((err) =>
              send({ ev: "error", message: err instanceof Error ? err.message : String(err) })
            );
          return;
        case "library_stop_train":
          return library.stopTraining(msg.variantId);
        case "chat_send":
          void runChat(msg.chatId, msg.variantId, msg.prompt, msg.maxTokens ?? 96, msg.temperature ?? 0.8);
          return;
        case "chat_stop":
          chatAborts.get(msg.chatId)?.abort();
          return;
        case "mcp":
          void mcp(msg.payload).then((payload) => send({ ev: "mcp_result", payload }));
          return;
        default:
          return send({ ev: "error", message: `Unknown op` });
      }
    } catch (err) {
      send({ ev: "error", message: err instanceof Error ? err.message : String(err) });
    }
  });

  ws.on("close", () => clients.delete(ws));
});

console.log(`llmdev bridge listening on ws://localhost:${PORT}`);
console.log(
  cudaAvailable()
    ? `compute backend: CUDA — ${cudaDeviceName()}`
    : "compute backend: js-cpu (build the addon with `npm run native:build` for GPU)"
);
