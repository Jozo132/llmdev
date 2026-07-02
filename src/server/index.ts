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
import { listDescriptors } from "../core/Registry.js";
import type { PipelineSpec } from "../core/types.js";
import type { ClientMessage, ServerMessage } from "./protocol.js";

const PORT = Number(process.env.LLMDEV_WS_PORT ?? 8081);
const engine = new Engine();

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

wss.on("connection", (ws) => {
  clients.add(ws);
  const send = (msg: ServerMessage) => ws.send(JSON.stringify(msg));
  // Sync new clients immediately.
  send({ ev: "catalog", descriptors: listDescriptors() });
  send({ ev: "state", state: engine.snapshot() });

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
