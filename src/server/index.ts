/**
 * Compute coordinator — binds the Engine/Library/Warehouse to the transport-
 * agnostic MessageBroker.
 *
 *   npm run server            → http+ws://localhost:8881
 *
 * Planes (see broker.ts):
 *   WS  /          UI protocol (ClientMessage/ServerMessage)
 *   WS  /worker    remote compute workers (register / task / event)
 *   REST /api/*    state · catalog · templates · library · datasets · POST op
 *
 * The same Engine instance backs the broker and the CLI, so a pipeline
 * started headlessly is mirrored on every connected canvas in real time.
 */
import { readFileSync } from "node:fs";
import "../nodes/index.js"; // register built-in node types

// Load secrets (HF_TOKEN…) from the gitignored .env when run standalone.
try { process.loadEnvFile(".env"); } catch { /* no .env — fine */ }
import { Engine } from "../core/Engine.js";
import { LibraryManager } from "../core/LibraryManager.js";
import { getDatasetDB } from "../core/DatasetDB.js";
import { listDescriptors } from "../core/Registry.js";
import { TEMPLATES } from "../core/templates.js";
import { cudaAvailable, cudaDeviceName } from "../ml/backend.js";
import type { PipelineSpec } from "../core/types.js";
import { MessageBroker } from "./broker.js";
import { createMcpHandler } from "./mcp.js";
import type { ClientMessage, ServerMessage } from "./protocol.js";

const PORT = Number(process.env.LLMDEV_WS_PORT ?? 8881);
const engine = new Engine();
const library = new LibraryManager(engine.artifactsDir);
const datasets = getDatasetDB(engine.artifactsDir);
const mcp = createMcpHandler(library);

// Active chat generations, cancellable per chatId.
const chatAborts = new Map<string, AbortController>();

const DEFAULT_PROJECT_ID = "default";

/** Snapshot the live engine graph into the SQLite warehouse (one transaction). */
function persistGraph(projectId = DEFAULT_PROJECT_ID, name?: string): void {
  const snap = engine.snapshot();
  datasets.saveGraph(
    projectId,
    name ?? snap.name ?? "untitled",
    snap.nodes.map((n) => ({
      id: n.id,
      type: n.type,
      x: n.position?.x ?? 0,
      y: n.position?.y ?? 0,
      params: n.params ?? {},
    })),
    snap.edges.map((e) => ({
      id: `${e.from.node}.${e.from.port}->${e.to.node}.${e.to.port}`,
      from: e.from,
      to: e.to,
    }))
  );
  broker.broadcast({ ev: "projects", projects: datasets.listProjects() });
}

/** Rehydrate a saved project graph into the engine (⇒ state broadcast). */
function restoreProject(projectId: string): boolean {
  const graph = datasets.loadGraph(projectId);
  if (!graph) return false;
  const spec: PipelineSpec = {
    name: graph.project.name,
    nodes: graph.nodes.map((n) => ({
      id: n.id, type: n.type, params: n.params, position: { x: n.x, y: n.y },
    })),
    edges: graph.edges.map((e) => ({ from: e.from, to: e.to })),
  };
  engine.load(spec);
  return true;
}

// Optionally preload a pipeline: `npm run server -- pipelines/poc-js-1m.json`
const preload = process.argv[2];
if (preload) {
  engine.load(JSON.parse(readFileSync(preload, "utf8")) as PipelineSpec);
  console.log(`Preloaded pipeline: ${preload}`);
}
// ── Token-by-token local chat inference ──────────────────────────────
async function runChat(
  chatId: string, variantId: string, prompt: string,
  maxTokens: number, temperature: number, topP: number
): Promise<void> {
  const abort = new AbortController();
  chatAborts.set(chatId, abort);
  try {
    const { model, tokenizer } = await library.loadForInference(variantId);
    const ids: number[] = Array.from(tokenizer.encode(prompt));
    for (let n = 0; n < maxTokens; n++) {
      if (abort.signal.aborted) {
        broker.broadcast({ ev: "chat_done", chatId, reason: "stopped" });
        return;
      }
      const token = model.nextToken(ids, temperature, topP);
      ids.push(token);
      broker.broadcast({ ev: "chat_token", chatId, token, text: tokenizer.decode([token]) });
      await new Promise((r) => setImmediate(r)); // stream frames between tokens
    }
    broker.broadcast({ ev: "chat_done", chatId, reason: "complete" });
  } catch (err) {
    broker.broadcast({
      ev: "chat_done", chatId, reason: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  } finally {
    chatAborts.delete(chatId);
  }
}

// ── Coordinator op dispatch (shared by UI WS frames and REST POST /api/op) ──
function handleOp(msg: ClientMessage, send: (msg: ServerMessage) => void): void {
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
          broker.broadcast({ ev: "error", message: err instanceof Error ? err.message : String(err) })
        );
      }
      return;
    case "stop":
      return engine.stop();
    case "pause_training":
      return engine.pause();
    case "resume_training":
      return engine.resume();
    case "cancel_training":
      return engine.stop(); // abort + wake paused nodes; trainer frees GPU ctx
    case "commit_training":
      return engine.commitEarly(); // freeze weights/moments, node ends 'done'
    case "update_learning_rate":
      return engine.setLearningRate(msg.lr); // hot lr swap, loop never pauses
    case "update_params":
      return engine.updateNodeParams(msg.nodeId, msg.params);
    case "move_node":
      return engine.updateNodePosition(msg.nodeId, msg.position);
    case "add_node":
      return engine.addNode(msg.node);
    case "remove_node":
      return engine.removeNode(msg.nodeId);
    case "add_edge":
      return engine.addEdge(msg.edge);
    case "remove_edge":
      return engine.removeEdge(msg.edge);
    case "get_templates":
      return send({ ev: "templates", templates: TEMPLATES });
    case "apply_template": {
      const tpl = TEMPLATES.find((t) => t.id === msg.templateId);
      if (!tpl) throw new Error(`Unknown template: ${msg.templateId}`);
      // Deep-copy so canvas edits never mutate the pristine template.
      engine.load(JSON.parse(JSON.stringify(tpl.spec)));
      return;
    }
    case "library_list":
      return send({ ev: "library", variants: library.list() });
    case "library_create":
      library.create(msg.name, msg.overrides ?? {});
      return;
    case "library_delete":
      return library.delete(msg.variantId);
    case "library_rename":
      library.rename(msg.variantId, msg.name);
      return;
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
      void runChat(msg.chatId, msg.variantId, msg.prompt, msg.maxTokens ?? 96, msg.temperature ?? 0.8, msg.topP ?? 1);
      return;
    case "chat_stop":
      chatAborts.get(msg.chatId)?.abort();
      return;
    case "mcp":
      void mcp(msg.payload).then((payload) => send({ ev: "mcp_result", payload }));
      return;
    case "save_graph":
      return persistGraph(msg.projectId ?? DEFAULT_PROJECT_ID, msg.name);
    case "load_project":
      // First open for this project (e.g. a library variant that never had a
      // graph): adopt the current canvas as its initial configuration so
      // every variant owns an individually stored topology from then on.
      if (!restoreProject(msg.projectId)) persistGraph(msg.projectId, msg.name);
      return;
    case "list_projects":
      return send({ ev: "projects", projects: datasets.listProjects() });
    case "rename_project":
      datasets.renameProject(msg.projectId, msg.name);
      return broker.broadcast({ ev: "projects", projects: datasets.listProjects() });
    case "delete_project":
      datasets.deleteProject(msg.projectId);
      return broker.broadcast({ ev: "projects", projects: datasets.listProjects() });
    default:
      return send({ ev: "error", message: `Unknown op` });
  }
}

const broker = new MessageBroker({
  port: PORT,
  onOp: handleOp,
  restGet: {
    state: () => engine.snapshot(),
    catalog: () => listDescriptors(),
    templates: () => TEMPLATES,
    library: () => library.list(),
    datasets: () => datasets.list(),
    projects: () => datasets.listProjects(),
  },
  onUiConnect: (send) => {
    send({ ev: "catalog", descriptors: listDescriptors() });
    send({ ev: "templates", templates: TEMPLATES });
    send({ ev: "state", state: engine.snapshot() });
    send({ ev: "library", variants: library.list() });
    send({ ev: "workers", workers: broker.listWorkers() });
    send({ ev: "projects", projects: datasets.listProjects() });
  },
});

// Engine events → all UI clients.
engine.on("state", (state) => broker.broadcast({ ev: "state", state }));
engine.on("metric", (metric) => broker.broadcast({ ev: "metric", metric }));
engine.on("log", ({ nodeId, message }) => {
  console.log(`[${nodeId}] ${message}`);
  broker.broadcast({ ev: "log", nodeId, message });
});

// Library events → all UI clients (variant list changes + live benchmarks).
library.on("library", (variants) => broker.broadcast({ ev: "library", variants }));
library.on("metric", (metric) => broker.broadcast({ ev: "variant_metric", metric }));

broker.listen();

// Resume the last saved workspace unless a pipeline file was preloaded.
if (!preload && restoreProject(DEFAULT_PROJECT_ID)) {
  console.log(`Restored saved workspace "${DEFAULT_PROJECT_ID}" from warehouse`);
}

console.log(`llmdev coordinator listening on http+ws://localhost:${PORT}  (ws /  ·  ws /worker  ·  REST /api/*)`);
console.log(
  cudaAvailable()
    ? `compute backend: CUDA — ${cudaDeviceName()}`
    : "compute backend: js-cpu (build the addon with `npm run native:build` for GPU)"
);
