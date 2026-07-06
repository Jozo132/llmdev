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
import { existsSync, readFileSync, unlinkSync } from "node:fs";
import path from "node:path";
import "../nodes/index.js"; // register built-in node types

// Load secrets (HF_TOKEN…) from the gitignored .env when run standalone.
try { process.loadEnvFile(".env"); } catch { /* no .env — fine */ }
import { Engine } from "../core/Engine.js";
import { LibraryManager, type ModelVariant } from "../core/LibraryManager.js";
import { getDatasetDB } from "../core/DatasetDB.js";
import { listDescriptors } from "../core/Registry.js";
import { TEMPLATES } from "../core/templates.js";
import { cudaAvailable, cudaDeviceName } from "../ml/backend.js";
import type { PipelineSpec, TrainedModelHandle } from "../core/types.js";
import { MessageBroker } from "./broker.js";
import { createMcpHandler } from "./mcp.js";
import type { ClientMessage, ModelAcceptanceProposal, ServerMessage } from "./protocol.js";

const PORT = Number(process.env.LLMDEV_WS_PORT ?? 8881);
const engine = new Engine();
const library = new LibraryManager(engine.artifactsDir);
const datasets = getDatasetDB(engine.artifactsDir);
const mcp = createMcpHandler(library);

// Active chat generations, cancellable per chatId.
const chatAborts = new Map<string, AbortController>();

const DEFAULT_PROJECT_ID = "default";
const pendingAcceptances = new Map<string, { projectId: string; handle: TrainedModelHandle; datasetLoss?: number }>();

function runtimeInfo(): { backend: string; deviceName?: string | null } {
  const deviceName = cudaAvailable() ? cudaDeviceName() : null;
  return { backend: deviceName ? "CUDA" : "js-cpu", deviceName };
}

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

function resetTrainerCheckpoint(nodeId: string): void {
  if (engine.running) throw new Error("Stop the pipeline before resetting model weights");
  const node = engine.snapshot().nodes.find((candidate) => candidate.id === nodeId);
  if (!node) throw new Error(`No such node: ${nodeId}`);
  if (node.type !== "train.poc") throw new Error("Reset model is only available on trainer nodes");
  const explicitCheckpoint = String(node.params.checkpoint ?? "").trim();
  const ckptName = explicitCheckpoint || `node-${nodeId}`;
  const ckptDir = path.join(engine.artifactsDir, "checkpoints");
  const files = [
    `${ckptName}.weights.bin`,
    `${ckptName}.adam.bin`,
    `${ckptName}.best.weights.bin`,
    `${ckptName}.best.adam.bin`,
  ].map((file) => path.join(ckptDir, file));
  let removed = 0;
  for (const file of files) {
    if (!existsSync(file)) continue;
    unlinkSync(file);
    removed++;
  }
  broker.broadcast({ ev: "log", nodeId, message: `Reset model checkpoint "${ckptName}" (${removed} file${removed === 1 ? "" : "s"} removed)` });
}

function techStack(config: Record<string, unknown>): string[] {
  const loraOn = config.fineTuneMode === "lora";
  const esOn = Boolean(config.stochasticExplorationPool);
  const mlp = String(config.mlp ?? "standard");
  const stack = [
    runtimeInfo().backend,
    String(config.mixer ?? "causal-mean"),
    mlp === "swiglu" ? "SwiGLU" : "MLP",
    String(config.loss ?? "cross-entropy"),
    loraOn ? "LoRA" : "Full Adam",
  ];
  if (esOn) stack.push(loraOn ? "ES population" : "ES temporal");
  return stack;
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function latestRunReportLoss(): number | undefined {
  const outputs = engine.outputsByNode();
  for (const nodeOutputs of Object.values(outputs)) {
    const report = nodeOutputs.report as { loss?: unknown } | undefined;
    const loss = finiteOrNull(report?.loss);
    if (loss != null) return loss;
  }
  return undefined;
}

function buildAcceptanceProposal(
  id: string, variant: ModelVariant, handle: TrainedModelHandle, datasetLoss?: number
): ModelAcceptanceProposal {
  const oldTrainLoss = finiteOrNull(variant.finalLoss);
  const newTrainLoss = finiteOrNull(handle.finalLoss);
  const oldDatasetLoss = finiteOrNull(variant.datasetLoss ?? variant.finalLoss);
  const newDatasetLoss = finiteOrNull(datasetLoss ?? handle.finalLoss);
  return {
    id,
    projectId: variant.id,
    modelName: variant.name,
    oldTrainLoss,
    newTrainLoss,
    trainLossDelta: oldTrainLoss != null && newTrainLoss != null ? newTrainLoss - oldTrainLoss : null,
    oldDatasetLoss,
    newDatasetLoss,
    datasetLossDelta: oldDatasetLoss != null && newDatasetLoss != null ? newDatasetLoss - oldDatasetLoss : null,
    oldTechStack: techStack(variant.config as unknown as Record<string, unknown>),
    newTechStack: techStack(handle.config as unknown as Record<string, unknown>),
  };
}

function createRunAcceptance(projectId?: string): void {
  if (!projectId || projectId === DEFAULT_PROJECT_ID) return;
  const variant = library.list().find((candidate) => candidate.id === projectId);
  if (!variant) return;
  const outputs = engine.outputsByNode();
  for (const nodeOutputs of Object.values(outputs)) {
    const model = nodeOutputs.model as TrainedModelHandle | undefined;
    if (!model?.weights || !model.config) continue;
    const datasetLoss = latestRunReportLoss() ?? model.finalLoss;
    const proposalId = `${projectId}:${Date.now().toString(36)}`;
    pendingAcceptances.set(proposalId, { projectId, handle: model, datasetLoss });
    broker.broadcast({
      ev: "model_acceptance_required",
      proposal: buildAcceptanceProposal(proposalId, variant, model, datasetLoss),
    });
    return;
  }
}

function acceptModelResult(proposalId: string): void {
  const pending = pendingAcceptances.get(proposalId);
  if (!pending) throw new Error("No pending model result to accept");
  library.persistPipelineResult(pending.projectId, pending.handle, pending.datasetLoss);
  pendingAcceptances.delete(proposalId);
  broker.broadcast({ ev: "log", nodeId: "", message: `Accepted model result for "${pending.projectId}"` });
  broker.broadcast({ ev: "library", variants: library.list() });
}

function rejectModelResult(proposalId: string): void {
  const pending = pendingAcceptances.get(proposalId);
  if (!pending) return;
  pendingAcceptances.delete(proposalId);
  broker.broadcast({ ev: "log", nodeId: "", message: `Rejected model result for "${pending.projectId}"; library weights unchanged` });
}

// Optionally preload a pipeline: `npm run server -- pipelines/poc-js-1m.json`
const preload = process.argv[2];
if (preload) {
  engine.load(JSON.parse(readFileSync(preload, "utf8")) as PipelineSpec);
  console.log(`Preloaded pipeline: ${preload}`);
}
// ── Token-by-token local chat inference ──────────────────────────────
function broadcastChats(): void {
  broker.broadcast({ ev: "chats", sessions: datasets.listChats() });
}

async function runChat(
  chatId: string, sessionId: string, variantId: string, prompt: string,
  maxTokens: number, temperature: number, topP: number
): Promise<void> {
  const abort = new AbortController();
  chatAborts.set(chatId, abort);
  let assistantText = "";
  try {
    const { model, tokenizer } = await library.loadForInference(variantId);
    const ids: number[] = Array.from(tokenizer.encode(prompt));
    for (let n = 0; n < maxTokens; n++) {
      if (abort.signal.aborted) {
        if (assistantText) datasets.appendChatMessage(sessionId, { role: "assistant", text: assistantText, variantId });
        broadcastChats();
        broker.broadcast({ ev: "chat_done", chatId, reason: "stopped" });
        return;
      }
      const token = model.nextToken(ids, temperature, topP);
      ids.push(token);
      const text = tokenizer.decode([token]);
      assistantText += text;
      broker.broadcast({ ev: "chat_token", chatId, token, text });
      await new Promise((r) => setImmediate(r)); // stream frames between tokens
    }
    datasets.appendChatMessage(sessionId, { role: "assistant", text: assistantText, variantId });
    broadcastChats();
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
        engine.run()
          .then(() => {
            createRunAcceptance(msg.projectId);
            broker.broadcast({ ev: "library", variants: library.list() });
          })
          .catch((err) =>
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
    case "accept_model_result":
      return acceptModelResult(msg.proposalId);
    case "reject_model_result":
      return rejectModelResult(msg.proposalId);
    case "update_learning_rate":
      return engine.setLearningRate(msg.lr); // hot lr swap, loop never pauses
    case "reset_model":
      return resetTrainerCheckpoint(msg.nodeId);
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
      if (!datasets.loadChat(msg.sessionId)) {
        datasets.createChat(msg.sessionId, msg.prompt.trim().slice(0, 48) || "New chat", msg.variantId);
      }
      datasets.appendChatMessage(msg.sessionId, { role: "user", text: msg.prompt, variantId: msg.variantId });
      send({ ev: "chats", sessions: datasets.listChats() });
      void runChat(msg.chatId, msg.sessionId, msg.variantId, msg.prompt, msg.maxTokens ?? 96, msg.temperature ?? 0.8, msg.topP ?? 1);
      return;
    case "chat_stop":
      chatAborts.get(msg.chatId)?.abort();
      return;
    case "list_chats":
      return send({ ev: "chats", sessions: datasets.listChats() });
    case "create_chat":
      datasets.createChat(msg.sessionId, msg.title ?? "New chat", msg.variantId ?? null);
      return send({ ev: "chats", sessions: datasets.listChats() });
    case "load_chat": {
      const session = datasets.loadChat(msg.sessionId);
      if (!session) throw new Error(`No chat session: ${msg.sessionId}`);
      return send({ ev: "chat_session", session });
    }
    case "rename_chat":
      datasets.renameChat(msg.sessionId, msg.title);
      return broker.broadcast({ ev: "chats", sessions: datasets.listChats() });
    case "delete_chat":
      datasets.deleteChat(msg.sessionId);
      return broker.broadcast({ ev: "chats", sessions: datasets.listChats() });
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
    send({ ev: "runtime", ...runtimeInfo() });
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
