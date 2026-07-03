/**
 * Pinia store — single source of truth mirroring the backend Engine over
 * WebSockets. Every canvas interaction is an optimistic local mutation plus a
 * command frame; the authoritative "state" broadcast reconciles all clients
 * (including the headless CLI's view of the world).
 */
import { defineStore } from "pinia";
import type {
  ArchTemplate, ClientMessage, EdgeSpec, MetricEvent, ModelVariant, NodeDescriptor,
  NodeInstanceSpec, PipelineStateSnapshot, ServerMessage, VariantMetric,
} from "../types";

const WS_URL = import.meta.env.VITE_WS_URL ?? `ws://${location.hostname}:8881`;
const METRIC_HISTORY = 300;
const ANALYTICS_HISTORY = 50_000; // non-truncated historic charts (safety cap)
const GRAPH_SAVE_DEBOUNCE_MS = 600; // coalesce drag/param bursts into one save

export interface ChatMessage {
  role: "user" | "assistant" | "tool";
  text: string;
  streaming?: boolean;
  variantId?: string;
}

export const usePipelineStore = defineStore("pipeline", {
  state: () => ({
    connected: false,
    state: null as PipelineStateSnapshot | null,
    catalog: [] as NodeDescriptor[],
    templates: [] as ArchTemplate[],
    metrics: {} as Record<string, MetricEvent[]>, // keyed by metric name
    history: {} as Record<string, MetricEvent[]>, // non-truncated analytics series
    nodeProgress: {} as Record<string, number>,   // nodeId → 0..1 live progress
    nodeTimes: {} as Record<string, number>,      // nodeId → execution_time_ms
    nodeElapsed: {} as Record<string, number>,    // nodeId → elapsedTimeMs (live)
    nodeEta: {} as Record<string, number>,        // nodeId → windowed etaMs
    workers: [] as Array<{ id: string; capabilities: Record<string, unknown>; connectedAt: string }>,
    logs: [] as Array<{ nodeId: string; message: string }>,
    selectedNodeId: null as string | null,
    lastError: "" as string,
    // ── model library ──
    library: [] as ModelVariant[],
    variantMetrics: {} as Record<string, VariantMetric[]>, // live sparklines
    /** Variant the user tried to open while a heavy process holds memory. */
    pendingOpenId: null as string | null,
    // ── chat sandbox ──
    chatMessages: [] as ChatMessage[],
    activeChatId: null as string | null,
    chatVariantId: null as string | null,
    mcpLog: [] as unknown[],
    mcpTools: [] as Array<{ name: string; description: string }>,
    // ── project persistence (SQLite warehouse) ──
    projects: [] as Array<{ id: string; name: string; updatedAt: number }>,
    activeProjectId: "default" as string,
    _saveTimer: null as ReturnType<typeof setTimeout> | null,
    _ws: null as WebSocket | null,
  }),

  getters: {
    selectedNode(s) {
      return s.state?.nodes.find((n) => n.id === s.selectedNodeId) ?? null;
    },
    selectedDescriptor(s): NodeDescriptor | null {
      const node = s.state?.nodes.find((n) => n.id === s.selectedNodeId);
      return node ? s.catalog.find((d) => d.type === node.type) ?? null : null;
    },
    latestMetric: (s) => (name: string): MetricEvent | null => {
      const arr = s.metrics[name];
      return arr?.length ? arr[arr.length - 1] : null;
    },
    /** Non-null while an active process owns the memory context. */
    busyReason(s): string | null {
      if (s.state?.running) return "a pipeline is executing (BPE tokenization / training)";
      if (s.library.some((v) => v.training)) return "a model variant is training";
      return null;
    },
  },

  actions: {
    connect() {
      if (this._ws) return;
      const ws = new WebSocket(WS_URL);
      this._ws = ws;
      ws.onopen = () => {
        this.connected = true;
        // E2E canvas restore: hydrate the project list and replay the active
        // project's persisted graph (SQLite graph_nodes/graph_edges rows) —
        // the server rebuilds the engine graph and broadcasts a full state
        // snapshot carrying every node's params + exact (x, y) position.
        this.restoreActiveProject();
      };
      ws.onclose = () => {
        this.connected = false;
        this._ws = null;
        setTimeout(() => this.connect(), 2000); // auto-reconnect
      };
      ws.onmessage = (e) => this.handle(JSON.parse(e.data) as ServerMessage);
    },

    /**
     * Trigger the backend load_project op for the active project. The server
     * reads the stored graph_nodes/graph_edges rows, re-instantiates the
     * engine graph (params + positions intact) and broadcasts "state";
     * idempotent — a missing project simply adopts the current canvas.
     */
    restoreActiveProject() {
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
      this.send({ op: "list_projects" });
      this.send({ op: "load_project", projectId: this.activeProjectId, name: this.state?.name });
    },

    handle(msg: ServerMessage) {
      switch (msg.ev) {
        case "state":
          this.state = msg.state;
          break;
        case "catalog":
          this.catalog = msg.descriptors;
          break;
        case "templates":
          this.templates = msg.templates;
          break;
        case "metric": {
          if (msg.metric.name === "node_progress") {
            this.nodeProgress[msg.metric.nodeId] = msg.metric.value;
            break;
          }
          if (msg.metric.name === "execution_time_ms") {
            this.nodeTimes[msg.metric.nodeId] = msg.metric.value;
            delete this.nodeEta[msg.metric.nodeId]; // node finished — clear ETA
            break;
          }
          if (msg.metric.name === "elapsed_ms") {
            this.nodeElapsed[msg.metric.nodeId] = msg.metric.value;
            break;
          }
          if (msg.metric.name === "eta_ms") {
            this.nodeEta[msg.metric.nodeId] = msg.metric.value;
            break;
          }
          const arr = (this.metrics[msg.metric.name] ??= []);
          arr.push(msg.metric);
          if (arr.length > METRIC_HISTORY) arr.shift();
          const hist = (this.history[msg.metric.name] ??= []);
          hist.push(msg.metric);
          if (hist.length > ANALYTICS_HISTORY) hist.shift();
          break;
        }
        case "log":
          this.logs.push(msg);
          if (this.logs.length > 500) this.logs.shift();
          break;
        case "error":
          this.lastError = msg.message;
          break;
        case "library":
          this.library = msg.variants;
          if (!this.chatVariantId && msg.variants.length) {
            this.chatVariantId = msg.variants[0].id;
          }
          break;
        case "variant_metric": {
          const arr = (this.variantMetrics[msg.metric.variantId] ??= []);
          arr.push(msg.metric);
          if (arr.length > METRIC_HISTORY) arr.shift();
          break;
        }
        case "chat_token": {
          if (msg.chatId !== this.activeChatId) break;
          const last = this.chatMessages[this.chatMessages.length - 1];
          if (last?.role === "assistant" && last.streaming) last.text += msg.text;
          break;
        }
        case "chat_done": {
          if (msg.chatId !== this.activeChatId) break;
          const last = this.chatMessages[this.chatMessages.length - 1];
          if (last?.streaming) last.streaming = false;
          if (msg.reason === "error" && msg.message) this.lastError = msg.message;
          this.activeChatId = null;
          break;
        }
        case "mcp_result": {
          this.mcpLog.push(msg.payload);
          // Keep the tool catalog fresh when a tools/list result arrives.
          const result = (msg.payload as { result?: { tools?: Array<{ name: string; description: string }> } })?.result;
          if (result?.tools) this.mcpTools = result.tools;
          this.chatMessages.push({
            role: "tool",
            text: JSON.stringify(msg.payload, null, 2),
          });
          break;
        }
        case "workers":
          this.workers = msg.workers;
          break;
        case "projects":
          this.projects = msg.projects;
          break;
        case "worker_event":
          break; // reserved: remote-worker telemetry mirror
      }
    },

    send(msg: ClientMessage) {
      this._ws?.send(JSON.stringify(msg));
    },

    /**
     * Debounced graph persistence: every create/rename/delete/param-update/
     * node-move coalesces into ONE save_graph frame; the server replaces the
     * project's layout state transactionally in the SQLite warehouse.
     */
    scheduleGraphSave() {
      if (this._saveTimer) clearTimeout(this._saveTimer);
      this._saveTimer = setTimeout(() => {
        this._saveTimer = null;
        this.send({ op: "save_graph", projectId: this.activeProjectId, name: this.state?.name });
      }, GRAPH_SAVE_DEBOUNCE_MS);
    },

    loadProject(projectId: string) {
      this.switchGraphProject(projectId);
    },

    /**
     * Switch the canvas to another project's individually stored graph.
     * Any pending debounced save is flushed under the OLD project first so
     * edits are never written into the wrong workspace; the server then
     * restores the target graph (or adopts the current one on first open).
     */
    switchGraphProject(projectId: string, name?: string) {
      if (projectId === this.activeProjectId) return;
      if (this._saveTimer) {
        clearTimeout(this._saveTimer);
        this._saveTimer = null;
        this.send({ op: "save_graph", projectId: this.activeProjectId, name: this.state?.name });
      }
      this.activeProjectId = projectId;
      this.metrics = {};
      this.selectedNodeId = null;
      this.send({ op: "load_project", projectId, name });
    },

    run() {
      this.metrics = {};
      this.history = {};
      this.nodeProgress = {};
      this.nodeTimes = {};
      this.nodeElapsed = {};
      this.nodeEta = {};
      this.send({ op: "run" });
    },
    stop() {
      this.send({ op: "stop" });
    },
    pauseTraining() {
      this.send({ op: "pause_training" });
    },
    resumeTraining() {
      this.send({ op: "resume_training" });
    },
    cancelTraining() {
      this.send({ op: "cancel_training" });
    },
    /** "Commit & Proceed": freeze weights/moments now, node completes as done. */
    commitTraining() {
      this.send({ op: "commit_training" });
    },
    /**
     * Real-time lr hot-tuning: the running trainer reads the new scalar at
     * the top of its NEXT step — no pause, no halt, no step-metric reset.
     */
    updateLearningRate(lr: number) {
      this.send({ op: "update_learning_rate", lr });
    },

    // ── interactive graph editing (optimism deferred to the state broadcast —
    //    the engine validates types/cycles and is the source of truth) ──
    addNode(node: NodeInstanceSpec) {
      this.send({ op: "add_node", node });
      this.scheduleGraphSave();
    },
    removeNode(nodeId: string) {
      if (this.selectedNodeId === nodeId) this.selectedNodeId = null;
      this.send({ op: "remove_node", nodeId });
      this.scheduleGraphSave();
    },
    addEdge(edge: EdgeSpec) {
      this.send({ op: "add_edge", edge });
      this.scheduleGraphSave();
    },
    removeEdge(edge: EdgeSpec) {
      this.send({ op: "remove_edge", edge });
      this.scheduleGraphSave();
    },
    applyTemplate(templateId: string) {
      this.metrics = {};
      this.selectedNodeId = null;
      this.send({ op: "apply_template", templateId });
      this.scheduleGraphSave();
    },

    updateParams(nodeId: string, params: Record<string, unknown>) {
      // Optimistic local apply for a snappy property panel.
      const node = this.state?.nodes.find((n) => n.id === nodeId);
      if (node) node.params = { ...node.params, ...params };
      this.send({ op: "update_params", nodeId, params });
      this.scheduleGraphSave();
    },

    moveNode(nodeId: string, position: { x: number; y: number }) {
      const node = this.state?.nodes.find((n) => n.id === nodeId);
      if (node) node.position = position;
      this.send({ op: "move_node", nodeId, position });
      this.scheduleGraphSave();
    },

    // ── model library ──
    cloneVariant(sourceId: string, name: string, overrides: Record<string, unknown> = {}) {
      this.send({ op: "library_clone", sourceId, name, overrides });
    },
    createVariant(name: string, overrides: Record<string, unknown> = {}) {
      this.send({ op: "library_create", name, overrides });
    },
    deleteVariant(variantId: string) {
      if (this.chatVariantId === variantId) this.chatVariantId = null;
      if (this.activeProjectId === variantId) this.activeProjectId = "default";
      this.send({ op: "library_delete", variantId });
      this.send({ op: "delete_project", projectId: variantId }); // drop its stored graph
    },
    renameVariant(variantId: string, name: string) {
      this.send({ op: "library_rename", variantId, name });
      this.send({ op: "rename_project", projectId: variantId, name }); // keep stored graph in sync
    },
    /**
     * Active-process blocker: opening another model while BPE tokenization or
     * training runs is intercepted — the UI must get an explicit force-cancel
     * confirmation before the active memory context is destroyed.
     */
    openVariant(variantId: string) {
      if (variantId === this.chatVariantId) return;
      if (this.busyReason) {
        this.pendingOpenId = variantId;
        return;
      }
      this.chatVariantId = variantId;
      this.activateVariantGraph(variantId);
    },
    /** Each library variant owns its own SQLite-stored graph configuration. */
    activateVariantGraph(variantId: string) {
      const variant = this.library.find((v) => v.id === variantId);
      this.switchGraphProject(variantId, variant?.name);
    },
    confirmForceOpen() {
      // Destroy the active context: abort the pipeline (frees worker threads /
      // GPU buffers) and stop every training variant.
      this.send({ op: "stop" });
      for (const v of this.library) {
        if (v.training) this.send({ op: "library_stop_train", variantId: v.id });
      }
      if (this.pendingOpenId) {
        this.chatVariantId = this.pendingOpenId;
        this.activateVariantGraph(this.pendingOpenId);
      }
      this.pendingOpenId = null;
    },
    cancelPendingOpen() {
      this.pendingOpenId = null;
    },
    trainVariant(variantId: string, steps = 30) {
      this.variantMetrics[variantId] = [];
      this.send({ op: "library_train", variantId, steps });
    },
    stopVariantTraining(variantId: string) {
      this.send({ op: "library_stop_train", variantId });
    },

    // ── chat sandbox ──
    sendChat(prompt: string, maxTokens = 96, temperature = 0.8, topP = 1) {
      if (!this.chatVariantId || this.activeChatId) return;
      const chatId = crypto.randomUUID();
      this.activeChatId = chatId;
      this.chatMessages.push({ role: "user", text: prompt });
      this.chatMessages.push({
        role: "assistant", text: "", streaming: true, variantId: this.chatVariantId,
      });
      this.send({
        op: "chat_send", chatId, variantId: this.chatVariantId, prompt, maxTokens, temperature, topP,
      });
    },
    stopChat() {
      if (this.activeChatId) this.send({ op: "chat_stop", chatId: this.activeChatId });
    },
    sendMcp(payload: unknown) {
      this.send({ op: "mcp", payload });
    },
  },
});
