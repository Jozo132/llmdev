/**
 * Pinia store — single source of truth mirroring the backend Engine over
 * WebSockets. Every canvas interaction is an optimistic local mutation plus a
 * command frame; the authoritative "state" broadcast reconciles all clients
 * (including the headless CLI's view of the world).
 */
import { defineStore } from "pinia";
import type {
  ClientMessage, MetricEvent, ModelVariant, NodeDescriptor, PipelineStateSnapshot,
  ServerMessage, VariantMetric,
} from "../types";

const WS_URL = import.meta.env.VITE_WS_URL ?? `ws://${location.hostname}:8081`;
const METRIC_HISTORY = 300;

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
    metrics: {} as Record<string, MetricEvent[]>, // keyed by metric name
    logs: [] as Array<{ nodeId: string; message: string }>,
    selectedNodeId: null as string | null,
    lastError: "" as string,
    // ── model library ──
    library: [] as ModelVariant[],
    variantMetrics: {} as Record<string, VariantMetric[]>, // live sparklines
    // ── chat sandbox ──
    chatMessages: [] as ChatMessage[],
    activeChatId: null as string | null,
    chatVariantId: null as string | null,
    mcpLog: [] as unknown[],
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
  },

  actions: {
    connect() {
      if (this._ws) return;
      const ws = new WebSocket(WS_URL);
      this._ws = ws;
      ws.onopen = () => (this.connected = true);
      ws.onclose = () => {
        this.connected = false;
        this._ws = null;
        setTimeout(() => this.connect(), 2000); // auto-reconnect
      };
      ws.onmessage = (e) => this.handle(JSON.parse(e.data) as ServerMessage);
    },

    handle(msg: ServerMessage) {
      switch (msg.ev) {
        case "state":
          this.state = msg.state;
          break;
        case "catalog":
          this.catalog = msg.descriptors;
          break;
        case "metric": {
          const arr = (this.metrics[msg.metric.name] ??= []);
          arr.push(msg.metric);
          if (arr.length > METRIC_HISTORY) arr.shift();
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
        case "mcp_result":
          this.mcpLog.push(msg.payload);
          this.chatMessages.push({
            role: "tool",
            text: JSON.stringify(msg.payload, null, 2),
          });
          break;
      }
    },

    send(msg: ClientMessage) {
      this._ws?.send(JSON.stringify(msg));
    },

    run() {
      this.metrics = {};
      this.send({ op: "run" });
    },
    stop() {
      this.send({ op: "stop" });
    },

    updateParams(nodeId: string, params: Record<string, unknown>) {
      // Optimistic local apply for a snappy property panel.
      const node = this.state?.nodes.find((n) => n.id === nodeId);
      if (node) node.params = { ...node.params, ...params };
      this.send({ op: "update_params", nodeId, params });
    },

    moveNode(nodeId: string, position: { x: number; y: number }) {
      const node = this.state?.nodes.find((n) => n.id === nodeId);
      if (node) node.position = position;
      this.send({ op: "move_node", nodeId, position });
    },

    // ── model library ──
    cloneVariant(sourceId: string, name: string, overrides: Record<string, unknown> = {}) {
      this.send({ op: "library_clone", sourceId, name, overrides });
    },
    trainVariant(variantId: string, steps = 30) {
      this.variantMetrics[variantId] = [];
      this.send({ op: "library_train", variantId, steps });
    },
    stopVariantTraining(variantId: string) {
      this.send({ op: "library_stop_train", variantId });
    },

    // ── chat sandbox ──
    sendChat(prompt: string, maxTokens = 96, temperature = 0.8) {
      if (!this.chatVariantId || this.activeChatId) return;
      const chatId = crypto.randomUUID();
      this.activeChatId = chatId;
      this.chatMessages.push({ role: "user", text: prompt });
      this.chatMessages.push({
        role: "assistant", text: "", streaming: true, variantId: this.chatVariantId,
      });
      this.send({
        op: "chat_send", chatId, variantId: this.chatVariantId, prompt, maxTokens, temperature,
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
