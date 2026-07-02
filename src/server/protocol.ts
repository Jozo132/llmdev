/**
 * WebSocket wire protocol shared by server and Vue client.
 * Every UI action maps 1:1 onto an engine operation, so the canvas and the
 * headless CLI are always looking at the same graph.
 */
import type { MetricEvent, NodeDescriptor, PipelineSpec, PipelineStateSnapshot } from "../core/types.js";
import type { ModelVariant, VariantMetric } from "../core/LibraryManager.js";

// Client → Server
export type ClientMessage =
  | { op: "get_state" }
  | { op: "get_catalog" }                                     // node palette
  | { op: "load_pipeline"; spec: PipelineSpec }
  | { op: "run" }
  | { op: "stop" }
  | { op: "update_params"; nodeId: string; params: Record<string, unknown> }
  | { op: "move_node"; nodeId: string; position: { x: number; y: number } }
  // ── model library ──
  | { op: "library_list" }
  | { op: "library_clone"; sourceId: string; name: string; overrides?: Record<string, unknown> }
  | { op: "library_train"; variantId: string; steps?: number; batchSize?: number; lr?: number }
  | { op: "library_stop_train"; variantId: string }
  // ── chat sandbox (token-by-token local inference) ──
  | { op: "chat_send"; chatId: string; variantId: string; prompt: string; maxTokens?: number; temperature?: number }
  | { op: "chat_stop"; chatId: string }
  // ── MCP: JSON-RPC 2.0 envelope (initialize / tools/list / tools/call) ──
  | { op: "mcp"; payload: unknown };

// Server → Client
export type ServerMessage =
  | { ev: "state"; state: PipelineStateSnapshot }
  | { ev: "catalog"; descriptors: NodeDescriptor[] }
  | { ev: "metric"; metric: MetricEvent }
  | { ev: "log"; nodeId: string; message: string }
  | { ev: "error"; message: string }
  | { ev: "library"; variants: ModelVariant[] }
  | { ev: "variant_metric"; metric: VariantMetric }
  | { ev: "chat_token"; chatId: string; token: number; text: string }
  | { ev: "chat_done"; chatId: string; reason: "complete" | "stopped" | "error"; message?: string }
  | { ev: "mcp_result"; payload: unknown };
