/**
 * WebSocket wire protocol shared by server and Vue client.
 * Every UI action maps 1:1 onto an engine operation, so the canvas and the
 * headless CLI are always looking at the same graph.
 */
import type { MetricEvent, NodeDescriptor, NodeInstanceSpec, EdgeSpec, PipelineSpec, PipelineStateSnapshot } from "../core/types.js";
import type { ModelVariant, VariantMetric } from "../core/LibraryManager.js";
import type { ArchTemplate } from "../core/templates.js";

// Client → Server
export type ClientMessage =
  | { op: "get_state" }
  | { op: "get_catalog" }                                     // node palette
  | { op: "load_pipeline"; spec: PipelineSpec }
  | { op: "run" }
  | { op: "stop" }
  // ── runtime execution control ──
  | { op: "pause_training" }   // suspend between steps, keep optimizer state hot
  | { op: "resume_training" }
  | { op: "cancel_training" }  // abort + release GPU context / host buffers
  | { op: "commit_training" }  // "Commit & Proceed": freeze weights, finish node as done
  | { op: "update_learning_rate"; lr: number } // hot-tune Adam lr mid-run, no pause/reset
  | { op: "update_params"; nodeId: string; params: Record<string, unknown> }
  | { op: "move_node"; nodeId: string; position: { x: number; y: number } }
  // ── interactive graph editing ──
  | { op: "add_node"; node: NodeInstanceSpec }
  | { op: "remove_node"; nodeId: string }
  | { op: "add_edge"; edge: EdgeSpec }
  | { op: "remove_edge"; edge: EdgeSpec }
  // ── architectural templates ──
  | { op: "get_templates" }
  | { op: "apply_template"; templateId: string }
  // ── model library ──
  | { op: "library_list" }
  | { op: "library_create"; name: string; overrides?: Record<string, unknown> }
  | { op: "library_delete"; variantId: string }
  | { op: "library_rename"; variantId: string; name: string }
  | { op: "library_clone"; sourceId: string; name: string; overrides?: Record<string, unknown> }
  | { op: "library_train"; variantId: string; steps?: number; batchSize?: number; lr?: number }
  | { op: "library_stop_train"; variantId: string }
  // ── chat sandbox (token-by-token local inference) ──
  | { op: "chat_send"; chatId: string; variantId: string; prompt: string; maxTokens?: number; temperature?: number; topP?: number }
  | { op: "chat_stop"; chatId: string }
  // ── MCP: JSON-RPC 2.0 envelope (initialize / tools/list / tools/call) ──
  | { op: "mcp"; payload: unknown }
  // ── project / visual-graph persistence (SQLite warehouse) ──
  | { op: "save_graph"; projectId?: string; name?: string }   // persist current engine graph
  | { op: "load_project"; projectId: string; name?: string }  // restore (or adopt current graph on first open)
  | { op: "list_projects" }
  | { op: "rename_project"; projectId: string; name: string }
  | { op: "delete_project"; projectId: string };

// Server → Client
export type ServerMessage =
  | { ev: "state"; state: PipelineStateSnapshot }
  | { ev: "catalog"; descriptors: NodeDescriptor[] }
  | { ev: "templates"; templates: ArchTemplate[] }
  | { ev: "metric"; metric: MetricEvent }
  | { ev: "log"; nodeId: string; message: string }
  | { ev: "error"; message: string }
  | { ev: "library"; variants: ModelVariant[] }
  | { ev: "variant_metric"; metric: VariantMetric }
  | { ev: "chat_token"; chatId: string; token: number; text: string }
  | { ev: "chat_done"; chatId: string; reason: "complete" | "stopped" | "error"; message?: string }
  | { ev: "mcp_result"; payload: unknown }
  | { ev: "projects"; projects: Array<{ id: string; name: string; updatedAt: number }> }
  // ── horizontal scaling: remote compute workers ──
  | { ev: "workers"; workers: Array<{ id: string; capabilities: Record<string, unknown>; connectedAt: string }> }
  | { ev: "worker_event"; workerId: string; payload: unknown };
