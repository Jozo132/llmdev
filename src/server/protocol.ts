/**
 * WebSocket wire protocol shared by server and Vue client.
 * Every UI action maps 1:1 onto an engine operation, so the canvas and the
 * headless CLI are always looking at the same graph.
 */
import type { MetricEvent, NodeDescriptor, PipelineSpec, PipelineStateSnapshot } from "../core/types.js";

// Client → Server
export type ClientMessage =
  | { op: "get_state" }
  | { op: "get_catalog" }                                     // node palette
  | { op: "load_pipeline"; spec: PipelineSpec }
  | { op: "run" }
  | { op: "stop" }
  | { op: "update_params"; nodeId: string; params: Record<string, unknown> }
  | { op: "move_node"; nodeId: string; position: { x: number; y: number } };

// Server → Client
export type ServerMessage =
  | { ev: "state"; state: PipelineStateSnapshot }
  | { ev: "catalog"; descriptors: NodeDescriptor[] }
  | { ev: "metric"; metric: MetricEvent }
  | { ev: "log"; nodeId: string; message: string }
  | { ev: "error"; message: string };
