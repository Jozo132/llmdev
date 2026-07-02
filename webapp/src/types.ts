/**
 * Client-side mirror of the server wire types (kept dependency-free so the
 * webapp doesn't need to compile backend sources).
 */
export type NodeStatus = "idle" | "running" | "done" | "error" | "skipped";

export interface ParamSchemaEntry {
  key: string;
  label: string;
  type: "string" | "number" | "boolean" | "select";
  options?: string[];
  default?: unknown;
  description?: string;
}

export interface PortSpec {
  name: string;
  dataType: string;
  required?: boolean;
}

export interface NodeDescriptor {
  type: string;
  label: string;
  category: "data" | "tokenizer" | "model" | "train" | "eval" | "export" | "custom";
  inputs: PortSpec[];
  outputs: PortSpec[];
  paramSchema: ParamSchemaEntry[];
}

export interface EdgeSpec {
  from: { node: string; port: string };
  to: { node: string; port: string };
}

export interface NodeStateSnapshot {
  id: string;
  type: string;
  label: string;
  status: NodeStatus;
  params: Record<string, unknown>;
  position?: { x: number; y: number };
  error?: string;
}

export interface PipelineStateSnapshot {
  name: string;
  running: boolean;
  nodes: NodeStateSnapshot[];
  edges: EdgeSpec[];
}

export interface MetricEvent {
  ts: number;
  nodeId: string;
  name: string;
  value: number;
  extra?: Record<string, unknown>;
}

export type ServerMessage =
  | { ev: "state"; state: PipelineStateSnapshot }
  | { ev: "catalog"; descriptors: NodeDescriptor[] }
  | { ev: "metric"; metric: MetricEvent }
  | { ev: "log"; nodeId: string; message: string }
  | { ev: "error"; message: string };

export type ClientMessage =
  | { op: "get_state" }
  | { op: "get_catalog" }
  | { op: "load_pipeline"; spec: { name: string; nodes: unknown[]; edges: EdgeSpec[] } }
  | { op: "run" }
  | { op: "stop" }
  | { op: "update_params"; nodeId: string; params: Record<string, unknown> }
  | { op: "move_node"; nodeId: string; position: { x: number; y: number } };
