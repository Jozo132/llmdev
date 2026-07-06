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
  /** Show only when the selected node params match these literal/regex rules. */
  visibleWhen?: Record<string, unknown>;
  /** Inline theory: what the parameter does mathematically. */
  theory?: string;
  /** Safe/reasonable operating range. */
  range?: string;
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
  /** Verbose architectural/mathematical theory for the help tooltip. */
  theory?: string;
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
  paused: boolean;
  nodes: NodeStateSnapshot[];
  edges: EdgeSpec[];
}

export interface NodeInstanceSpec {
  id: string;
  type: string;
  params?: Record<string, unknown>;
  position?: { x: number; y: number };
}

export interface ArchTemplate {
  id: string;
  name: string;
  description: string;
  designParams: number;
  spec: { name: string; nodes: NodeInstanceSpec[]; edges: EdgeSpec[] };
}

export interface MetricEvent {
  ts: number;
  nodeId: string;
  name: string;
  value: number;
  extra?: Record<string, unknown>;
}

export interface ModelConfig {
  vocabSize: number;
  dModel: number;
  contextLength: number;
  hiddenDim: number;
  mixer: string;
  loss: string;
  stochasticExplorationPool?: boolean;
  populationSize?: number;
  survivalCount?: number;
  survivalRate?: number;
  stochasticMutationSigma?: number;
}

export interface BenchmarkPoint {
  step: number;
  loss: number;
  ts: number;
}

export interface ModelVariant {
  id: string;
  name: string;
  source: "export" | "clone";
  parentId?: string;
  config: ModelConfig;
  paramCount: number;
  weightsPath: string;
  tokenizerPath?: string;
  createdAt: string;
  finalLoss?: number;
  datasetLoss?: number;
  history: BenchmarkPoint[];
  training: boolean;
}

export interface ModelAcceptanceProposal {
  id: string;
  projectId: string;
  modelName: string;
  oldTrainLoss: number | null;
  newTrainLoss: number | null;
  trainLossDelta: number | null;
  oldDatasetLoss: number | null;
  newDatasetLoss: number | null;
  datasetLossDelta: number | null;
  oldTechStack: string[];
  newTechStack: string[];
}

export interface ChatSessionMeta {
  id: string;
  title: string;
  variantId: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface ChatStoredMessage {
  id: number;
  sessionId: string;
  role: "user" | "assistant" | "system" | "tool";
  text: string;
  variantId: string | null;
  createdAt: number;
}

export interface ChatSession extends ChatSessionMeta {
  messages: ChatStoredMessage[];
}

export interface VariantMetric {
  variantId: string;
  step: number;
  loss: number;
  tokensPerSec: number;
}

export type ServerMessage =
  | { ev: "state"; state: PipelineStateSnapshot }
  | { ev: "runtime"; backend: string; deviceName?: string | null }
  | { ev: "projects"; projects: Array<{ id: string; name: string; updatedAt: number }> }
  | { ev: "catalog"; descriptors: NodeDescriptor[] }
  | { ev: "templates"; templates: ArchTemplate[] }
  | { ev: "metric"; metric: MetricEvent }
  | { ev: "log"; nodeId: string; message: string }
  | { ev: "error"; message: string }
  | { ev: "library"; variants: ModelVariant[] }
  | { ev: "model_acceptance_required"; proposal: ModelAcceptanceProposal }
  | { ev: "variant_metric"; metric: VariantMetric }
  | { ev: "chat_token"; chatId: string; token: number; text: string }
  | { ev: "chat_done"; chatId: string; reason: "complete" | "stopped" | "error"; message?: string }
  | { ev: "chats"; sessions: ChatSessionMeta[] }
  | { ev: "chat_session"; session: ChatSession }
  | { ev: "mcp_result"; payload: unknown }
  | { ev: "workers"; workers: Array<{ id: string; capabilities: Record<string, unknown>; connectedAt: string }> }
  | { ev: "worker_event"; workerId: string; payload: unknown };

export type ClientMessage =
  | { op: "get_state" }
  | { op: "get_catalog" }
  | { op: "load_pipeline"; spec: { name: string; nodes: unknown[]; edges: EdgeSpec[] } }
  | { op: "run"; projectId?: string }
  | { op: "stop" }
  | { op: "pause_training" }
  | { op: "resume_training" }
  | { op: "cancel_training" }
  | { op: "commit_training" }
  | { op: "accept_model_result"; proposalId: string }
  | { op: "reject_model_result"; proposalId: string }
  | { op: "update_learning_rate"; lr: number }
  | { op: "reset_model"; nodeId: string }
  | { op: "update_params"; nodeId: string; params: Record<string, unknown> }
  | { op: "move_node"; nodeId: string; position: { x: number; y: number } }
  | { op: "add_node"; node: NodeInstanceSpec }
  | { op: "remove_node"; nodeId: string }
  | { op: "add_edge"; edge: EdgeSpec }
  | { op: "remove_edge"; edge: EdgeSpec }
  | { op: "get_templates" }
  | { op: "apply_template"; templateId: string }
  | { op: "library_list" }
  | { op: "library_create"; name: string; overrides?: Record<string, unknown> }
  | { op: "library_delete"; variantId: string }
  | { op: "library_rename"; variantId: string; name: string }
  | { op: "library_clone"; sourceId: string; name: string; overrides?: Record<string, unknown> }
  | { op: "library_train"; variantId: string; steps?: number; batchSize?: number; lr?: number }
  | { op: "library_stop_train"; variantId: string }
  | { op: "chat_send"; chatId: string; sessionId: string; variantId: string; prompt: string; maxTokens?: number; temperature?: number; topP?: number }
  | { op: "chat_stop"; chatId: string }
  | { op: "list_chats" }
  | { op: "create_chat"; sessionId: string; title?: string; variantId?: string | null }
  | { op: "load_chat"; sessionId: string }
  | { op: "rename_chat"; sessionId: string; title: string }
  | { op: "delete_chat"; sessionId: string }
  | { op: "mcp"; payload: unknown }
  | { op: "save_graph"; projectId?: string; name?: string }
  | { op: "load_project"; projectId: string; name?: string }
  | { op: "list_projects" }
  | { op: "rename_project"; projectId: string; name: string }
  | { op: "delete_project"; projectId: string };
