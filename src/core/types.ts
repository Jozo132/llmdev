/**
 * Core typings for the plug-and-play node-graph pipeline.
 *
 * Design goals:
 *  - Nodes are pure units of work with typed input/output ports.
 *  - Wiring is data (a PipelineSpec JSON), so the graph can be built headlessly
 *    via CLI or interactively from the Vue canvas over WebSockets.
 *  - Any layer of the ML stack (attention, loss, optimizer) is injectable so
 *    custom PyTorch/C++/CUDA implementations can be swapped in later.
 */

/** Well-known payload types that flow along edges. Extensible by string. */
export type PortDataType =
  | "text-stream"      // AsyncIterable<string> — raw documents
  | "token-file"       // TokenFileRef — path to a packed uint16 .bin
  | "tokenizer"        // TokenizerHandle
  | "model-config"     // ModelConfig
  | "model"            // TrainedModelHandle
  | "metrics"          // EvalReport
  | "artifact"         // ExportedArtifact
  | (string & {});

export interface PortSpec {
  name: string;
  dataType: PortDataType;
  required?: boolean;
}

/** JSON-serializable parameter bag shown/edited in the UI property panel. */
export type NodeParams = Record<string, unknown>;

export interface ParamSchemaEntry {
  key: string;
  label: string;
  type: "string" | "number" | "boolean" | "select";
  options?: string[];       // for "select"
  default?: unknown;
  description?: string;
  /** Show only when the selected node params match these literal/regex rules. */
  visibleWhen?: Record<string, unknown>;
  /** Inline theory: what the parameter does mathematically (gradients, VRAM…). */
  theory?: string;
  /** Strict safe/reasonable operating range shown in the UI. */
  range?: string;
}

/** Static, serializable description of a node class (used by UI + registry). */
export interface NodeDescriptor {
  /** Registry key, e.g. "data.jsIngestion" */
  type: string;
  label: string;
  category: "data" | "tokenizer" | "model" | "train" | "eval" | "export" | "custom";
  inputs: PortSpec[];
  outputs: PortSpec[];
  paramSchema: ParamSchemaEntry[];
  /** Verbose architectural/mathematical theory for the help tooltip. */
  theory?: string;
}

export type NodeStatus = "idle" | "running" | "done" | "error" | "skipped";

export interface NodeRunContext {
  /** Runtime node id, stable across graph saves and reruns. */
  nodeId: string;
  /** Resolved artifacts directory (respects LLMDEV_ARTIFACTS_DIR). */
  artifactsDir: string;
  /** Emit a metric sample (streamed live to the web UI). */
  metric(name: string, value: number, extra?: Record<string, unknown>): void;
  /** Structured log line, mirrored to CLI stdout and WebSocket clients. */
  log(message: string): void;
  /** Cooperative cancellation — long loops must poll this. */
  signal: AbortSignal;
  /**
   * "Commit Early / Proceed" seam — when true, training loops must stop
   * iterating IMMEDIATELY after the current step, keep weights + optimizer
   * moments exactly as they are, and return normally so the node completes
   * as 'done' and downstream nodes fire.
   */
  shouldCommit(): boolean;
  /**
   * Cooperative pause gate — resolves immediately when not paused, otherwise
   * blocks until resume/cancel. Weight & optimizer state stay untouched in
   * host/device memory while suspended.
   */
  waitIfPaused(): Promise<void>;
  /**
   * Live learning-rate override (update_learning_rate op). Trainers read
   * this at the top of EVERY step so the user can hot-tune Adam's lr while
   * the loop is executing — no pause, no halt, no metric reset. null ⇒ use
   * the node's configured lr.
   */
  getLrOverride(): number | null;
}

/**
 * A pipeline node. Implementations are completely modular: the engine only
 * knows about this interface, so custom nodes (alternative attention layers,
 * C++-backed trainers, PyTorch bridges) plug in without engine changes.
 */
export interface PipelineNode {
  readonly descriptor: NodeDescriptor;
  params: NodeParams;
  /**
   * Execute with named inputs; return named outputs.
   * Payloads are opaque to the engine — only edges give them meaning.
   */
  run(
    inputs: Record<string, unknown>,
    ctx: NodeRunContext
  ): Promise<Record<string, unknown>>;
}

/** Factory signature registered per node type. */
export type NodeFactory = (params?: NodeParams) => PipelineNode;

// ── Graph wiring (pure data — this is what the UI drags around) ──────────────

export interface NodeInstanceSpec {
  id: string;              // unique within the pipeline
  type: string;            // registry key
  params?: NodeParams;
  /** UI-only hint; engine ignores it. */
  position?: { x: number; y: number };
}

export interface EdgeSpec {
  from: { node: string; port: string };
  to: { node: string; port: string };
}

export interface PipelineSpec {
  name: string;
  nodes: NodeInstanceSpec[];
  edges: EdgeSpec[];
}

// ── Payload shapes passed along edges ────────────────────────────────────────

export interface TokenFileRef {
  path: string;
  /** Token count (uint16 each ⇒ bytes = tokens * 2). */
  tokens: number;
  vocabSize: number;
}

export interface TokenizerHandle {
  vocabSize: number;
  encode(text: string): Uint16Array;
  decode(tokens: ArrayLike<number>): string;
  /** Serializable state (merges/vocab) for export. */
  toJSON(): unknown;
}

export interface ModelConfig {
  vocabSize: number;
  dModel: number;
  contextLength: number;
  hiddenDim: number;
  /** Injection point: name of a registered SequenceMixer (attention) impl. */
  mixer: string;
  /** Injection point: name of a registered loss function. */
  loss: string;
  // ── Architectural design fields (drive the parameter calculator & future
  //    projection-based blocks; the single-block PoC trainer records them) ──
  nLayers?: number;
  nHeads?: number;
  /** KV heads for Grouped-Query Attention (kvHeads < nHeads ⇒ GQA). */
  kvHeads?: number;
  mlp?: "standard" | "swiglu";
  // ── Fine-tuning mode ──
  /** "full" (default) trains every parameter; "lora" freezes the base and
   *  trains low-rank adapters A[d×r]/B[r×d] on the attention q/v projections. */
  fineTuneMode?: "full" | "lora";
  /** LoRA rank r (adapter bottleneck width). */
  loraRank?: number;
  /** LoRA α — adapter contribution is scaled by α/r. */
  loraAlpha?: number;
  /** Enable stochastic ES mutations alongside the gradient path. */
  stochasticExplorationPool?: boolean;
  /** Parallel population size for LoRA; forced to 1 in full-parameter mode. */
  populationSize?: number;
  /** Top-N survivor count used by the ES reducer. */
  survivalCount?: number;
  /** Survivor percentage used to derive Top-N for LoRA ES. */
  survivalRate?: number;
  /** Mutation volatility σ for pseudo-Gaussian perturbations. */
  stochasticMutationSigma?: number;
}

export interface TrainedModelHandle {
  config: ModelConfig;
  paramCount: number;
  /** Flat parameter buffer — the "1M parameter weight array". */
  weights: Float32Array;
  /** LoRA adapter buffer (A/B matrices) when fineTuneMode === "lora". */
  lora?: Float32Array;
  finalLoss: number;
  stepsCompleted: number;
}

export interface EvalReport {
  loss: number;
  perplexity: number;
  sample: string;
}

export interface ExportedArtifact {
  path: string;
  bytes: number;
  format: string;
}

// ── Live state serialization (Engine → WebSocket → Vue canvas) ───────────────

export interface NodeStateSnapshot {
  id: string;
  type: string;
  label: string;
  status: NodeStatus;
  params: NodeParams;
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

export interface MetricEvent {
  ts: number;
  nodeId: string;
  name: string;          // "loss" | "tokens_per_sec" | "vram_mb" | ...
  value: number;
  extra?: Record<string, unknown>;
}
