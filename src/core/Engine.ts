/**
 * Engine — loads a PipelineSpec, topologically orders the node graph, and
 * executes it, routing outputs to downstream inputs along typed edges.
 *
 * Emits (EventEmitter):
 *  - "state"  (PipelineStateSnapshot) on any node status change
 *  - "metric" (MetricEvent)           on every ctx.metric() call
 *  - "log"    ({ nodeId, message })
 */
import { EventEmitter } from "node:events";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { createNode } from "./Registry.js";
import type {
  EdgeSpec,
  MetricEvent,
  NodeInstanceSpec,
  NodeStateSnapshot,
  NodeStatus,
  PipelineNode,
  PipelineSpec,
  PipelineStateSnapshot,
} from "./types.js";

interface RuntimeNode {
  spec: PipelineSpec["nodes"][number];
  node: PipelineNode;
  status: NodeStatus;
  error?: string;
  outputs?: Record<string, unknown>;
}

export class Engine extends EventEmitter {
  private nodes = new Map<string, RuntimeNode>();
  private edges: EdgeSpec[] = [];
  private spec: PipelineSpec | null = null;
  private abort: AbortController | null = null;
  private _paused = false;
  private pauseWaiters: Array<() => void> = [];
  /** "Commit Early" flag — scoped to the currently executing node. */
  private commitFlag = false;
  /** Live lr override — written by update_learning_rate, read hot per step. */
  private lrOverride: number | null = null;
  readonly artifactsDir: string;

  constructor(artifactsDir = process.env.LLMDEV_ARTIFACTS_DIR ?? "artifacts") {
    super();
    this.artifactsDir = artifactsDir;
    mkdirSync(path.join(this.artifactsDir, "tokens"), { recursive: true });
    mkdirSync(path.join(this.artifactsDir, "checkpoints"), { recursive: true });
    mkdirSync(path.join(this.artifactsDir, "exports"), { recursive: true });
  }

  get running(): boolean {
    return this.abort !== null;
  }

  get paused(): boolean {
    return this._paused;
  }

  /**
   * Gracefully suspend training: nodes block at the pause gate BETWEEN steps,
   * so weight matrices and optimizer moments (host + CUDA) stay exactly as
   * they are — no state is flushed or lost.
   */
  pause(): void {
    if (!this.running || this._paused) return;
    this._paused = true;
    this.emitState();
  }

  resume(): void {
    if (!this._paused) return;
    this._paused = false;
    for (const w of this.pauseWaiters.splice(0)) w();
    this.emitState();
  }

  /** Cancel: abort the run AND wake paused nodes so they observe the abort. */
  stop(): void {
    this.abort?.abort();
    this.resume();
  }

  /**
   * "Commit & Proceed": interrupt the remaining iterations of the running
   * training node, freeze weights + Adam moments exactly where they are, and
   * let the node finish as 'done' so downstream nodes fire automatically.
   */
  commitEarly(): void {
    if (!this.running) return;
    this.commitFlag = true;
    this.resume(); // wake a paused loop so it can observe the commit
  }

  /**
   * Real-time lr hot-tuning: update the active scalar consumed by the running
   * training loop's NEXT Adam step. The loop never pauses and step metrics
   * are untouched — the trainer simply reads the new value on its next
   * iteration (between CUDA kernel launches, so device state stays coherent).
   */
  setLearningRate(lr: number): void {
    if (!Number.isFinite(lr) || lr <= 0) throw new Error(`Invalid learning rate: ${lr}`);
    this.lrOverride = lr;
    this.emit("log", { nodeId: "", message: `learning rate hot-tuned → ${lr.toExponential(2)}` });
  }

  private waitIfPaused = async (): Promise<void> => {
    while (this._paused && !this.abort?.signal.aborted) {
      await new Promise<void>((r) => this.pauseWaiters.push(r));
    }
  };

  load(spec: PipelineSpec): void {
    if (this.running) throw new Error("Cannot load while running");
    this.nodes.clear();
    for (const n of spec.nodes) {
      this.nodes.set(n.id, {
        spec: n,
        node: createNode(n.type, n.params),
        status: "idle",
      });
    }
    this.edges = spec.edges;
    this.spec = spec;
    this.validate();
    this.emitState();
  }

  updateNodeParams(nodeId: string, params: Record<string, unknown>): void {
    const rt = this.nodes.get(nodeId);
    if (!rt) throw new Error(`No such node: ${nodeId}`);
    rt.node.params = { ...rt.node.params, ...params };
    rt.spec.params = { ...rt.spec.params, ...params };
    this.emitState();
  }

  updateNodePosition(nodeId: string, position: { x: number; y: number }): void {
    const rt = this.nodes.get(nodeId);
    if (!rt) throw new Error(`No such node: ${nodeId}`);
    rt.spec.position = position;
    this.emitState();
  }

  // ── Interactive graph editing (canvas ↔ engine) ────────────────────────

  addNode(spec: NodeInstanceSpec): void {
    this.assertEditable();
    if (this.nodes.has(spec.id)) throw new Error(`Duplicate node id: ${spec.id}`);
    this.nodes.set(spec.id, { spec, node: createNode(spec.type, spec.params), status: "idle" });
    this.spec?.nodes.push(spec);
    this.emitState();
  }

  removeNode(nodeId: string): void {
    this.assertEditable();
    if (!this.nodes.delete(nodeId)) throw new Error(`No such node: ${nodeId}`);
    this.edges = this.edges.filter((e) => e.from.node !== nodeId && e.to.node !== nodeId);
    if (this.spec) {
      this.spec.nodes = this.spec.nodes.filter((n) => n.id !== nodeId);
      this.spec.edges = this.edges;
    }
    this.emitState();
  }

  /** Connect ports with strict dataType validation + cycle rejection. */
  addEdge(edge: EdgeSpec): void {
    this.assertEditable();
    const dup = this.edges.some(
      (e) => e.to.node === edge.to.node && e.to.port === edge.to.port
    );
    if (dup) throw new Error(`Input ${edge.to.node}.${edge.to.port} is already connected`);
    this.edges.push(edge);
    if (this.spec) this.spec.edges = this.edges;
    try {
      this.validate(); // type-checks ports + rejects cycles
    } catch (err) {
      this.edges.pop(); // roll back the invalid edge
      if (this.spec) this.spec.edges = this.edges;
      throw err;
    }
    this.emitState();
  }

  removeEdge(edge: EdgeSpec): void {
    this.assertEditable();
    this.edges = this.edges.filter(
      (e) =>
        !(e.from.node === edge.from.node && e.from.port === edge.from.port &&
          e.to.node === edge.to.node && e.to.port === edge.to.port)
    );
    if (this.spec) this.spec.edges = this.edges;
    this.emitState();
  }

  private assertEditable(): void {
    if (this.running) throw new Error("Stop the pipeline before editing the graph");
    if (!this.spec) this.spec = { name: "untitled", nodes: [], edges: [] };
  }

  /** Kahn topological sort — rejects cycles, dangling edges. */
  private topoOrder(): string[] {
    const indegree = new Map<string, number>();
    for (const id of this.nodes.keys()) indegree.set(id, 0);
    for (const e of this.edges) {
      indegree.set(e.to.node, (indegree.get(e.to.node) ?? 0) + 1);
    }
    const queue = [...indegree.entries()].filter(([, d]) => d === 0).map(([id]) => id);
    const order: string[] = [];
    while (queue.length) {
      const id = queue.shift()!;
      order.push(id);
      for (const e of this.edges) {
        if (e.from.node !== id) continue;
        const d = indegree.get(e.to.node)! - 1;
        indegree.set(e.to.node, d);
        if (d === 0) queue.push(e.to.node);
      }
    }
    if (order.length !== this.nodes.size) {
      throw new Error("Pipeline graph contains a cycle");
    }
    return order;
  }

  private validate(): void {
    for (const e of this.edges) {
      const from = this.nodes.get(e.from.node);
      const to = this.nodes.get(e.to.node);
      if (!from) throw new Error(`Edge references unknown node "${e.from.node}"`);
      if (!to) throw new Error(`Edge references unknown node "${e.to.node}"`);
      const outPort = from.node.descriptor.outputs.find((p) => p.name === e.from.port);
      const inPort = to.node.descriptor.inputs.find((p) => p.name === e.to.port);
      if (!outPort) throw new Error(`"${e.from.node}" has no output port "${e.from.port}"`);
      if (!inPort) throw new Error(`"${e.to.node}" has no input port "${e.to.port}"`);
      if (outPort.dataType !== inPort.dataType) {
        throw new Error(
          `Type mismatch ${e.from.node}.${e.from.port} (${outPort.dataType}) → ` +
            `${e.to.node}.${e.to.port} (${inPort.dataType})`
        );
      }
    }
    this.topoOrder();
  }

  async run(): Promise<void> {
    if (!this.spec) throw new Error("No pipeline loaded");
    if (this.running) throw new Error("Already running");
    this.abort = new AbortController();
    const { signal } = this.abort;

    for (const rt of this.nodes.values()) {
      rt.status = "idle";
      rt.error = undefined;
      rt.outputs = undefined;
    }
    this.emitState();

    try {
      for (const id of this.topoOrder()) {
        if (signal.aborted) break;
        const rt = this.nodes.get(id)!;

        // Gather inputs from upstream outputs.
        const inputs: Record<string, unknown> = {};
        let upstreamFailed = false;
        for (const e of this.edges) {
          if (e.to.node !== id) continue;
          const up = this.nodes.get(e.from.node)!;
          if (up.status !== "done" || !up.outputs) {
            upstreamFailed = true;
            break;
          }
          inputs[e.to.port] = up.outputs[e.from.port];
        }
        if (upstreamFailed) {
          rt.status = "skipped";
          this.emitState();
          continue;
        }

        rt.status = "running";
        this.emitState();
        // ── High-resolution profiling + windowed velocity/ETA tracking ─────
        const startTime = performance.now();
        let lastTickTime = startTime;
        // Rolling window of (time, progress) samples for velocity estimation.
        const window: Array<{ t: number; p: number }> = [];
        const WINDOW_MS = 15_000;
        const WINDOW_MAX = 32;
        const emitMetric = (name: string, value: number, extra?: Record<string, unknown>): void => {
          const ev: MetricEvent = { ts: Date.now(), nodeId: id, name, value, extra };
          this.emit("metric", ev);
          if (name !== "node_progress") return;
          // Derive elapsedTimeMs + windowed rolling-average etaMs.
          const now = performance.now();
          lastTickTime = now;
          window.push({ t: now, p: value });
          while (window.length > WINDOW_MAX || (window.length > 2 && now - window[0].t > WINDOW_MS)) {
            window.shift();
          }
          const elapsed = now - startTime;
          this.emit("metric", { ts: Date.now(), nodeId: id, name: "elapsed_ms", value: elapsed } as MetricEvent);
          const first = window[0];
          const dp = value - first.p;
          const dt = now - first.t;
          if (dp > 1e-6 && dt > 0 && value < 1) {
            const velocity = dp / dt; // progress per ms over the window
            this.emit("metric", {
              ts: Date.now(), nodeId: id, name: "eta_ms",
              value: (1 - value) / velocity,
              extra: { velocity, elapsedMs: elapsed },
            } as MetricEvent);
          }
        };
        emitMetric("node_progress", 0);
        // Keep the elapsed clock ticking even between progress updates.
        const clock = setInterval(() => {
          this.emit("metric", {
            ts: Date.now(), nodeId: id, name: "elapsed_ms",
            value: performance.now() - startTime,
            extra: { sinceTickMs: performance.now() - lastTickTime },
          } as MetricEvent);
        }, 1000);
        this.commitFlag = false; // commit control is scoped per node
        this.lrOverride = null;  // lr override is scoped per node run
        try {
          rt.outputs = await rt.node.run(inputs, {
            nodeId: id,
            artifactsDir: this.artifactsDir,
            signal,
            shouldCommit: () => this.commitFlag,
            waitIfPaused: this.waitIfPaused,
            getLrOverride: () => this.lrOverride,
            metric: emitMetric,
            log: (message) => this.emit("log", { nodeId: id, message }),
          });
          rt.status = signal.aborted ? "skipped" : "done";
          if (rt.status === "done") emitMetric("node_progress", 1);
        } catch (err) {
          rt.status = "error";
          rt.error = err instanceof Error ? err.message : String(err);
          this.emit("log", { nodeId: id, message: `ERROR: ${rt.error}` });
        }
        clearInterval(clock);
        this.commitFlag = false;
        emitMetric("execution_time_ms", performance.now() - startTime);
        this.emitState();
      }
    } finally {
      this.abort = null;
      this._paused = false;
      for (const w of this.pauseWaiters.splice(0)) w();
      this.emitState();
    }
  }

  snapshot(): PipelineStateSnapshot {
    const nodes: NodeStateSnapshot[] = [...this.nodes.values()].map((rt) => ({
      id: rt.spec.id,
      type: rt.spec.type,
      label: rt.node.descriptor.label,
      status: rt.status,
      params: rt.node.params,
      position: rt.spec.position,
      error: rt.error,
    }));
    return {
      name: this.spec?.name ?? "(none)",
      running: this.running,
      paused: this._paused,
      nodes,
      edges: this.edges,
    };
  }

  private emitState(): void {
    this.emit("state", this.snapshot());
  }

  outputsByNode(): Record<string, Record<string, unknown>> {
    const out: Record<string, Record<string, unknown>> = {};
    for (const [id, rt] of this.nodes) {
      if (rt.outputs) out[id] = rt.outputs;
    }
    return out;
  }
}
