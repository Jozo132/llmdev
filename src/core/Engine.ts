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

  stop(): void {
    this.abort?.abort();
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
        try {
          rt.outputs = await rt.node.run(inputs, {
            artifactsDir: this.artifactsDir,
            signal,
            metric: (name, value, extra) => {
              const ev: MetricEvent = { ts: Date.now(), nodeId: id, name, value, extra };
              this.emit("metric", ev);
            },
            log: (message) => this.emit("log", { nodeId: id, message }),
          });
          rt.status = signal.aborted ? "skipped" : "done";
        } catch (err) {
          rt.status = "error";
          rt.error = err instanceof Error ? err.message : String(err);
          this.emit("log", { nodeId: id, message: `ERROR: ${rt.error}` });
        }
        this.emitState();
      }
    } finally {
      this.abort = null;
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
      nodes,
      edges: this.edges,
    };
  }

  private emitState(): void {
    this.emit("state", this.snapshot());
  }
}
