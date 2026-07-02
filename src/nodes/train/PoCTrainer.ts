/**
 * PoCTrainer — TrainingEngineNode implementation.
 *
 * Minimal but REAL training loop over the packed uint16 token shard:
 *   • memory-maps the shard as a Uint16Array (no parsing, no copies)
 *   • samples random context windows into batches
 *   • forward pass → softmax cross-entropy → exact backward pass
 *   • Adam update over the flat ~1M-parameter Float32Array
 * Streams loss / tokens-per-sec / VRAM metrics through the engine bus so the
 * Vue canvas and CLI see identical telemetry. Pure JS today; the TinyLM mixer
 * and loss slots are the seams for CUDA/PyTorch replacements tomorrow.
 */
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { TinyLM } from "../../ml/model.js";
import type {
  ModelConfig, NodeDescriptor, NodeParams, NodeRunContext, PipelineNode,
  TokenFileRef, TrainedModelHandle,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "train.poc",
  label: "PoC Trainer (1M params)",
  category: "train",
  inputs: [
    { name: "config", dataType: "model-config", required: true },
    { name: "tokens", dataType: "token-file", required: true },
  ],
  outputs: [{ name: "model", dataType: "model" }],
  paramSchema: [
    { key: "steps", label: "Training steps", type: "number", default: 30 },
    { key: "batchSize", label: "Batch size", type: "number", default: 4 },
    { key: "lr", label: "Learning rate", type: "number", default: 0.003 },
    { key: "logEvery", label: "Log every N steps", type: "number", default: 1 },
  ],
};

/** Best-effort VRAM sample via nvidia-smi; resolves 0 when no GPU visible. */
function sampleVramMb(): Promise<number> {
  return new Promise((resolve) => {
    execFile(
      "nvidia-smi",
      ["--query-gpu=memory.used", "--format=csv,noheader,nounits"],
      { timeout: 1500 },
      (err, stdout) => resolve(err ? 0 : parseInt(stdout.trim().split("\n")[0], 10) || 0)
    );
  });
}

export class PoCTrainer implements PipelineNode {
  readonly descriptor = DESCRIPTOR;
  params: NodeParams;

  constructor(params: NodeParams = {}) {
    this.params = {
      ...Object.fromEntries(DESCRIPTOR.paramSchema.map((p) => [p.key, p.default])),
      ...params,
    };
  }

  async run(inputs: Record<string, unknown>, ctx: NodeRunContext) {
    const config = inputs.config as ModelConfig;
    const tokenRef = inputs.tokens as TokenFileRef;
    if (!config || !tokenRef) throw new Error("PoCTrainer requires 'config' and 'tokens'");
    const p = this.params as { steps: number; batchSize: number; lr: number; logEvery: number };

    // Load the shard as raw uint16 — a 2M-token shard is only 4MB of RAM.
    const raw = await readFile(tokenRef.path);
    const data = new Uint16Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 2));
    const windowLen = config.contextLength; // ctx-1 inputs + 1 target
    if (data.length < windowLen + 1) {
      throw new Error(`Shard too small: ${data.length} tokens < window ${windowLen + 1}`);
    }

    const model = new TinyLM(config);
    ctx.log(`TinyLM instantiated: ${model.paramCount.toLocaleString()} parameters ` +
            `(${(model.paramCount * 4 / 1024 / 1024).toFixed(1)}MB fp32)`);
    ctx.metric("param_count", model.paramCount);

    const sampleWindow = (): Uint16Array => {
      const start = Math.floor(Math.random() * (data.length - windowLen));
      return data.subarray(start, start + windowLen);
    };

    let finalLoss = NaN;
    for (let step = 1; step <= p.steps; step++) {
      if (ctx.signal.aborted) {
        ctx.log(`Aborted at step ${step}`);
        break;
      }
      const batch = Array.from({ length: p.batchSize }, sampleWindow);
      const t0 = performance.now();
      const { loss, tokensProcessed } = model.step(batch, p.lr);
      const dt = (performance.now() - t0) / 1000;
      finalLoss = loss;

      if (step % p.logEvery === 0 || step === p.steps) {
        const tps = tokensProcessed / dt;
        const vram = await sampleVramMb();
        const rssMb = process.memoryUsage().rss / 1024 / 1024;
        ctx.metric("loss", loss, { step });
        ctx.metric("tokens_per_sec", tps, { step });
        ctx.metric("vram_mb", vram, { step });
        ctx.metric("rss_mb", rssMb, { step });
        ctx.log(
          `step ${step}/${p.steps}  loss=${loss.toFixed(4)}  ` +
          `${tps.toFixed(0)} tok/s  vram=${vram}MB  rss=${rssMb.toFixed(0)}MB`
        );
      }
      // Yield to the event loop so WebSocket frames flush between steps.
      await new Promise((r) => setImmediate(r));
    }

    const handle: TrainedModelHandle = {
      config,
      paramCount: model.paramCount,
      weights: model.params,
      finalLoss,
      stepsCompleted: p.steps,
    };
    ctx.log(`Training done — final loss ${finalLoss.toFixed(4)}`);
    return { model: handle };
  }
}

export const pocTrainerDescriptor = DESCRIPTOR;
