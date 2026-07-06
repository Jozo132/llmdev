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
import { existsSync, mkdirSync, renameSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { TinyLM } from "../../ml/model.js";
import { gpuReduceFittest, gpuStochasticMutate } from "../../ml/backend.js";
import type {
  ModelConfig, NodeDescriptor, NodeParams, NodeRunContext, PipelineNode,
  TokenFileRef, TrainedModelHandle,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "train.poc",
  label: "Trainer",
  category: "train",
  theory:
    "Minimal-but-real training loop: sample random context windows, forward " +
    "pass, softmax cross-entropy, exact hand-derived backward, Adam update. " +
    "Adam keeps two moments per parameter (m̂ momentum, v̂ curvature) ⇒ " +
    "optimizer state = 2× model size in RAM/VRAM — why big-model training " +
    "costs ≈3× weights. Pause blocks between steps preserving both moments; " +
    "cancel frees the CUDA context immediately.",
  inputs: [
    { name: "config", dataType: "model-config", required: true },
    { name: "tokens", dataType: "token-file", required: true },
  ],
  outputs: [{ name: "model", dataType: "model" }],
  paramSchema: [
    { key: "steps", label: "Training steps", type: "number", default: 30,
      theory: "Each step = one gradient update on batchSize windows. Loss " +
        "should fall from ln(V) (uniform prediction) within tens of steps.",
      range: "10–100 quick runs · thousands for real training" },
    { key: "batchSize", label: "Batch size", type: "number", default: 4,
      theory: "Windows averaged per update. Larger batch ⇒ lower gradient " +
        "variance ⇒ supports higher lr (linear-scaling heuristic), but memory " +
        "and step time grow linearly.",
      range: "2–32 on this hardware" },
    { key: "lr", label: "Learning rate", type: "number", default: 0.003,
      theory: "Adam step scale: Δw = lr·m̂/(√v̂+ε). Too high ⇒ loss spikes/NaN " +
        "(update overshoots curvature); too low ⇒ wasted compute. Rule of " +
        "thumb: peak lr ≈ 3e-4 to 3e-3 for Adam at small scale, decayed over " +
        "training.",
      range: "1e-4 – 5e-3 (Adam)" },
    { key: "logEvery", label: "Log every N steps", type: "number", default: 1 },
    { key: "checkpoint", label: "Checkpoint name", type: "string", default: "",
      description: "Warm-start resume slot under <artifacts>/checkpoints/",
      theory: "When set, the trainer looks for <name>.weights.bin AND " +
        "<name>.adam.bin. If both exist with matching shapes, random init is " +
        "skipped entirely: weights stream straight into the hot GPU layer " +
        "buffers (llm_ctx_sync_layer) and Adam resumes with its exact first/" +
        "second moments + step counter — the bias-correction schedule " +
        "continues from where it left off. Both files are re-written " +
        "atomically after the run.",
      range: "empty = fresh init every run" },
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

const MAX_TRAIN_TELEMETRY_SAMPLES = 4_096;

function clampInt(value: unknown, fallback: number, lo: number, hi: number): number {
  const parsed = Math.round(Number(value));
  return Math.min(hi, Math.max(lo, Number.isFinite(parsed) ? parsed : fallback));
}

function pcgHash(input: number): number {
  const state = (Math.imul(input >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

function unitRandom(seed: number): number {
  return (((pcgHash(seed) >>> 8) + 1) * 2 ** -24);
}

function pseudoGaussian(seed: number): number {
  const u1 = Math.max(unitRandom(seed), 1e-7);
  const u2 = unitRandom(seed ^ 0x9e3779b9);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function stochasticMutate(
  base: Float32Array, population: Float32Array,
  paramCount: number, populationSize: number, step: number, sigma: number
): boolean {
  if (gpuStochasticMutate(base, population, paramCount, populationSize, step, sigma)) return true;
  for (let candidate = 0; candidate < populationSize; candidate++) {
    const candidateOffset = candidate * paramCount;
    for (let weightIdx = 0; weightIdx < paramCount; weightIdx++) {
      const seed = pcgHash((weightIdx + step * 1337 + candidate * 0x85ebca6b) >>> 0);
      population[candidateOffset + weightIdx] = base[weightIdx] + sigma * pseudoGaussian(seed);
    }
  }
  return false;
}

function reduceFittestCpu(
  base: Float32Array, population: Float32Array, losses: Float32Array,
  paramCount: number, populationSize: number, survivorCount: number,
  step: number, blend: number, sigma: number
): { bestLoss: number; variance: number; bestIndex: number } {
  const order = Array.from({ length: populationSize }, (_, candidate) => candidate)
    .sort((left, right) => losses[left] - losses[right]);
  const survivors = new Float32Array(survivorCount * paramCount);
  for (let survivor = 0; survivor < survivorCount; survivor++) {
    const sourceOffset = order[survivor] * paramCount;
    survivors.set(population.subarray(sourceOffset, sourceOffset + paramCount), survivor * paramCount);
  }
  for (let candidate = 0; candidate < populationSize; candidate++) {
    const survivor = candidate % survivorCount;
    const targetOffset = candidate * paramCount;
    const survivorOffset = survivor * paramCount;
    for (let weightIdx = 0; weightIdx < paramCount; weightIdx++) {
      let value = survivors[survivorOffset + weightIdx];
      if (candidate >= survivorCount) {
        const seed = pcgHash((weightIdx + step * 7331 + candidate * 0xc2b2ae35) >>> 0);
        value += sigma * 0.35 * pseudoGaussian(seed);
      }
      population[targetOffset + weightIdx] = value;
    }
  }
  for (let weightIdx = 0; weightIdx < paramCount; weightIdx++) {
    base[weightIdx] = base[weightIdx] * (1 - blend) + survivors[weightIdx] * blend;
  }
  const mean = losses.reduce((sum, value) => sum + value, 0) / populationSize;
  const variance = losses.reduce((sum, value) => sum + (value - mean) ** 2, 0) / populationSize;
  return { bestLoss: losses[order[0]], variance, bestIndex: order[0] };
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
    if (!config || !tokenRef) throw new Error("Trainer requires 'config' and 'tokens'");
    const p = this.params as { steps: number; batchSize: number; lr: number; logEvery: number; checkpoint?: string };
    const requestedLogEvery = Math.max(1, Math.floor(Number(p.logEvery) || 1));
    const logEvery = Math.max(requestedLogEvery, Math.ceil((Number(p.steps) || 1) / MAX_TRAIN_TELEMETRY_SAMPLES));
    const shouldEmitStep = (step: number): boolean => step % logEvery === 0 || step === p.steps;

    // Load the shard as raw uint16 — a 2M-token shard is only 4MB of RAM.
    const raw = await readFile(tokenRef.path);
    const data = new Uint16Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 2));
    const windowLen = config.contextLength; // ctx-1 inputs + 1 target
    if (data.length < windowLen + 1) {
      throw new Error(`Shard too small: ${data.length} tokens < window ${windowLen + 1}`);
    }

    // ── Warm-start: resume weights + Adam moments from a checkpoint slot ──
    const explicitCheckpoint = String(p.checkpoint ?? "").trim();
    const ckptName = explicitCheckpoint || `node-${ctx.nodeId}`;
    const ckptDir = path.join(ctx.artifactsDir, "checkpoints");
    const ckptWeights = ckptName ? path.join(ckptDir, `${ckptName}.weights.bin`) : "";
    const ckptAdam = ckptName ? path.join(ckptDir, `${ckptName}.adam.bin`) : "";
    let resumeW: Float32Array | null = null;
    let resumeS: Float32Array | null = null;
    if (ckptName && existsSync(ckptWeights) && existsSync(ckptAdam)) {
      const [wBuf, sBuf] = await Promise.all([readFile(ckptWeights), readFile(ckptAdam)]);
      const w = new Float32Array(wBuf.buffer, wBuf.byteOffset, Math.floor(wBuf.byteLength / 4));
      const s = new Float32Array(sBuf.buffer, sBuf.byteOffset, Math.floor(sBuf.byteLength / 4));
      if (w.length === TinyLM.paramCountFor(config) &&
          s.length === TinyLM.trainerStateLengthFor(config)) {
        resumeW = w;
        resumeS = s;
      } else {
        ctx.log(`Checkpoint "${ckptName}" shape mismatch — fresh init instead`);
      }
    }

    // Warm start skips random init entirely; loadWeights streams the host
    // arrays into the hot GPU contexts (syncE + llm_ctx_sync_layer per layer).
    const model = new TinyLM(config, undefined, { skipInit: !!resumeW });
    if (resumeW && resumeS) {
      model.loadWeights(resumeW);
      model.restoreTrainerState(resumeS);
      ctx.log(`Warm-start: resumed "${ckptName}" at Adam step ${model.adamStep} — ` +
              `weights + both optimizer moments restored, device buffers hot`);
    }
    ctx.log(`TinyLM instantiated: ${model.paramCount.toLocaleString()} parameters ` +
            `across ${model.nLayers} transformer layer(s) ` +
            `(${(model.paramCount * 4 / 1024 / 1024).toFixed(1)}MB fp32) — ` +
            `forward runs layers 0→${model.nLayers - 1}, backward ${model.nLayers - 1}→0`);
    if (logEvery > requestedLogEvery) {
      ctx.log(`Telemetry sampling auto-raised from every ${requestedLogEvery} step(s) to every ${logEvery} step(s) ` +
              `to cap live chart traffic at about ${MAX_TRAIN_TELEMETRY_SAMPLES.toLocaleString()} samples`);
    }
    if (model.isLora) {
      ctx.log(`LoRA fine-tuning active: r=${config.loraRank ?? 8}, α=${config.loraAlpha ?? 16} ` +
              `(scale ${model.loraScale.toFixed(3)}) — base weights FROZEN, ` +
              `${model.loraParams.length.toLocaleString()} trainable adapter params ` +
              `(${(100 * model.loraParams.length / model.paramCount).toFixed(2)}% of full), ` +
              `zero per-step host→device weight sync`);
      ctx.metric("lora_param_count", model.loraParams.length);
    }
    ctx.metric("param_count", model.paramCount);
    ctx.metric("layer_count", model.nLayers);

    const sampleWindow = (): Uint16Array => {
      const start = Math.floor(Math.random() * (data.length - windowLen));
      return data.subarray(start, start + windowLen);
    };

    const esEnabled = !!config.stochasticExplorationPool;
    const esSigma = Math.max(0, Number(config.stochasticMutationSigma ?? 0.002));
    const esPopulationSize = esEnabled
      ? (model.isLora ? clampInt(config.populationSize, 4, 1, 64) : 1)
      : 0;
    const esSurvivalCount = esEnabled
      ? clampInt(config.survivalCount, 1, 1, Math.max(1, esPopulationSize))
      : 0;
    const esValidationBatchSize = Math.max(1, Math.min(2, p.batchSize));
    const esBlend = 0.15;
    let loraPopulation: Float32Array | null = null;
    let fullMutationCandidate: Float32Array | null = null;
    let w_checkpoint_snapshot: Float32Array | null = null;
    if (esEnabled && model.isLora) {
      loraPopulation = new Float32Array(esPopulationSize * model.loraParams.length);
      ctx.log(`Stochastic ES enabled: LoRA parallel pool P=${esPopulationSize}, N=${esSurvivalCount}, sigma=${esSigma}`);
      ctx.metric("es_exploration_status", 1, { step: 0, mode: "lora", populationSize: esPopulationSize, survivorCount: esSurvivalCount });
    } else if (esEnabled) {
      fullMutationCandidate = new Float32Array(model.paramCount);
      w_checkpoint_snapshot = new Float32Array(model.paramCount);
      ctx.log(`Stochastic ES enabled: full-parameter temporal mode forced to P=1, sigma=${esSigma}`);
      ctx.metric("es_exploration_status", 1, { step: 0, mode: "full", populationSize: 1, survivorCount: 1 });
    }

    let finalLoss = NaN;
    let stepsCompleted = 0;
    let committedEarly = false;

    // ── Best-loss checkpoint tracking ──
    // Every step's loss is compared against the running record; improvements
    // snapshot weights + full Adam state (host copies) and persist them to
    // the distinct best.* pair so the FINAL artifacts ship the optimum, not
    // whatever the last trailing step happened to produce.
    let lowestRecordLoss = Infinity;
    let bestWeights: Float32Array | null = null;
    let bestState: Float32Array | null = null;
    let bestStep = 0;
    const ckptBestWeights = ckptName ? path.join(ckptDir, `${ckptName}.best.weights.bin`) : "";
    const ckptBestAdam = ckptName ? path.join(ckptDir, `${ckptName}.best.adam.bin`) : "";
    const persistBest = async (): Promise<void> => {
      if (!ckptName || !bestWeights || !bestState) return;
      mkdirSync(ckptDir, { recursive: true });
      const tmpW = `${ckptBestWeights}.tmp-${process.pid}`;
      const tmpA = `${ckptBestAdam}.tmp-${process.pid}`;
      await writeFile(tmpW, Buffer.from(bestWeights.buffer, bestWeights.byteOffset, bestWeights.byteLength));
      await writeFile(tmpA, Buffer.from(bestState.buffer, bestState.byteOffset, bestState.byteLength));
      renameSync(tmpW, ckptBestWeights);
      renameSync(tmpA, ckptBestAdam);
    };

    let activeLr = p.lr;
    try {
      for (let step = 1; step <= p.steps; step++) {
        // Pause gate: blocks between steps — weights + Adam moments stay
        // resident in host/device memory until resume or cancel.
        await ctx.waitIfPaused();
        if (ctx.signal.aborted) {
          ctx.log(`Cancelled at step ${step} — releasing GPU context`);
          break;
        }
        // "Commit Early / Proceed": checked between steps (i.e. between CUDA
        // kernel launches — the device queue is empty here), so weights and
        // both Adam moments freeze exactly as-is and the node returns 'done'.
        if (ctx.shouldCommit()) {
          committedEarly = true;
          ctx.log(`Commit & Proceed at step ${step - 1}/${p.steps} — weights + Adam moments frozen, skipping remaining iterations`);
          break;
        }
        // Hot lr: read the live override every iteration — the slider frame
        // lands between steps, so the NEXT Adam update uses the new scalar
        // with zero pause and zero metric/moment reset.
        const lrNow = ctx.getLrOverride() ?? p.lr;
        if (lrNow !== activeLr) {
          ctx.log(`learning rate hot-swapped ${activeLr.toExponential(2)} → ${lrNow.toExponential(2)} at step ${step} (loop never paused)`);
          activeLr = lrNow;
        }
        const emitThisStep = shouldEmitStep(step);
        if (emitThisStep) ctx.metric("learning_rate", activeLr, { step });
        const batch = Array.from({ length: p.batchSize }, sampleWindow);
        const t0 = performance.now();
        const { loss, tokensProcessed } = model.step(batch, activeLr);
        const dt = (performance.now() - t0) / 1000;
        let recordLoss = loss;
        finalLoss = loss;
        stepsCompleted = step;

        if (esEnabled) {
          const validationBatch = Array.from({ length: esValidationBatchSize }, sampleWindow);
          const baselineEvalLoss = model.evalLoss(validationBatch);
          if (model.isLora && loraPopulation && model.loraParams.length > 0) {
            const paramCount = model.loraParams.length;
            stochasticMutate(model.loraParams, loraPopulation, paramCount, esPopulationSize, step, esSigma);
            const candidateLosses = new Float32Array(esPopulationSize);
            for (let candidate = 0; candidate < esPopulationSize; candidate++) {
              const candidateOffset = candidate * paramCount;
              candidateLosses[candidate] = model.evalLossWithLora(
                validationBatch,
                loraPopulation.subarray(candidateOffset, candidateOffset + paramCount),
              );
            }
            let bestCandidateLoss = Infinity;
            for (let candidate = 0; candidate < esPopulationSize; candidate++) {
              if (candidateLosses[candidate] < bestCandidateLoss) bestCandidateLoss = candidateLosses[candidate];
            }
            const blend = bestCandidateLoss <= baselineEvalLoss ? esBlend : 0;
            const reduced = gpuReduceFittest(
              model.loraParams, loraPopulation, candidateLosses, paramCount,
              esPopulationSize, esSurvivalCount, step, blend, esSigma,
            );
            const fitness = reduced.ok
              ? reduced
              : { ok: false, ...reduceFittestCpu(model.loraParams, loraPopulation, candidateLosses, paramCount, esPopulationSize, esSurvivalCount, step, blend, esSigma) };
            const lossDelta = fitness.bestLoss - baselineEvalLoss;
            recordLoss = Math.min(recordLoss, fitness.bestLoss);
            finalLoss = recordLoss;
            if (emitThisStep) {
              ctx.metric("es_best_loss", fitness.bestLoss, { step, mode: "lora", bestIndex: fitness.bestIndex, native: reduced.ok });
              ctx.metric("population_variance", fitness.variance, { step, mode: "lora" });
              ctx.metric("stochastic_loss_delta", lossDelta, { step, mode: "lora", baseline: baselineEvalLoss });
              ctx.metric("es_exploration_status", blend > 0 ? 2 : 1, {
                step, mode: "lora", status: blend > 0 ? "survivor_blended" : "survivors_retained", populationSize: esPopulationSize,
              });
            }
          } else if (fullMutationCandidate && w_checkpoint_snapshot) {
            w_checkpoint_snapshot.set(model.params);
            stochasticMutate(w_checkpoint_snapshot, fullMutationCandidate, model.paramCount, 1, step, esSigma);
            const candidateLoss = model.evalLossWithWeights(validationBatch, fullMutationCandidate);
            const accepted = Number.isFinite(candidateLoss) && candidateLoss <= baselineEvalLoss;
            if (accepted) {
              model.loadWeights(fullMutationCandidate);
              recordLoss = Math.min(recordLoss, candidateLoss);
              finalLoss = recordLoss;
            } else {
              model.loadWeights(w_checkpoint_snapshot);
            }
            if (emitThisStep) {
              ctx.metric("es_best_loss", candidateLoss, { step, mode: "full", bestIndex: 0 });
              ctx.metric("population_variance", 0, { step, mode: "full" });
              ctx.metric("stochastic_loss_delta", candidateLoss - baselineEvalLoss, { step, mode: "full", baseline: baselineEvalLoss });
              ctx.metric("es_exploration_status", accepted ? 2 : 0, {
                step, mode: "full", status: accepted ? "mutation_accepted" : "snapshot_restored", populationSize: 1,
              });
            }
          }
        }

        // Best-loss gate: evaluated after EVERY step.
        if (recordLoss < lowestRecordLoss) {
          lowestRecordLoss = recordLoss;
          bestStep = step;
          bestWeights = model.params.slice();
          bestState = model.serializeTrainerState();
          await persistBest(); // best.weights.bin + best.adam.bin
        }

        if (emitThisStep) {
          const tps = tokensProcessed / dt;
          const vram = await sampleVramMb();
          const rssMb = process.memoryUsage().rss / 1024 / 1024;
          if (Number.isFinite(lowestRecordLoss)) {
            ctx.metric("best_loss", lowestRecordLoss, { step: bestStep, sampledAtStep: step });
          }
          ctx.metric("loss", loss, { step });
          ctx.metric("tokens_per_sec", tps, { step });
          ctx.metric("vram_mb", vram, { step });
          ctx.metric("rss_mb", rssMb, { step });
          ctx.metric("node_progress", step / p.steps, { step });
          ctx.log(
            `step ${step}/${p.steps}  loss=${loss.toFixed(4)}  ` +
            `${tps.toFixed(0)} tok/s  vram=${vram}MB  rss=${rssMb.toFixed(0)}MB`
          );
        }
        // Yield to the event loop so WebSocket frames flush between steps.
        await new Promise((r) => setImmediate(r));
      }
    } finally {
      // Weights already live in the host Float32Array; the device-side
      // embedding/grad buffers can be released immediately (idempotent).
      model.backend.dispose();
    }

    // ── Training finalization hook ──
    // Natural completion AND "Commit & Proceed" both land here: the exported
    // deployment state is the BEST-loss snapshot, not the last trailing step.
    const finalWeights = bestWeights ?? model.params;
    const finalState = bestState ?? model.serializeTrainerState();
    if (bestWeights) {
      ctx.log(`Finalizing from best checkpoint: loss ${lowestRecordLoss.toFixed(4)} @ step ${bestStep} ` +
              `(last step was ${Number.isFinite(finalLoss) ? finalLoss.toFixed(4) : "n/a"})`);
    }

    // Persist the checkpoint slot atomically (tmp + rename) so the next run
    // on this variant resumes from the exact optimum captured above.
    if (ckptName) {
      mkdirSync(ckptDir, { recursive: true });
      const tmpW = `${ckptWeights}.tmp-${process.pid}`;
      const tmpA = `${ckptAdam}.tmp-${process.pid}`;
      await writeFile(tmpW, Buffer.from(finalWeights.buffer, finalWeights.byteOffset, finalWeights.byteLength));
      await writeFile(tmpA, Buffer.from(finalState.buffer, finalState.byteOffset, finalState.byteLength));
      renameSync(tmpW, ckptWeights);
      renameSync(tmpA, ckptAdam);
      ctx.log(`Checkpoint "${ckptName}" saved — weights.bin + adam.bin (best-loss state)`);
    }

    // LoRA adapters live inside the trainer state blob — recover the best
    // adapter slice so merged/adapter exports also ship the optimum.
    let finalLora: Float32Array | undefined;
    if (model.isLora) {
      const P = model.paramCount, Q = model.loraParams.length;
      finalLora = bestState ? bestState.slice(1 + 2 * P, 1 + 2 * P + Q) : model.loraParams;
    }

    const handle: TrainedModelHandle = {
      config,
      paramCount: model.paramCount,
      weights: finalWeights,
      ...(finalLora ? { lora: finalLora } : {}),
      finalLoss: Number.isFinite(lowestRecordLoss) ? lowestRecordLoss : finalLoss,
      stepsCompleted,
    };
    ctx.log(
      committedEarly
        ? `Training committed early — ${stepsCompleted}/${p.steps} steps, best loss ${lowestRecordLoss.toFixed(4)}`
        : `Training done — best loss ${Number.isFinite(lowestRecordLoss) ? lowestRecordLoss.toFixed(4) : finalLoss.toFixed(4)}`
    );
    return { model: handle };
  }
}

export const pocTrainerDescriptor = DESCRIPTOR;
