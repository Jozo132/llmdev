/**
 * Injectable ML building blocks.
 *
 * These registries are the injection points for future custom algorithms:
 * register a WASM/N-API C++ kernel, a PyTorch-bridge mixer, or an exotic loss
 * under a name, then reference that name from the ModelArchitectureNode params
 * — zero engine changes required.
 */
import { attnLastForwardGpu } from "./backend.js";

/**
 * SequenceMixer = the "attention" slot. Given per-position embeddings x
 * [T × d, row-major], produce mixed states m [T × d] causally, and be able to
 * backpropagate dL/dm into dL/dx.
 */
export interface SequenceMixer {
  readonly name: string;
  forward(x: Float32Array, T: number, d: number, out: Float32Array): void;
  backward(dOut: Float32Array, T: number, d: number, dX: Float32Array): void;
}

/**
 * CausalMeanMixer — parameter-free causal cumulative mean ("attention where
 * every past token gets equal weight"). Deliberately dumb: it exists to prove
 * the slot works. Swap in real softmax attention / a CUDA kernel later.
 *   m_t = (1/(t+1)) Σ_{s≤t} x_s      ⇒      dX_s += Σ_{t≥s} dOut_t/(t+1)
 */
export class CausalMeanMixer implements SequenceMixer {
  readonly name = "causal-mean";

  forward(x: Float32Array, T: number, d: number, out: Float32Array): void {
    const acc = new Float32Array(d);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < d; j++) acc[j] += x[t * d + j];
      const inv = 1 / (t + 1);
      for (let j = 0; j < d; j++) out[t * d + j] = acc[j] * inv;
    }
  }

  backward(dOut: Float32Array, T: number, d: number, dX: Float32Array): void {
    // Suffix-sum of dOut_t/(t+1), accumulated back into each source position.
    const suffix = new Float32Array(d);
    for (let t = T - 1; t >= 0; t--) {
      const inv = 1 / (t + 1);
      for (let j = 0; j < d; j++) suffix[j] += dOut[t * d + j] * inv;
      for (let j = 0; j < d; j++) dX[t * d + j] += suffix[j];
    }
  }
}

/** Loss slot: logits [V] + target id → (loss, dLogits written in place). */
export interface LossFn {
  readonly name: string;
  /** Returns scalar loss; writes dL/dLogits into dLogits. */
  compute(logits: Float32Array, target: number, dLogits: Float32Array): number;
}

/** Numerically-stable softmax cross-entropy with exact analytic gradient. */
export class CrossEntropyLoss implements LossFn {
  readonly name = "cross-entropy";

  compute(logits: Float32Array, target: number, dLogits: Float32Array): number {
    const V = logits.length;
    let max = -Infinity;
    for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    for (let i = 0; i < V; i++) {
      dLogits[i] = Math.exp(logits[i] - max);
      sum += dLogits[i];
    }
    const inv = 1 / sum;
    for (let i = 0; i < V; i++) dLogits[i] *= inv; // now softmax probs
    const loss = -Math.log(Math.max(dLogits[target], 1e-12));
    dLogits[target] -= 1; // ∂CE/∂logits = softmax − onehot
    return loss;
  }
}

// ── Registries ────────────────────────────────────────────────────────────────

const mixers = new Map<string, () => SequenceMixer>();
const losses = new Map<string, () => LossFn>();

export function registerMixer(name: string, factory: () => SequenceMixer): void {
  mixers.set(name, factory);
}
export function registerLoss(name: string, factory: () => LossFn): void {
  losses.set(name, factory);
}
export function createMixer(name: string): SequenceMixer {
  const f = mixers.get(name);
  if (!f) throw new Error(`Unknown mixer "${name}". Registered: ${[...mixers.keys()]}`);
  return f();
}
export function createLoss(name: string): LossFn {
  const f = losses.get(name);
  if (!f) throw new Error(`Unknown loss "${name}". Registered: ${[...losses.keys()]}`);
  return f();
}

/**
 * SoftmaxAttentionMixer — parameter-free causal scaled-dot-product attention
 * (q = k = v = x), engineered for LONG contexts (2048–4096) without OOM:
 *
 *  • FORWARD uses flash-attention-style streaming: keys are consumed in a
 *    single online-softmax pass (running max + running denominator), so peak
 *    memory is O(d) per query instead of materializing the O(T²) score matrix.
 *    When the CUDA addon is built, the tiled `attnLastForward` kernel runs the
 *    same online-softmax on-device in shared-memory KV slices.
 *  • BACKWARD applies activation checkpointing: attention weights are NEVER
 *    stored — they are recomputed from x in two cheap O(T·d) passes, trading
 *    ~2× flops for O(T²)→O(T) memory.
 *
 * The current trainer only backpropagates through the final position's mixed
 * state, so this mixer computes exactly that row (out rows < T−1 stay zero),
 * with the exact softmax Jacobian in backward.
 */
export class SoftmaxAttentionMixer implements SequenceMixer {
  readonly name = "softmax-attn";
  private attnGpu: ((x: Float32Array, T: number, d: number, out: Float32Array) => boolean) | null;

  constructor(attnGpu?: (x: Float32Array, T: number, d: number, out: Float32Array) => boolean) {
    this.attnGpu = attnGpu ?? null;
  }

  forward(x: Float32Array, T: number, d: number, out: Float32Array): void {
    this.lastX = x; // checkpoint: keep the input ref for backward recompute
    if (this.attnGpu && this.attnGpu(x, T, d, out)) return; // CUDA path
    const scale = 1 / Math.sqrt(d);
    const qOff = (T - 1) * d;
    // Online softmax (single streaming pass, O(d) state).
    let runMax = -Infinity;
    let denom = 0;
    const acc = new Float32Array(d);
    for (let j = 0; j < T; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += x[qOff + k] * x[j * d + k];
      s *= scale;
      if (s > runMax) {
        const corr = Math.exp(runMax - s);
        denom *= corr;
        for (let k = 0; k < d; k++) acc[k] *= corr;
        runMax = s;
      }
      const w = Math.exp(s - runMax);
      denom += w;
      for (let k = 0; k < d; k++) acc[k] += w * x[j * d + k];
    }
    const inv = 1 / denom;
    for (let k = 0; k < d; k++) out[qOff + k] = acc[k] * inv;
  }

  backward(dOut: Float32Array, T: number, d: number, dX: Float32Array): void {
    // Activation checkpointing: recompute weights from the stashed input.
    if (!this.lastX) throw new Error("softmax-attn backward called before forward");
    const x = this.lastX;
    const scale = 1 / Math.sqrt(d);
    const qOff = (T - 1) * d;
    const dm = dOut.subarray(qOff, qOff + d);

    // Checkpoint recompute pass 1: max + denominator (no T² storage).
    let max = -Infinity;
    const scores = new Float32Array(T); // O(T), not O(T²)
    for (let j = 0; j < T; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += x[qOff + k] * x[j * d + k];
      scores[j] = s * scale;
      if (scores[j] > max) max = scores[j];
    }
    let denom = 0;
    for (let j = 0; j < T; j++) {
      scores[j] = Math.exp(scores[j] - max);
      denom += scores[j];
    }
    // scores[j] is now a_j (softmax weight).
    for (let j = 0; j < T; j++) scores[j] /= denom;

    // Exact softmax Jacobian: ds_j = a_j (da_j − Σ_k a_k da_k), da_j = dm·x_j
    let mean = 0;
    const da = new Float32Array(T);
    for (let j = 0; j < T; j++) {
      let v = 0;
      for (let k = 0; k < d; k++) v += dm[k] * x[j * d + k];
      da[j] = v;
      mean += scores[j] * v;
    }
    for (let j = 0; j < T; j++) {
      const ds = scores[j] * (da[j] - mean) * scale;
      for (let k = 0; k < d; k++) {
        dX[j * d + k] += scores[j] * dm[k] + ds * x[qOff + k]; // value + key paths
        dX[qOff + k] += ds * x[j * d + k];                     // query path
      }
    }
    this.lastX = null;
  }

  private lastX: Float32Array | null = null;
}

/**
 * EmaMixer ("ssm-ema") — attention-free state-space stand-in:
 *   m_t = α·m_{t−1} + (1−α)·x_t
 * A diagonal linear recurrence — the degenerate core of Mamba/S4-style SSMs
 * (fixed decay instead of input-selective Δ). O(T·d) time, O(d) state, so
 * context length scales linearly. Replace via registerMixer for real Mamba.
 */
export class EmaMixer implements SequenceMixer {
  readonly name = "ssm-ema";
  private readonly alpha = 0.9;

  forward(x: Float32Array, T: number, d: number, out: Float32Array): void {
    const a = this.alpha, b = 1 - a;
    const state = new Float32Array(d);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < d; k++) {
        state[k] = a * state[k] + b * x[t * d + k];
        out[t * d + k] = state[k];
      }
    }
  }

  backward(dOut: Float32Array, T: number, d: number, dX: Float32Array): void {
    // dX_s = (1−α)·Σ_{t≥s} α^{t−s}·dOut_t — reverse-scan running sum.
    const a = this.alpha, b = 1 - a;
    const run = new Float32Array(d);
    for (let t = T - 1; t >= 0; t--) {
      for (let k = 0; k < d; k++) {
        run[k] = dOut[t * d + k] + a * run[k];
        dX[t * d + k] += b * run[k];
      }
    }
  }
}

// Built-ins.
registerMixer("causal-mean", () => new CausalMeanMixer());
registerMixer("softmax-attn", () => new SoftmaxAttentionMixer(attnLastForwardGpu));
registerMixer("ssm-ema", () => new EmaMixer());
registerLoss("cross-entropy", () => new CrossEntropyLoss());
