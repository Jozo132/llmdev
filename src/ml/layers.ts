/**
 * Injectable ML building blocks.
 *
 * These registries are the injection points for future custom algorithms:
 * register a WASM/N-API C++ kernel, a PyTorch-bridge mixer, or an exotic loss
 * under a name, then reference that name from the ModelArchitectureNode params
 * — zero engine changes required.
 */

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

// Built-ins.
registerMixer("causal-mean", () => new CausalMeanMixer());
registerLoss("cross-entropy", () => new CrossEntropyLoss());
