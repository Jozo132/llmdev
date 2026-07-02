/**
 * TinyLM — a deliberately "dumb" ~1M parameter language model in pure TS.
 *
 * Architecture (per position t):
 *   x_t   = E[token_t] + P[t]                    (embeddings, learned)
 *   m_t   = Mixer(x_{0..t})                      (injectable attention slot)
 *   h_t   = relu(m_t · W1 + b1)                  (MLP up)
 *   y_t   = h_t · W2 + b2                        (MLP down)
 *   logit = y_t · Eᵀ                             (weight-tied output head)
 *
 * Backward pass is EXACT (hand-derived) for every parameter — this is a real
 * gradient-descent trainer, not a mock — while staying small enough to audit.
 * The Mixer and Loss are injected via src/ml/layers.ts registries.
 */
import type { ModelConfig } from "../core/types.js";
import { createBackend, type ComputeBackend } from "./backend.js";
import { createLoss, createMixer, type LossFn, type SequenceMixer } from "./layers.js";

export interface StepResult {
  loss: number;
  tokensProcessed: number;
}

export class TinyLM {
  readonly cfg: ModelConfig;
  readonly params: Float32Array;   // single flat buffer — "the 1M weight array"
  readonly grads: Float32Array;
  private readonly m: Float32Array; // Adam moment 1
  private readonly v: Float32Array; // Adam moment 2
  private adamT = 0;

  // Views into the flat buffer (offsets, row-major).
  private readonly oE: number;   // [V × d]
  private readonly oP: number;   // [ctx × d]
  private readonly oW1: number;  // [d × h]
  private readonly oB1: number;  // [h]
  private readonly oW2: number;  // [h × d]
  private readonly oB2: number;  // [d]

  private mixer: SequenceMixer;
  private loss: LossFn;
  readonly backend: ComputeBackend;

  constructor(cfg: ModelConfig, backend?: ComputeBackend) {
    this.cfg = cfg;
    const { vocabSize: V, dModel: d, contextLength: ctx, hiddenDim: h } = cfg;
    this.oE = 0;
    this.oP = this.oE + V * d;
    this.oW1 = this.oP + ctx * d;
    this.oB1 = this.oW1 + d * h;
    this.oW2 = this.oB1 + h;
    this.oB2 = this.oW2 + h * d;
    const total = this.oB2 + d;

    this.params = new Float32Array(total);
    this.grads = new Float32Array(total);
    this.m = new Float32Array(total);
    this.v = new Float32Array(total);
    this.mixer = createMixer(cfg.mixer);
    this.loss = createLoss(cfg.loss);
    this.initWeights();
    // The heavy O(V·d) tied-head math is delegated to a backend (CPU or the
    // N-API CUDA bridge) which pins E + grad-E in device memory.
    this.backend = backend ?? createBackend();
    this.backend.init(this.params.subarray(this.oE, this.oE + V * d), V, d);
  }

  get paramCount(): number {
    return this.params.length;
  }

  /**
   * Named-tensor layout of the flat parameter buffer — the serialization
   * contract used by the GGUF / safetensors exporters (src/core/Exporters.ts).
   * Offsets are element (not byte) offsets; shapes are row-major.
   */
  static tensorLayout(cfg: ModelConfig): Array<{ name: string; offset: number; shape: number[] }> {
    const { vocabSize: V, dModel: d, contextLength: ctx, hiddenDim: h } = cfg;
    let o = 0;
    const t = (name: string, shape: number[]) => {
      const entry = { name, offset: o, shape };
      o += shape.reduce((a, b) => a * b, 1);
      return entry;
    };
    return [
      t("token_embd.weight", [V, d]),   // tied output head reads this too
      t("pos_embd.weight", [ctx, d]),
      t("ffn_up.weight", [d, h]),
      t("ffn_up.bias", [h]),
      t("ffn_down.weight", [h, d]),
      t("ffn_down.bias", [d]),
    ];
  }

  private initWeights(): void {
    // Small deterministic-ish gaussian init (Box–Muller).
    const scale = 0.02;
    for (let i = 0; i < this.params.length; i++) {
      const u1 = Math.random() || 1e-9;
      const u2 = Math.random();
      this.params[i] = scale * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
  }

  /**
   * One training step on a batch of token windows.
   * Each window predicts its LAST token from the preceding context
   * (keeps pure-JS compute tractable; per-position loss is a drop-in later).
   */
  step(batch: Uint16Array[], lr: number): StepResult {
    const { vocabSize: V, dModel: d, hiddenDim: h } = this.cfg;
    const p = this.params;
    const g = this.grads;
    g.fill(0);

    let totalLoss = 0;
    let tokens = 0;

    // Reusable scratch buffers (no per-step allocation churn ⇒ no OOM creep).
    const dLogits = new Float32Array(V);

    for (const window of batch) {
      const T = window.length - 1;      // context positions
      const target = window[T];         // next-token target
      tokens += window.length;

      // ── Forward ────────────────────────────────────────────────────────────
      const x = new Float32Array(T * d);
      for (let t = 0; t < T; t++) {
        const eOff = this.oE + window[t] * d;
        const pOff = this.oP + t * d;
        for (let j = 0; j < d; j++) x[t * d + j] = p[eOff + j] + p[pOff + j];
      }

      const mix = new Float32Array(T * d);
      this.mixer.forward(x, T, d, mix);
      const mLast = mix.subarray((T - 1) * d, T * d); // predict from final state

      const hid = new Float32Array(h);
      for (let k = 0; k < h; k++) {
        let acc = p[this.oB1 + k];
        for (let j = 0; j < d; j++) acc += mLast[j] * p[this.oW1 + j * h + k];
        hid[k] = acc > 0 ? acc : 0; // relu
      }

      const y = new Float32Array(d);
      for (let j = 0; j < d; j++) {
        let acc = p[this.oB2 + j];
        for (let k = 0; k < h; k++) acc += hid[k] * p[this.oW2 + k * d + j];
        y[j] = acc;
      }

      // Tied output head: logits[i] = y · E[i]
      const logits = new Float32Array(V);
      for (let i = 0; i < V; i++) {
        let acc = 0;
        const eOff = this.oE + i * d;
        for (let j = 0; j < d; j++) acc += y[j] * p[eOff + j];
        logits[i] = acc;
      }

      // ── Loss + Backward ────────────────────────────────────────────────────
      totalLoss += this.loss.compute(logits, target, dLogits);

      // d y and d E (tied head): logits = E · y
      const dY = new Float32Array(d);
      for (let i = 0; i < V; i++) {
        const dl = dLogits[i];
        if (dl === 0) continue;
        const eOff = this.oE + i * d;
        for (let j = 0; j < d; j++) {
          dY[j] += dl * p[eOff + j];
          g[eOff + j] += dl * y[j];
        }
      }

      // MLP down: y = hid·W2 + b2
      const dHid = new Float32Array(h);
      for (let j = 0; j < d; j++) {
        const dyj = dY[j];
        g[this.oB2 + j] += dyj;
        for (let k = 0; k < h; k++) {
          g[this.oW2 + k * d + j] += hid[k] * dyj;
          dHid[k] += p[this.oW2 + k * d + j] * dyj;
        }
      }

      // relu + MLP up: hid = relu(mLast·W1 + b1)
      const dMLast = new Float32Array(d);
      for (let k = 0; k < h; k++) {
        if (hid[k] <= 0) continue; // relu gate
        const dhk = dHid[k];
        g[this.oB1 + k] += dhk;
        for (let j = 0; j < d; j++) {
          g[this.oW1 + j * h + k] += mLast[j] * dhk;
          dMLast[j] += p[this.oW1 + j * h + k] * dhk;
        }
      }

      // Mixer backward: only the last mixed state received gradient.
      const dMix = new Float32Array(T * d);
      dMix.set(dMLast, (T - 1) * d);
      const dX = new Float32Array(T * d);
      this.mixer.backward(dMix, T, d, dX);

      // Embedding + positional gradients.
      for (let t = 0; t < T; t++) {
        const eOff = this.oE + window[t] * d;
        const pOff = this.oP + t * d;
        for (let j = 0; j < d; j++) {
          g[eOff + j] += dX[t * d + j];
          g[pOff + j] += dX[t * d + j];
        }
      }
    }

    // ── Adam update over the flat 1M-param buffer ───────────────────────
    this.backend.flushGradE(g.subarray(this.oE, this.oE + V * d));
    const invB = 1 / batch.length;
    this.adamT++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.adamT);
    const c2 = 1 - Math.pow(b2, this.adamT);
    for (let i = 0; i < p.length; i++) {
      const gi = g[i] * invB;
      this.m[i] = b1 * this.m[i] + (1 - b1) * gi;
      this.v[i] = b2 * this.v[i] + (1 - b2) * gi * gi;
      p[i] -= (lr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + eps);
    }
    // Weights changed — mirror E to the device (no-op on CPU).
    this.backend.syncE();

    return { loss: totalLoss / batch.length, tokensProcessed: tokens };
  }

  /** Loss-only evaluation (no gradient accumulation side effects kept). */
  evalLoss(batch: Uint16Array[]): number {
    const saved = this.params.slice();
    const savedM = this.m.slice();
    const savedV = this.v.slice();
    const savedT = this.adamT;
    const { loss } = this.step(batch, 0); // lr=0 ⇒ params unchanged by update
    this.params.set(saved);
    this.m.set(savedM);
    this.v.set(savedV);
    this.adamT = savedT;
    return loss;
  }

  /** Greedy sampling for smoke-test generations. */
  generate(prompt: Uint16Array, maxNew: number): number[] {
    const out = Array.from(prompt);
    for (let n = 0; n < maxNew; n++) out.push(this.nextToken(out, 0));
    return out;
  }

  /**
   * Predict the next token for a context (public inference API used by the
   * Chat Sandbox). temperature 0 ⇒ greedy argmax; >0 ⇒ softmax sampling.
   * topP < 1 restricts sampling to the smallest set of tokens whose
   * cumulative probability ≥ topP (nucleus sampling — cuts the long tail).
   */
  nextToken(contextIds: ArrayLike<number>, temperature = 0, topP = 1): number {
    const logits = this.logitsFor(contextIds);
    const V = logits.length;
    if (temperature <= 0) {
      let best = 0;
      for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    let max = -Infinity;
    for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    const probs = new Float32Array(V);
    for (let i = 0; i < V; i++) {
      probs[i] = Math.exp((logits[i] - max) / temperature);
      sum += probs[i];
    }
    for (let i = 0; i < V; i++) probs[i] /= sum;

    if (topP < 1) {
      // Nucleus: keep highest-prob tokens until cumulative ≥ topP.
      const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
      let cum = 0;
      let cut = V;
      for (let r = 0; r < V; r++) {
        cum += probs[order[r]];
        if (cum >= topP) { cut = r + 1; break; }
      }
      let r = Math.random() * cum;
      for (let i = 0; i < cut; i++) {
        r -= probs[order[i]];
        if (r <= 0) return order[i];
      }
      return order[cut - 1];
    }

    let r = Math.random();
    for (let i = 0; i < V; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return V - 1;
  }

  /** Full next-token logits for the last ≤ctx−1 ids of the context. */
  logitsFor(contextIds: ArrayLike<number>): Float32Array {
    const { vocabSize: V, dModel: d, hiddenDim: h, contextLength: ctx } = this.cfg;
    const p = this.params;
    const all = Array.from(contextIds);
    const window = all.slice(Math.max(0, all.length - (ctx - 1)));
    const T = Math.max(1, window.length);
    const x = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      const eOff = this.oE + (window[t] ?? 0) * d;
      const pOff = this.oP + t * d;
      for (let j = 0; j < d; j++) x[t * d + j] = p[eOff + j] + p[pOff + j];
    }
    const mix = new Float32Array(T * d);
    this.mixer.forward(x, T, d, mix);
    const mLast = mix.subarray((T - 1) * d, T * d);
    const hid = new Float32Array(h);
    for (let k = 0; k < h; k++) {
      let acc = p[this.oB1 + k];
      for (let j = 0; j < d; j++) acc += mLast[j] * p[this.oW1 + j * h + k];
      hid[k] = acc > 0 ? acc : 0;
    }
    const y = new Float32Array(d);
    for (let j = 0; j < d; j++) {
      let acc = p[this.oB2 + j];
      for (let k = 0; k < h; k++) acc += hid[k] * p[this.oW2 + k * d + j];
      y[j] = acc;
    }
    const logits = new Float32Array(V);
    this.backend.logitsForward(y, logits);
    return logits;
  }
}
