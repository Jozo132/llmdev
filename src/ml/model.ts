/**
 * TinyLM — a compact multi-layer transformer LM in pure TS (CUDA-assisted).
 *
 * Architecture (deep stack, L = cfg.nLayers layers):
 *   x_t      = E[token_t] + P[t]                       (learned embeddings)
 *   h^0      = x
 *   per layer l ∈ [0, L):
 *     q,k,v  = h^l·Wq_l , h^l·Wk_l , h^l·Wv_l          (per-layer projections)
 *     m      = causal-softmax(q·kᵀ/√d)·v               ("softmax-attn")
 *              — or an injected registry mixer applied to v
 *     a      = h^l + m                                 (residual)
 *     u      = relu(a·Wg_l + bg_l)                     (MLP gate/up)
 *     h^l+1  = a + u·Wd_l + bd_l                       (MLP down, residual)
 *   logits   = h^L[T−1] · Eᵀ                           (weight-tied head)
 *
 * FORWARD runs the hidden state sequentially layer 0 → L−1; BACKWARD is EXACT
 * (hand-derived) and walks the stack in reverse order L−1 → 0, propagating
 * activation gradients through residuals, the softmax Jacobian, and the
 * per-layer projection matrices. The tied-head O(V·d) math is delegated to
 * the ComputeBackend (CPU or the N-API CUDA bridge); per-layer weight
 * mirrors are allocated as discrete device buffers via backend.initLayers.
 */
import type { ModelConfig } from "../core/types.js";
import { createBackend, gpuSgemmAcc, type ComputeBackend } from "./backend.js";
import { createLoss, createMixer, type LossFn, type SequenceMixer } from "./layers.js";

export interface StepResult {
  loss: number;
  tokensProcessed: number;
}

/** Element offsets of one layer's weights inside the flat parameter buffer. */
interface LayerOffsets {
  wq: number;  // [d × d] attention query projection
  wk: number;  // [d × d] attention key projection
  wv: number;  // [d × d] attention value projection
  wg: number;  // [d × h] MLP gate/up
  bg: number;  // [h]
  wd: number;  // [h × d] MLP down
  bd: number;  // [d]
}

/** Per-layer activation checkpoint recorded by forward for exact backward. */
interface LayerCache {
  hIn: Float32Array;          // [T × d] layer input
  q: Float32Array;            // [T × d]
  k: Float32Array;            // [T × d]
  v: Float32Array;            // [T × d]
  probs: Float32Array | null; // [T × T] causal softmax rows (attention path)
  a: Float32Array;            // [T × d] post-attention residual state
  u: Float32Array;            // [T × h] post-relu MLP activations
  xaq: Float32Array | null;   // [T × r] LoRA X·Aq activation (adapter path)
  xav: Float32Array | null;   // [T × r] LoRA X·Av activation
}

/** Element offsets of one layer's LoRA adapters inside the flat lora buffer. */
interface LoraOffsets {
  aq: number; // [d × r] A for Wq (gaussian init)
  bq: number; // [r × d] B for Wq (zero init ⇒ initial delta B·A = 0)
  av: number; // [d × r] A for Wv
  bv: number; // [r × d] B for Wv
}

const layerParamCount = (d: number, h: number): number =>
  3 * d * d + d * h + h + h * d + d;

/** LoRA adapter parameter count (q + v targets × (A + B) per layer). */
const loraLayerParamCount = (d: number, r: number): number => 4 * d * r;

export class TinyLM {
  readonly cfg: ModelConfig;
  readonly params: Float32Array;   // single flat buffer — the weight array
  readonly grads: Float32Array;
  private readonly m: Float32Array; // Adam moment 1
  private readonly v: Float32Array; // Adam moment 2
  private adamT = 0;

  // ── LoRA fine-tuning state (empty buffers when fineTuneMode !== "lora") ──
  /** Flat adapter buffer: per layer Aq[d×r]·Bq[r×d]·Av[d×r]·Bv[r×d]. */
  readonly loraParams: Float32Array;
  readonly loraGrads: Float32Array;
  private readonly loraM: Float32Array;
  private readonly loraV: Float32Array;
  private readonly lora: { r: number; scale: number; layers: LoraOffsets[] } | null;

  private readonly L: number;      // transformer depth (nLayers)
  private readonly oE: number;     // [V × d]
  private readonly oP: number;     // [ctx × d]
  private readonly layers: LayerOffsets[];

  /** True ⇒ built-in full causal softmax attention; false ⇒ registry mixer on v. */
  private readonly useAttention: boolean;
  private readonly mixers: SequenceMixer[]; // per-layer instances (non-attn path)
  private loss: LossFn;
  readonly backend: ComputeBackend;

  constructor(cfg: ModelConfig, backend?: ComputeBackend, opts?: { skipInit?: boolean }) {
    this.cfg = cfg;
    const { vocabSize: V, dModel: d, contextLength: ctx, hiddenDim: h } = cfg;
    const L = Math.max(1, cfg.nLayers ?? 1);
    this.L = L;
    this.oE = 0;
    this.oP = this.oE + V * d;
    let o = this.oP + ctx * d;
    this.layers = Array.from({ length: L }, () => {
      const l: LayerOffsets = {
        wq: o, wk: o + d * d, wv: o + 2 * d * d,
        wg: o + 3 * d * d, bg: o + 3 * d * d + d * h,
        wd: o + 3 * d * d + d * h + h, bd: o + 3 * d * d + d * h + h + h * d,
      };
      o += layerParamCount(d, h);
      return l;
    });

    this.params = new Float32Array(o);
    this.grads = new Float32Array(o);
    this.m = new Float32Array(o);
    this.v = new Float32Array(o);
    this.useAttention = cfg.mixer === "softmax-attn";
    // Registry mixers (causal-mean, ssm-ema, custom) mix the v-projection;
    // one instance per layer because backward keeps per-forward state.
    this.mixers = this.useAttention
      ? []
      : Array.from({ length: L }, () => createMixer(cfg.mixer));
    this.loss = createLoss(cfg.loss);

    // ── LoRA adapters on the attention q/v projections ──
    if (cfg.fineTuneMode === "lora") {
      const r = Math.min(Math.max(1, Math.round(cfg.loraRank ?? 8)), d);
      const alpha = cfg.loraAlpha ?? 16;
      const per = loraLayerParamCount(d, r);
      this.loraParams = new Float32Array(L * per);
      this.lora = {
        r,
        scale: alpha / r,
        layers: Array.from({ length: L }, (_, l) => ({
          aq: l * per,
          bq: l * per + d * r,
          av: l * per + d * r + r * d,
          bv: l * per + 2 * d * r + r * d,
        })),
      };
      // A ← gaussian, B ← 0 ⇒ initial adapter delta B×A = 0 (identity start).
      for (const lo of this.lora.layers) {
        for (let i = lo.aq; i < lo.aq + d * r; i++) this.loraParams[i] = this.gauss(0.02);
        for (let i = lo.av; i < lo.av + d * r; i++) this.loraParams[i] = this.gauss(0.02);
      }
    } else {
      this.lora = null;
      this.loraParams = new Float32Array(0);
    }
    this.loraGrads = new Float32Array(this.loraParams.length);
    this.loraM = new Float32Array(this.loraParams.length);
    this.loraV = new Float32Array(this.loraParams.length);

    // Warm-start callers (checkpoint resumption) pass skipInit and load real
    // weights via loadWeights() — random init is skipped entirely.
    if (!opts?.skipInit) this.initWeights();
    // The heavy O(V·d) tied-head math is delegated to a backend (CPU or the
    // N-API CUDA bridge) which pins E + grad-E in device memory.
    this.backend = backend ?? createBackend();
    this.backend.init(this.params.subarray(this.oE, this.oE + V * d), V, d);
    // Allocate discrete per-layer device weight buffers + push initial values.
    this.backend.initLayers?.(L, d, h);
    this.syncLayerWeights();
  }

  get paramCount(): number {
    return this.params.length;
  }

  get nLayers(): number {
    return this.L;
  }

  /** True when the base weights are frozen and only LoRA A/B train. */
  get isLora(): boolean {
    return this.lora !== null;
  }

  /** LoRA scaling factor α/r (0 when LoRA is inactive). */
  get loraScale(): number {
    return this.lora?.scale ?? 0;
  }

  /**
   * Named-tensor layout of the flat parameter buffer — the serialization
   * contract used by the GGUF / safetensors exporters (src/core/Exporters.ts).
   * Per-layer tensors follow llama.cpp naming (blk.{i}.attn_q.weight …) so
   * external runtimes can address every layer index. Offsets are element
   * (not byte) offsets; shapes are row-major.
   */
  static tensorLayout(cfg: ModelConfig): Array<{ name: string; offset: number; shape: number[] }> {
    const { vocabSize: V, dModel: d, contextLength: ctx, hiddenDim: h } = cfg;
    const L = Math.max(1, cfg.nLayers ?? 1);
    let o = 0;
    const t = (name: string, shape: number[]) => {
      const entry = { name, offset: o, shape };
      o += shape.reduce((a, b) => a * b, 1);
      return entry;
    };
    const out = [
      t("token_embd.weight", [V, d]),   // tied output head reads this too
      t("pos_embd.weight", [ctx, d]),
    ];
    for (let l = 0; l < L; l++) {
      out.push(
        t(`blk.${l}.attn_q.weight`, [d, d]),
        t(`blk.${l}.attn_k.weight`, [d, d]),
        t(`blk.${l}.attn_v.weight`, [d, d]),
        t(`blk.${l}.ffn_gate.weight`, [d, h]),
        t(`blk.${l}.ffn_gate.bias`, [h]),
        t(`blk.${l}.ffn_down.weight`, [h, d]),
        t(`blk.${l}.ffn_down.bias`, [d]),
      );
    }
    return out;
  }

  /** Total parameter count for a config (mirrors the constructor layout). */
  static paramCountFor(cfg: ModelConfig): number {
    const { vocabSize: V, dModel: d, contextLength: ctx, hiddenDim: h } = cfg;
    const L = Math.max(1, cfg.nLayers ?? 1);
    return V * d + ctx * d + L * layerParamCount(d, h);
  }

  /** LoRA adapter parameter count for a config (0 unless fineTuneMode=lora). */
  static loraParamCountFor(cfg: ModelConfig): number {
    if (cfg.fineTuneMode !== "lora") return 0;
    const d = cfg.dModel;
    const r = Math.min(Math.max(1, Math.round(cfg.loraRank ?? 8)), d);
    return Math.max(1, cfg.nLayers ?? 1) * loraLayerParamCount(d, r);
  }

  /**
   * Decompose a flat LoRA buffer into per-target adapters for exporters:
   * merged-checkpoint fusion (W' = W + scale·A·B) or isolated adapter files.
   */
  static loraAdaptersFor(
    cfg: ModelConfig, lora: Float32Array
  ): Array<{ target: string; A: Float32Array; B: Float32Array; scale: number; d: number; r: number }> {
    if (cfg.fineTuneMode !== "lora") return [];
    const d = cfg.dModel;
    const r = Math.min(Math.max(1, Math.round(cfg.loraRank ?? 8)), d);
    const scale = (cfg.loraAlpha ?? 16) / r;
    const L = Math.max(1, cfg.nLayers ?? 1);
    const per = loraLayerParamCount(d, r);
    const out: Array<{ target: string; A: Float32Array; B: Float32Array; scale: number; d: number; r: number }> = [];
    for (let l = 0; l < L; l++) {
      const o = l * per;
      out.push(
        { target: `blk.${l}.attn_q.weight`, A: lora.subarray(o, o + d * r),
          B: lora.subarray(o + d * r, o + 2 * d * r), scale, d, r },
        { target: `blk.${l}.attn_v.weight`, A: lora.subarray(o + 2 * d * r, o + 3 * d * r),
          B: lora.subarray(o + 3 * d * r, o + 4 * d * r), scale, d, r },
      );
    }
    return out;
  }

  private gauss(s: number): number {
    const u1 = Math.random() || 1e-9;
    const u2 = Math.random();
    return s * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  private initWeights(): void {
    // Small gaussian init (Box–Muller); projection scale shrinks with depth
    // (GPT-2-style 1/√(2L) residual-branch damping keeps deep stacks stable).
    const scale = 0.02;
    const residScale = scale / Math.sqrt(2 * this.L);
    const embEnd = this.oP + this.cfg.contextLength * this.cfg.dModel;
    for (let i = 0; i < this.params.length; i++) {
      const u1 = Math.random() || 1e-9;
      const u2 = Math.random();
      const s = i < embEnd ? scale : residScale;
      this.params[i] = s * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    // Biases start at zero.
    const { dModel: d, hiddenDim: h } = this.cfg;
    for (const l of this.layers) {
      this.params.fill(0, l.bg, l.bg + h);
      this.params.fill(0, l.bd, l.bd + d);
    }
  }

  /** Mirror per-layer weight matrices to the device (no-op on CPU). */
  private syncLayerWeights(): void {
    if (!this.backend.syncLayer) return;
    const { dModel: d, hiddenDim: h } = this.cfg;
    const p = this.params;
    for (let l = 0; l < this.L; l++) {
      const off = this.layers[l];
      this.backend.syncLayer(
        l,
        p.subarray(off.wq, off.wq + d * d),
        p.subarray(off.wk, off.wk + d * d),
        p.subarray(off.wv, off.wv + d * d),
        p.subarray(off.wg, off.wg + d * h),
        p.subarray(off.wd, off.wd + h * d),
      );
    }
  }

  // ── Warm-start / checkpoint resumption ────────────────────────────────

  /**
   * Load a full weight checkpoint into the host buffer AND stream it into the
   * hot GPU device contexts (E via syncE, every layer's discrete buffers via
   * llm_ctx_sync_layer) so training/inference resumes seamlessly.
   */
  loadWeights(w: Float32Array): void {
    if (w.length !== this.params.length) {
      throw new Error(`Weight shape mismatch: ${w.length} vs ${this.params.length}`);
    }
    this.params.set(w);
    this.backend.syncE();
    this.syncLayerWeights();
  }

  loadLoraAdapters(lora: Float32Array): void {
    if (lora.length !== this.loraParams.length) {
      throw new Error(`LoRA shape mismatch: ${lora.length} vs ${this.loraParams.length}`);
    }
    this.loraParams.set(lora);
  }

  evalLossWithWeights(batch: Uint16Array[], weights: Float32Array): number {
    const saved = this.params.slice();
    this.loadWeights(weights);
    const loss = this.evalLoss(batch);
    this.loadWeights(saved);
    return loss;
  }

  evalLossWithLora(batch: Uint16Array[], lora: Float32Array): number {
    const saved = this.loraParams.slice();
    this.loadLoraAdapters(lora);
    const loss = this.evalLoss(batch);
    this.loadLoraAdapters(saved);
    return loss;
  }

  /** Expected serializeTrainerState length for a config (adam.bin contract). */
  static trainerStateLengthFor(cfg: ModelConfig): number {
    return 1 + 2 * TinyLM.paramCountFor(cfg) + 3 * TinyLM.loraParamCountFor(cfg);
  }

  /**
   * Optimizer-state snapshot: [adamT, m, v, loraParams, loraM, loraV].
   * Written to *.adam.bin so a later run resumes from the exact step
   * (bias-correction schedule included) it left off.
   */
  serializeTrainerState(): Float32Array {
    const P = this.params.length, Q = this.loraParams.length;
    const out = new Float32Array(1 + 2 * P + 3 * Q);
    out[0] = this.adamT;
    out.set(this.m, 1);
    out.set(this.v, 1 + P);
    out.set(this.loraParams, 1 + 2 * P);
    out.set(this.loraM, 1 + 2 * P + Q);
    out.set(this.loraV, 1 + 2 * P + 2 * Q);
    return out;
  }

  /** Restore an optimizer-state snapshot; false on shape mismatch. */
  restoreTrainerState(s: Float32Array): boolean {
    const P = this.params.length, Q = this.loraParams.length;
    if (s.length !== 1 + 2 * P + 3 * Q) return false;
    this.adamT = s[0];
    this.m.set(s.subarray(1, 1 + P));
    this.v.set(s.subarray(1 + P, 1 + 2 * P));
    this.loraParams.set(s.subarray(1 + 2 * P, 1 + 2 * P + Q));
    this.loraM.set(s.subarray(1 + 2 * P + Q, 1 + 2 * P + 2 * Q));
    this.loraV.set(s.subarray(1 + 2 * P + 2 * Q));
    return true;
  }

  /** Resumed Adam step counter (0 for a fresh model). */
  get adamStep(): number {
    return this.adamT;
  }

  // ── Forward stack ──────────────────────────────────────────────────────────

  /**
   * Run the hidden state sequentially through layers 0 → L−1.
   * When `caches` is provided, per-layer activations are checkpointed for the
   * exact reverse-order backward pass. Returns the final hidden states [T×d].
   */
  private forwardStack(window: ArrayLike<number>, T: number, caches: LayerCache[] | null): Float32Array {
    const { dModel: d, hiddenDim: h } = this.cfg;
    const p = this.params;

    let hState = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      const eOff = this.oE + (window[t] ?? 0) * d;
      const pOff = this.oP + t * d;
      for (let j = 0; j < d; j++) hState[t * d + j] = p[eOff + j] + p[pOff + j];
    }

    const scale = 1 / Math.sqrt(d);
    for (let l = 0; l < this.L; l++) {
      const off = this.layers[l];
      const hIn = hState;

      // Projections: q/k/v = hIn · W  ([T×d]·[d×d])
      const q = new Float32Array(T * d);
      const k = new Float32Array(T * d);
      const v = new Float32Array(T * d);
      for (let t = 0; t < T; t++) {
        for (let j = 0; j < d; j++) {
          const xtj = hIn[t * d + j];
          if (xtj === 0) continue;
          const rq = off.wq + j * d, rk = off.wk + j * d, rv = off.wv + j * d;
          for (let c = 0; c < d; c++) {
            q[t * d + c] += xtj * p[rq + c];
            k[t * d + c] += xtj * p[rk + c];
            v[t * d + c] += xtj * p[rv + c];
          }
        }
      }

      // LoRA adapter path: q += (α/r)·(X·Aq)·Bq, v += (α/r)·(X·Av)·Bv.
      // The [T×r] bottleneck activations are cached for the A/B backward.
      let xaq: Float32Array | null = null;
      let xav: Float32Array | null = null;
      if (this.lora) {
        const { r, scale: s } = this.lora;
        const lo = this.lora.layers[l];
        const lp = this.loraParams;
        xaq = new Float32Array(T * r);
        xav = new Float32Array(T * r);
        for (let t = 0; t < T; t++) {
          for (let j = 0; j < d; j++) {
            const xtj = hIn[t * d + j];
            if (xtj === 0) continue;
            const ra = lo.aq + j * r, rv2 = lo.av + j * r;
            for (let c = 0; c < r; c++) {
              xaq[t * r + c] += xtj * lp[ra + c];
              xav[t * r + c] += xtj * lp[rv2 + c];
            }
          }
        }
        const bq = lp.subarray(lo.bq, lo.bq + r * d);
        const bv = lp.subarray(lo.bv, lo.bv + r * d);
        // CUDA k_sgemm_acc when the matmul is big enough to beat the PCIe
        // round-trip; exact CPU fallback otherwise (identical semantics).
        const gpuWorthIt = T * d >= 16384;
        if (!(gpuWorthIt && gpuSgemmAcc(xaq, bq, q, T, r, d, s))) {
          for (let t = 0; t < T; t++)
            for (let c = 0; c < r; c++) {
              const w = s * xaq[t * r + c];
              if (w === 0) continue;
              for (let j = 0; j < d; j++) q[t * d + j] += w * bq[c * d + j];
            }
        }
        if (!(gpuWorthIt && gpuSgemmAcc(xav, bv, v, T, r, d, s))) {
          for (let t = 0; t < T; t++)
            for (let c = 0; c < r; c++) {
              const w = s * xav[t * r + c];
              if (w === 0) continue;
              for (let j = 0; j < d; j++) v[t * d + j] += w * bv[c * d + j];
            }
        }
      }

      // Sequence mixing → m [T×d]
      const mMix = new Float32Array(T * d);
      let probs: Float32Array | null = null;
      if (this.useAttention) {
        // Full causal softmax attention, rows checkpointed for backward.
        probs = new Float32Array(T * T);
        for (let t = 0; t < T; t++) {
          let max = -Infinity;
          for (let j = 0; j <= t; j++) {
            let s = 0;
            for (let c = 0; c < d; c++) s += q[t * d + c] * k[j * d + c];
            s *= scale;
            probs[t * T + j] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let j = 0; j <= t; j++) {
            const w = Math.exp(probs[t * T + j] - max);
            probs[t * T + j] = w;
            sum += w;
          }
          const inv = 1 / sum;
          for (let j = 0; j <= t; j++) {
            const w = (probs[t * T + j] *= inv);
            for (let c = 0; c < d; c++) mMix[t * d + c] += w * v[j * d + c];
          }
        }
      } else {
        this.mixers[l].forward(v, T, d, mMix);
      }

      // Residual + MLP (per position).
      const a = new Float32Array(T * d);
      for (let i = 0; i < T * d; i++) a[i] = hIn[i] + mMix[i];

      const u = new Float32Array(T * h);
      const hOut = new Float32Array(T * d);
      for (let t = 0; t < T; t++) {
        for (let c = 0; c < h; c++) {
          let acc = p[off.bg + c];
          for (let j = 0; j < d; j++) acc += a[t * d + j] * p[off.wg + j * h + c];
          u[t * h + c] = acc > 0 ? acc : 0; // relu
        }
        for (let j = 0; j < d; j++) {
          let acc = p[off.bd + j] + a[t * d + j]; // residual
          for (let c = 0; c < h; c++) acc += u[t * h + c] * p[off.wd + c * d + j];
          hOut[t * d + j] = acc;
        }
      }

      caches?.push({ hIn, q, k, v, probs, a, u, xaq, xav });
      hState = hOut;
    }
    return hState;
  }

  /**
   * One training step on a batch of token windows.
   * Each window predicts its LAST token from the preceding context; gradients
   * backpropagate through the whole depth in reverse layer order L−1 → 0.
   */
  step(batch: Uint16Array[], lr: number): StepResult {
    const { vocabSize: V, dModel: d, hiddenDim: h } = this.cfg;
    const p = this.params;
    const g = this.grads;
    g.fill(0);
    this.loraGrads.fill(0);
    // LoRA ⇒ base weights are FROZEN: skip every base-gradient write — only
    // the A/B adapters accumulate gradient (and only they sync to device).
    const frozen = this.lora !== null;

    let totalLoss = 0;
    let tokens = 0;

    const dLogits = new Float32Array(V);
    const logits = new Float32Array(V);
    const scale = 1 / Math.sqrt(d);

    for (const window of batch) {
      const T = window.length - 1;      // context positions
      const target = window[T];         // next-token target
      tokens += window.length;

      // ── Forward: layer 0 → L−1 with activation checkpoints ────────────────
      const caches: LayerCache[] = [];
      const hFinal = this.forwardStack(window, T, caches);
      const y = hFinal.subarray((T - 1) * d, T * d);

      // Tied output head via the ComputeBackend (CPU or CUDA).
      this.backend.logitsForward(y, logits);

      // ── Loss + head backward ───────────────────────────────────────────────
      totalLoss += this.loss.compute(logits, target, dLogits);
      const dY = new Float32Array(d);
      this.backend.logitsBackward(y, dLogits, dY); // accumulates grad-E internally

      // ── Backward: layer L−1 → 0 (exact, reverse order) ─────────────────────
      let dH = new Float32Array(T * d);
      dH.set(dY, (T - 1) * d);

      for (let l = this.L - 1; l >= 0; l--) {
        const off = this.layers[l];
        const { hIn, q, k, v, probs, a, u, xaq, xav } = caches[l];
        const dOut = dH;

        // hOut = a + u·Wd + bd
        const dA = new Float32Array(T * d);
        dA.set(dOut);
        const dU = new Float32Array(T * h);
        for (let t = 0; t < T; t++) {
          for (let j = 0; j < d; j++) {
            const doj = dOut[t * d + j];
            if (doj === 0) continue;
            if (!frozen) g[off.bd + j] += doj;
            for (let c = 0; c < h; c++) {
              if (!frozen) g[off.wd + c * d + j] += u[t * h + c] * doj;
              dU[t * h + c] += p[off.wd + c * d + j] * doj;
            }
          }
        }
        // u = relu(a·Wg + bg)
        for (let t = 0; t < T; t++) {
          for (let c = 0; c < h; c++) {
            if (u[t * h + c] <= 0) continue; // relu gate
            const duc = dU[t * h + c];
            if (duc === 0) continue;
            if (!frozen) g[off.bg + c] += duc;
            for (let j = 0; j < d; j++) {
              if (!frozen) g[off.wg + j * h + c] += a[t * d + j] * duc;
              dA[t * d + j] += p[off.wg + j * h + c] * duc;
            }
          }
        }

        // a = hIn + m  ⇒ dM = dA, dHin starts as dA
        const dHin = new Float32Array(T * d);
        dHin.set(dA);
        const dQ = new Float32Array(T * d);
        const dK = new Float32Array(T * d);
        const dV = new Float32Array(T * d);

        if (this.useAttention && probs) {
          // m_t = Σ_{j≤t} P_tj·v_j — exact softmax-Jacobian backward.
          const dPRow = new Float32Array(T);
          for (let t = 0; t < T; t++) {
            let rowDot = 0;
            for (let j = 0; j <= t; j++) {
              let dp = 0;
              for (let c = 0; c < d; c++) dp += dA[t * d + c] * v[j * d + c];
              dPRow[j] = dp;
              const w = probs[t * T + j];
              rowDot += w * dp;
              if (w !== 0) {
                for (let c = 0; c < d; c++) dV[j * d + c] += w * dA[t * d + c];
              }
            }
            for (let j = 0; j <= t; j++) {
              const dS = probs[t * T + j] * (dPRow[j] - rowDot) * scale;
              if (dS === 0) continue;
              for (let c = 0; c < d; c++) {
                dQ[t * d + c] += dS * k[j * d + c];
                dK[j * d + c] += dS * q[t * d + c];
              }
            }
          }
        } else {
          // Registry mixer mixed the v-projection: dV = mixerᵀ(dM).
          this.mixers[l].backward(dA, T, d, dV);
        }

        // Projections: q/k/v = hIn·W  ⇒ gW += hInᵀ·dProj, dHin += dProj·Wᵀ
        for (let t = 0; t < T; t++) {
          for (let j = 0; j < d; j++) {
            const xtj = hIn[t * d + j];
            let acc = 0;
            const rq = off.wq + j * d, rk = off.wk + j * d, rv = off.wv + j * d;
            for (let c = 0; c < d; c++) {
              const dq = dQ[t * d + c], dk = dK[t * d + c], dv = dV[t * d + c];
              if (!frozen) {
                g[rq + c] += xtj * dq;
                g[rk + c] += xtj * dk;
                g[rv + c] += xtj * dv;
              }
              acc += dq * p[rq + c] + dk * p[rk + c] + dv * p[rv + c];
            }
            dHin[t * d + j] += acc;
          }
        }

        // LoRA backward — ONLY A/B receive gradient (base W stays frozen):
        //   dB += s·(X·A)ᵀ·dProj      d(XA) = s·dProj·Bᵀ
        //   dA += Xᵀ·d(XA)             dHin += d(XA)·Aᵀ
        if (this.lora && xaq && xav) {
          const { r, scale: s } = this.lora;
          const lo = this.lora.layers[l];
          const lp = this.loraParams;
          const lg = this.loraGrads;
          for (const [xa, dProj, aOff, bOff] of [
            [xaq, dQ, lo.aq, lo.bq],
            [xav, dV, lo.av, lo.bv],
          ] as Array<[Float32Array, Float32Array, number, number]>) {
            const dXA = new Float32Array(T * r);
            for (let t = 0; t < T; t++) {
              for (let c = 0; c < r; c++) {
                const xac = xa[t * r + c];
                let acc = 0;
                for (let j = 0; j < d; j++) {
                  const dpj = dProj[t * d + j];
                  lg[bOff + c * d + j] += s * xac * dpj;   // dB
                  acc += dpj * lp[bOff + c * d + j];
                }
                dXA[t * r + c] = s * acc;
              }
              for (let j = 0; j < d; j++) {
                const xtj = hIn[t * d + j];
                let acc = 0;
                for (let c = 0; c < r; c++) {
                  const dxac = dXA[t * r + c];
                  lg[aOff + j * r + c] += xtj * dxac;      // dA
                  acc += dxac * lp[aOff + j * r + c];
                }
                dHin[t * d + j] += acc;                    // through-adapter grad
              }
            }
          }
        }
        dH = dHin;
      }

      // Embedding + positional gradients from dH^0 (frozen under LoRA).
      if (!frozen) {
        for (let t = 0; t < T; t++) {
          const eOff = this.oE + window[t] * d;
          const pOff = this.oP + t * d;
          for (let j = 0; j < d; j++) {
            g[eOff + j] += dH[t * d + j];
            g[pOff + j] += dH[t * d + j];
          }
        }
      }
    }

    // ── Adam update ─────────────────────────────────────────────────────
    // Drain the device grad-E accumulator either way (keeps it clean); under
    // LoRA the result is discarded because the base is frozen.
    this.backend.flushGradE(g.subarray(this.oE, this.oE + V * d));
    const invB = 1 / batch.length;
    this.adamT++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.adamT);
    const c2 = 1 - Math.pow(b2, this.adamT);
    if (this.lora) {
      // LoRA: optimizer touches ONLY the adapters — base weights, E and every
      // per-layer device mirror stay bit-identical, so NO host→device weight
      // sync happens at all (the big Unsloth-style step-time win).
      const lp = this.loraParams, lg = this.loraGrads;
      for (let i = 0; i < lp.length; i++) {
        const gi = lg[i] * invB;
        this.loraM[i] = b1 * this.loraM[i] + (1 - b1) * gi;
        this.loraV[i] = b2 * this.loraV[i] + (1 - b2) * gi * gi;
        lp[i] -= (lr * (this.loraM[i] / c1)) / (Math.sqrt(this.loraV[i] / c2) + eps);
      }
    } else {
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * invB;
        this.m[i] = b1 * this.m[i] + (1 - b1) * gi;
        this.v[i] = b2 * this.v[i] + (1 - b2) * gi * gi;
        p[i] -= (lr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + eps);
      }
      // Weights changed — mirror E + per-layer matrices to the device.
      this.backend.syncE();
      this.syncLayerWeights();
    }

    return { loss: totalLoss / batch.length, tokensProcessed: tokens };
  }

  /** Loss-only evaluation (no gradient accumulation side effects kept). */
  evalLoss(batch: Uint16Array[]): number {
    const saved = this.params.slice();
    const savedM = this.m.slice();
    const savedV = this.v.slice();
    const savedLora = this.loraParams.slice();
    const savedLoraM = this.loraM.slice();
    const savedLoraV = this.loraV.slice();
    const savedT = this.adamT;
    const { loss } = this.step(batch, 0); // lr=0 ⇒ params unchanged by update
    this.params.set(saved);
    this.m.set(savedM);
    this.v.set(savedV);
    this.loraParams.set(savedLora);
    this.loraM.set(savedLoraM);
    this.loraV.set(savedLoraV);
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
    const { vocabSize: V, dModel: d, contextLength: ctx } = this.cfg;
    const all = Array.from(contextIds);
    const window = all.slice(Math.max(0, all.length - (ctx - 1)));
    const T = Math.max(1, window.length);
    const hFinal = this.forwardStack(window, T, null);
    const y = hFinal.subarray((T - 1) * d, T * d);
    const logits = new Float32Array(V);
    this.backend.logitsForward(y, logits);
    return logits;
  }
}
