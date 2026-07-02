/**
 * paramMath — mathematical parameter counter for transformer-family designs.
 *
 * Computes the total parameter count from architectural hyperparameters using
 * the standard closed-form formulas, with a per-component breakdown and the
 * formula text itself (surfaced as inline theory in the canvas badge).
 *
 * Components (d = dModel, V = vocab, L = layers, h = heads, kv = kvHeads,
 * f = hiddenDim):
 *   embedding      V·d                    (+ V·d output head unless tied)
 *   positional     ctx·d                  (learned absolute; RoPE would be 0)
 *   attention/layer d²·(2 + 2·kv/h)       (Wq,Wo full; Wk,Wv shrunk by GQA)
 *   mlp/layer      standard: 2·d·f        (up + down)
 *                  swiglu:   3·d·f        (gate + up + down)
 *   norms/layer    2·d (pre-attn + pre-mlp) + final d
 */
export interface ArchDesign {
  vocabSize: number;
  dModel: number;
  contextLength: number;
  hiddenDim: number;
  nLayers: number;
  nHeads: number;
  kvHeads: number;
  mlp: "standard" | "swiglu";
  tieEmbeddings: boolean;
}

export interface ParamBreakdown {
  embedding: number;
  positional: number;
  attentionPerLayer: number;
  mlpPerLayer: number;
  normsPerLayer: number;
  outputHead: number;
  total: number;
  formula: string[];
}

export function countParams(a: ArchDesign): ParamBreakdown {
  const { vocabSize: V, dModel: d, contextLength: ctx, hiddenDim: f } = a;
  const L = Math.max(1, a.nLayers);
  const h = Math.max(1, a.nHeads);
  const kv = Math.min(Math.max(1, a.kvHeads), h);

  const embedding = V * d;
  const positional = ctx * d;
  const attentionPerLayer = d * d * (2 + (2 * kv) / h);
  const mlpPerLayer = (a.mlp === "swiglu" ? 3 : 2) * d * f;
  const normsPerLayer = 2 * d;
  const outputHead = a.tieEmbeddings ? 0 : V * d;
  const total =
    embedding + positional + L * (attentionPerLayer + mlpPerLayer + normsPerLayer) + d + outputHead;

  return {
    embedding,
    positional,
    attentionPerLayer,
    mlpPerLayer,
    normsPerLayer,
    outputHead,
    total,
    formula: [
      `embedding  = V·d = ${V}·${d} = ${fmt(embedding)}`,
      `positional = ctx·d = ${ctx}·${d} = ${fmt(positional)}`,
      `attn/layer = d²·(2 + 2·kv/h) = ${d}²·(2 + 2·${kv}/${h}) = ${fmt(attentionPerLayer)}`,
      `mlp/layer  = ${a.mlp === "swiglu" ? "3" : "2"}·d·f = ${fmt(mlpPerLayer)}  (${a.mlp})`,
      `norms      = L·2d + d = ${fmt(L * normsPerLayer + d)}`,
      `head       = ${a.tieEmbeddings ? "0 (tied to embedding)" : `V·d = ${fmt(outputHead)}`}`,
      `TOTAL      = emb + pos + L·(attn + mlp + norms) + head = ${fmt(total)}`,
    ],
  };
}

const fmt = (n: number): string =>
  n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);

export { fmt as fmtParams };
