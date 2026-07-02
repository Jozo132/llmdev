/**
 * Client-side mirror of src/core/paramMath.ts — drives the live parameter
 * count badge on the canvas as the user edits architecture/tokenizer values.
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
      `embedding  = V·d = ${V}·${d} = ${fmtParams(embedding)}`,
      `positional = ctx·d = ${ctx}·${d} = ${fmtParams(positional)}`,
      `attn/layer = d²·(2 + 2·kv/h) = ${d}²·(2 + 2·${kv}/${h}) = ${fmtParams(attentionPerLayer)}`,
      `mlp/layer  = ${a.mlp === "swiglu" ? "3" : "2"}·d·f = ${fmtParams(mlpPerLayer)}  (${a.mlp})`,
      `norms      = L·2d + d = ${fmtParams(L * normsPerLayer + d)}`,
      `head       = ${a.tieEmbeddings ? "0 (tied)" : `V·d = ${fmtParams(outputHead)}`}`,
      `TOTAL      = emb + pos + L·(attn + mlp + norms) = ${fmtParams(total)}`,
    ],
  };
}

export function fmtParams(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);
}
