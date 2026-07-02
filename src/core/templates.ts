/**
 * Architectural Templates — verified, ready-to-train graph topologies.
 * Applying one replaces the engine's pipeline with a known-good 6-node chain
 * whose architecture params are pre-tuned to land near the stated size
 * (per src/core/paramMath.ts design formulas).
 */
import type { PipelineSpec } from "./types.js";
import { countParams } from "./paramMath.js";

export interface ArchTemplate {
  id: string;
  name: string;
  description: string;
  designParams: number; // formula-based design size (informational)
  spec: PipelineSpec;
}

function chain(
  name: string,
  tokenizerParams: Record<string, unknown>,
  archParams: Record<string, unknown>,
  trainParams: Record<string, unknown>
): PipelineSpec {
  return {
    name,
    nodes: [
      { id: "ingest", type: "data.jsIngestion", position: { x: 40, y: 220 },
        params: { dataset: "codeparrot/github-code-clean", config: "JavaScript-mit", maxDocs: 1000 } },
      { id: "tokenize", type: "tokenizer.byteBpe", position: { x: 300, y: 220 }, params: tokenizerParams },
      { id: "arch", type: "model.architecture", position: { x: 560, y: 220 }, params: archParams },
      { id: "train", type: "train.poc", position: { x: 820, y: 220 }, params: trainParams },
      { id: "eval", type: "eval.basic", position: { x: 1080, y: 220 }, params: {} },
      { id: "export", type: "export.binary", position: { x: 1340, y: 220 },
        params: { outFile: `exports/${name}` } },
    ],
    edges: [
      { from: { node: "ingest", port: "text" }, to: { node: "tokenize", port: "text" } },
      { from: { node: "tokenize", port: "tokens" }, to: { node: "arch", port: "tokens" } },
      { from: { node: "arch", port: "config" }, to: { node: "train", port: "config" } },
      { from: { node: "arch", port: "tokens" }, to: { node: "train", port: "tokens" } },
      { from: { node: "train", port: "model" }, to: { node: "eval", port: "model" } },
      { from: { node: "arch", port: "tokens" }, to: { node: "eval", port: "tokens" } },
      { from: { node: "tokenize", port: "tokenizer" }, to: { node: "eval", port: "tokenizer" } },
      { from: { node: "tokenize", port: "tokenizer" }, to: { node: "export", port: "tokenizer" } },
      { from: { node: "eval", port: "model" }, to: { node: "export", port: "model" } },
    ],
  };
}

const gptDesign = {
  vocabSize: 16384, dModel: 320, contextLength: 1024, hiddenDim: 1280,
  nLayers: 4, nHeads: 8, kvHeads: 8, mlp: "standard" as const, tieEmbeddings: true,
};
const mambaDesign = {
  vocabSize: 16384, dModel: 352, contextLength: 2048, hiddenDim: 1408,
  nLayers: 4, nHeads: 1, kvHeads: 1, mlp: "standard" as const, tieEmbeddings: true,
};
const hybridDesign = {
  vocabSize: 16384, dModel: 320, contextLength: 2048, hiddenDim: 1024,
  nLayers: 4, nHeads: 8, kvHeads: 2, mlp: "swiglu" as const, tieEmbeddings: true,
};

export const TEMPLATES: ArchTemplate[] = [
  {
    id: "gpt-dense-10m",
    name: "10M GPT-Style Dense Transformer",
    description:
      "Classic decoder-only stack: full multi-head softmax attention (h=kv=8), " +
      "standard 2-matrix MLP, learned positions, tied embeddings. The reference " +
      "architecture every scaling law is measured against.",
    designParams: countParams(gptDesign).total,
    spec: chain(
      "gpt-dense-10m",
      { vocabSize: gptDesign.vocabSize, maxTokens: 8_000_000, outFile: "tokens/gpt-10m.bin" },
      { dModel: gptDesign.dModel, hiddenDim: gptDesign.hiddenDim, contextLength: gptDesign.contextLength,
        nLayers: gptDesign.nLayers, nHeads: gptDesign.nHeads, kvHeads: gptDesign.kvHeads,
        mlp: gptDesign.mlp, mixer: "softmax-attn", loss: "cross-entropy" },
      { steps: 50, batchSize: 2, lr: 0.002 }
    ),
  },
  {
    id: "mamba-ssm-10m",
    name: "10M Mamba State-Space Model",
    description:
      "Attention-free: a recurrent state-space mixer carries context in O(T) time " +
      "and O(1) state instead of O(T²) attention. Mapped to the registered " +
      "'ssm-ema' mixer (exponential-moving-average selective-scan stand-in — " +
      "swap in a full Mamba kernel via registerMixer).",
    designParams: countParams(mambaDesign).total,
    spec: chain(
      "mamba-ssm-10m",
      { vocabSize: mambaDesign.vocabSize, maxTokens: 8_000_000, outFile: "tokens/mamba-10m.bin" },
      { dModel: mambaDesign.dModel, hiddenDim: mambaDesign.hiddenDim, contextLength: mambaDesign.contextLength,
        nLayers: mambaDesign.nLayers, nHeads: 1, kvHeads: 1,
        mlp: mambaDesign.mlp, mixer: "ssm-ema", loss: "cross-entropy" },
      { steps: 50, batchSize: 2, lr: 0.002 }
    ),
  },
  {
    id: "hybrid-gqa-swiglu-10m",
    name: "Hybrid GQA/SwiGLU Coder",
    description:
      "Modern-inference-optimized recipe: Grouped-Query Attention (8 query heads " +
      "share 2 KV heads ⇒ 4× smaller KV-cache) + SwiGLU gated MLP (3 matrices, " +
      "better loss per parameter). Long 2048-token context via streaming attention.",
    designParams: countParams(hybridDesign).total,
    spec: chain(
      "hybrid-gqa-swiglu-10m",
      { vocabSize: hybridDesign.vocabSize, maxTokens: 8_000_000, outFile: "tokens/hybrid-10m.bin" },
      { dModel: hybridDesign.dModel, hiddenDim: hybridDesign.hiddenDim, contextLength: hybridDesign.contextLength,
        nLayers: hybridDesign.nLayers, nHeads: hybridDesign.nHeads, kvHeads: hybridDesign.kvHeads,
        mlp: hybridDesign.mlp, mixer: "softmax-attn", loss: "cross-entropy" },
      { steps: 50, batchSize: 2, lr: 0.002 }
    ),
  },
];
