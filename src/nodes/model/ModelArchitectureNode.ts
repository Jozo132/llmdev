/**
 * ModelArchitectureNode — produces a ModelConfig for downstream trainers.
 * The `mixer` and `loss` params resolve through the src/ml/layers registries;
 * design fields (nLayers/nHeads/kvHeads/mlp) drive the mathematical parameter
 * calculator (src/core/paramMath.ts) and are recorded in the config for
 * projection-based blocks. NOTE: the current PoC trainer executes a single
 * mixer+MLP block; the design fields describe the target architecture that
 * the calculator and exporters annotate.
 */
import { countParams, fmtParams } from "../../core/paramMath.js";
import type {
  ModelConfig, NodeDescriptor, NodeParams, NodeRunContext, PipelineNode, TokenFileRef,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "model.architecture",
  label: "Model Architecture",
  category: "model",
  theory:
    "Defines the transformer topology. A decoder block = sequence mixer " +
    "(attention/SSM) + position-wise MLP, each wrapped in residual + norm. " +
    "Standard attention computes softmax(QKᵀ/√d)V — O(T²) pairwise scores. " +
    "Grouped-Query Attention (kvHeads < nHeads) shares K/V projections across " +
    "query-head groups: quality ≈ MHA but the KV-cache shrinks by nHeads/kvHeads, " +
    "the dominant VRAM cost at inference. State-space mixers (Mamba-family) " +
    "replace attention with a linear recurrence: O(T) time, O(1) state. " +
    "Total parameters ≈ V·d + ctx·d + L·(d²(2+2kv/h) + {2|3}·d·f + 2d).",
  inputs: [{ name: "tokens", dataType: "token-file", required: true }],
  outputs: [
    { name: "config", dataType: "model-config" },
    { name: "tokens", dataType: "token-file" }, // pass-through for the trainer
  ],
  paramSchema: [
    { key: "dModel", label: "Embedding dim (d)", type: "number", default: 120,
      theory: "Width of every token's hidden vector. Parameters scale ~d² in " +
        "attention and ~d·f in the MLP; activation memory scales linearly. " +
        "Too small ⇒ representational bottleneck; too large ⇒ undertrained " +
        "dims at fixed data (Chinchilla: tokens ≈ 20× params).",
      range: "64–512 (toy) · 768–4096 (production)" },
    { key: "hiddenDim", label: "MLP hidden dim (f)", type: "number", default: 256,
      theory: "MLP expansion width. Convention f = 4d (standard) or ≈ 8d/3 " +
        "(SwiGLU, compensating its 3rd matrix). The MLP is where most " +
        "knowledge is stored — ~2/3 of non-embedding parameters.",
      range: "2d – 4d" },
    { key: "contextLength", label: "Context window (ctx)", type: "number", default: 64,
      theory: "Max tokens attended per step. Attention flops scale O(T²); the " +
        "softmax-attn mixer streams KV tiles with online softmax (flash-style) " +
        "so MEMORY stays O(T·d) — 2048–4096 fits the 5060 Ti. The positional " +
        "table adds ctx·d params. Longer ctx ⇒ more gradient signal per batch " +
        "but slower steps.",
      range: "32–4096 (streaming mixer) · ≤512 with causal-mean" },
    { key: "nLayers", label: "Layer count (L)", type: "number", default: 1,
      theory: "Blocks stacked in series. Depth builds hierarchical features; " +
        "params grow linearly, gradient path length too (residuals + norms " +
        "keep it trainable). For small models, width beats depth beyond ~8.",
      range: "1–12 for ≤100M-param models" },
    { key: "nHeads", label: "Attention heads (h)", type: "number", default: 1,
      theory: "Attention runs h parallel subspaces of size d/h. More heads = " +
        "more relation types per layer at constant parameter cost. d must be " +
        "divisible by h; head_dim = d/h below 32 starts hurting quality.",
      range: "d/h between 32 and 128" },
    { key: "kvHeads", label: "KV heads (GQA)", type: "number", default: 1,
      theory: "Grouped-Query Attention: kvHeads K/V projections shared by " +
        "nHeads queries. KV-cache VRAM = 2·L·T·d·(kv/h)·bytes — kv=h/4 cuts " +
        "the cache 4× with ≈0 quality loss (Llama-3 recipe). kv=1 is MQA.",
      range: "1 – nHeads (must divide nHeads)" },
    { key: "mlp", label: "MLP variant", type: "select",
      options: ["standard", "swiglu"], default: "standard",
      theory: "standard: down(act(up(x))) — 2 matrices, 2df params. swiglu: " +
        "down(silu(gate(x))⊙up(x)) — 3 matrices, 3df params; the learned gate " +
        "gives better loss per parameter (PaLM/Llama default).",
      range: "swiglu for new designs" },
    { key: "mixer", label: "Attention/Mixer impl", type: "select",
      options: ["causal-mean", "softmax-attn", "ssm-ema"], default: "causal-mean",
      theory: "The sequence-mixing slot. causal-mean: parameter-free uniform " +
        "past-average (baseline). softmax-attn: content-based scaled dot-" +
        "product with flash-style KV streaming + checkpointed backward. " +
        "ssm-ema: attention-free exponential-decay recurrence (Mamba-family " +
        "stand-in). Register custom kernels via registerMixer().",
      range: "softmax-attn for GPT-style · ssm-ema for linear-time long context" },
    { key: "loss", label: "Loss fn", type: "select",
      options: ["cross-entropy"], default: "cross-entropy",
      theory: "Cross-entropy = −log p(target). Its gradient at the logits is " +
        "softmax(z) − onehot(y) — bounded and well-scaled, which is why LM " +
        "training is stable. exp(loss) = perplexity.",
      range: "cross-entropy (add z-loss/MoE-aux via registerLoss)" },
    { key: "fineTuneMode", label: "Fine-Tuning Mode", type: "select",
      options: ["Full Parameter", "LoRA (Low-Rank Adaptation)"], default: "Full Parameter",
      theory: "Full Parameter trains every weight. LoRA freezes the base and " +
        "injects trainable low-rank adapters on the attention q/v projections: " +
        "W' = W + (α/r)·A·B with A∈R^{d×r} gaussian-init and B∈R^{r×d} " +
        "zero-init (initial delta = 0, so training starts from the exact base " +
        "model). Only A/B receive gradients — optimizer state shrinks from " +
        "2×params to 2×(4·L·d·r) and per-step host→device weight sync drops " +
        "to ZERO because the frozen base never changes on-device.",
      range: "LoRA for fine-tuning a warm-started checkpoint" },
    { key: "loraRank", label: "LoRA Rank (r)", type: "number", default: 8,
      theory: "Adapter bottleneck width. Trainable params scale linearly with " +
        "r; expressiveness saturates quickly — r=8–16 matches full fine-tune " +
        "quality on most adaptation tasks (LoRA paper, Tab. 6).",
      range: "4–64" },
    { key: "loraAlpha", label: "LoRA Alpha (α)", type: "number", default: 16,
      theory: "Adapter output scale α/r. Keeping α fixed while sweeping r " +
        "keeps the effective adapter learning rate constant; α = 2r is the " +
        "common heuristic.",
      range: "8–64 (typically 2× rank)" },
  ],
};

export class ModelArchitectureNode implements PipelineNode {
  readonly descriptor = DESCRIPTOR;
  params: NodeParams;

  constructor(params: NodeParams = {}) {
    this.params = {
      ...Object.fromEntries(DESCRIPTOR.paramSchema.map((p) => [p.key, p.default])),
      ...params,
    };
  }

  async run(inputs: Record<string, unknown>, ctx: NodeRunContext) {
    const tokens = inputs.tokens as TokenFileRef;
    if (!tokens) throw new Error("ModelArchitectureNode requires 'tokens' input");
    const p = this.params as {
      dModel: number; hiddenDim: number; contextLength: number;
      nLayers: number; nHeads: number; kvHeads: number;
      mlp: "standard" | "swiglu"; mixer: string; loss: string;
      fineTuneMode: string; loraRank: number; loraAlpha: number;
    };
    const loraOn = /lora/i.test(String(p.fineTuneMode ?? ""));
    const config: ModelConfig = {
      vocabSize: tokens.vocabSize,
      dModel: p.dModel,
      hiddenDim: p.hiddenDim,
      contextLength: p.contextLength,
      mixer: p.mixer,
      loss: p.loss,
      nLayers: p.nLayers,
      nHeads: p.nHeads,
      kvHeads: p.kvHeads,
      mlp: p.mlp,
      fineTuneMode: loraOn ? "lora" : "full",
      ...(loraOn ? { loraRank: p.loraRank, loraAlpha: p.loraAlpha } : {}),
    };

    // Design-formula parameter count (matches the live canvas calculator).
    const breakdown = countParams({
      vocabSize: config.vocabSize, dModel: config.dModel,
      contextLength: config.contextLength, hiddenDim: config.hiddenDim,
      nLayers: p.nLayers ?? 1, nHeads: p.nHeads ?? 1, kvHeads: p.kvHeads ?? 1,
      mlp: p.mlp ?? "standard", tieEmbeddings: true,
    });
    for (const line of breakdown.formula) ctx.log(line);
    ctx.metric("param_count", breakdown.total);
    if (loraOn) {
      ctx.log(`Fine-tuning mode: LoRA r=${p.loraRank}, α=${p.loraAlpha} — base frozen, ` +
              `adapters on attn q/v (${4 * (p.nLayers ?? 1) * p.dModel * p.loraRank} trainable params)`);
    }
    ctx.log(`Design total: ${fmtParams(breakdown.total)} params (${p.mixer}, L=${p.nLayers}, h=${p.nHeads}/kv=${p.kvHeads})`);
    return { config, tokens };
  }
}

export const modelArchitectureDescriptor = DESCRIPTOR;
