/**
 * ModelArchitectureNode — produces a ModelConfig for downstream trainers.
 * The `mixer` and `loss` params are names resolved through the src/ml/layers
 * registries, so alternative attention layers / custom loss functions
 * (JS, WASM, N-API C++, PyTorch bridges) are swappable from the UI or JSON
 * without touching the trainer.
 *
 * Default dims target ≈1.05M parameters:
 *   E: 8192×120 = 983,040   P: 64×120 = 7,680
 *   W1: 120×256 + 256 = 30,976   W2: 256×120 + 120 = 30,840   ⇒ 1,052,536
 */
import type {
  ModelConfig, NodeDescriptor, NodeParams, NodeRunContext, PipelineNode, TokenFileRef,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "model.architecture",
  label: "Model Architecture",
  category: "model",
  inputs: [{ name: "tokens", dataType: "token-file", required: true }],
  outputs: [
    { name: "config", dataType: "model-config" },
    { name: "tokens", dataType: "token-file" }, // pass-through for the trainer
  ],
  paramSchema: [
    { key: "dModel", label: "Embedding dim", type: "number", default: 120 },
    { key: "hiddenDim", label: "MLP hidden dim", type: "number", default: 256 },
    { key: "contextLength", label: "Context length", type: "number", default: 64 },
    { key: "mixer", label: "Attention/Mixer impl", type: "select",
      options: ["causal-mean"], default: "causal-mean",
      description: "Injectable — register custom kernels in src/ml/layers.ts" },
    { key: "loss", label: "Loss fn", type: "select",
      options: ["cross-entropy"], default: "cross-entropy" },
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
      dModel: number; hiddenDim: number; contextLength: number; mixer: string; loss: string;
    };
    const config: ModelConfig = {
      vocabSize: tokens.vocabSize,
      dModel: p.dModel,
      hiddenDim: p.hiddenDim,
      contextLength: p.contextLength,
      mixer: p.mixer,
      loss: p.loss,
    };
    const paramEstimate =
      config.vocabSize * config.dModel +
      config.contextLength * config.dModel +
      config.dModel * config.hiddenDim + config.hiddenDim +
      config.hiddenDim * config.dModel + config.dModel;
    ctx.log(`Architecture: ${JSON.stringify(config)} — ≈${(paramEstimate / 1e6).toFixed(2)}M params`);
    ctx.metric("param_count", paramEstimate);
    return { config, tokens };
  }
}

export const modelArchitectureDescriptor = DESCRIPTOR;
