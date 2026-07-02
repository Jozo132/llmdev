/**
 * EvaluationNode — held-out loss + perplexity on fresh random windows from the
 * shard, plus a greedy generation smoke test decoded through the tokenizer.
 */
import { readFile } from "node:fs/promises";
import { TinyLM } from "../../ml/model.js";
import type {
  EvalReport, NodeDescriptor, NodeParams, NodeRunContext, PipelineNode,
  TokenFileRef, TokenizerHandle, TrainedModelHandle,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "eval.basic",
  label: "Evaluation",
  category: "eval",
  theory:
    "Held-out loss on fresh random windows the optimizer never stepped on. " +
    "Perplexity = exp(loss) — the effective branching factor: ppl 500 means " +
    "the model is as uncertain as choosing among 500 equally-likely tokens. " +
    "Untrained baseline: ppl = vocab size. The greedy generation sample is a " +
    "smoke test for mode collapse (repeating one token).",
  inputs: [
    { name: "model", dataType: "model", required: true },
    { name: "tokens", dataType: "token-file", required: true },
    { name: "tokenizer", dataType: "tokenizer" },
  ],
  outputs: [
    { name: "report", dataType: "metrics" },
    { name: "model", dataType: "model" }, // pass-through for export
  ],
  paramSchema: [
    { key: "evalBatches", label: "Eval batches", type: "number", default: 4,
      theory: "More batches ⇒ tighter loss estimate (stderr ∝ 1/√n).",
      range: "4–32" },
    { key: "batchSize", label: "Batch size", type: "number", default: 4 },
    { key: "sampleTokens", label: "Sample gen tokens", type: "number", default: 24 },
  ],
};

export class EvaluationNode implements PipelineNode {
  readonly descriptor = DESCRIPTOR;
  params: NodeParams;

  constructor(params: NodeParams = {}) {
    this.params = {
      ...Object.fromEntries(DESCRIPTOR.paramSchema.map((p) => [p.key, p.default])),
      ...params,
    };
  }

  async run(inputs: Record<string, unknown>, ctx: NodeRunContext) {
    const handle = inputs.model as TrainedModelHandle;
    const tokenRef = inputs.tokens as TokenFileRef;
    const tokenizer = inputs.tokenizer as TokenizerHandle | undefined;
    if (!handle || !tokenRef) throw new Error("EvaluationNode requires 'model' and 'tokens'");
    const p = this.params as { evalBatches: number; batchSize: number; sampleTokens: number };

    // Rehydrate the model around the trained flat weight buffer.
    const model = new TinyLM(handle.config);
    model.params.set(handle.weights);

    const raw = await readFile(tokenRef.path);
    const data = new Uint16Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 2));
    const windowLen = handle.config.contextLength;

    let total = 0;
    for (let b = 0; b < p.evalBatches; b++) {
      if (ctx.signal.aborted) break;
      const batch = Array.from({ length: p.batchSize }, () => {
        const start = Math.floor(Math.random() * (data.length - windowLen));
        return data.subarray(start, start + windowLen);
      });
      total += model.evalLoss(batch);
    }
    const loss = total / p.evalBatches;
    const perplexity = Math.exp(loss);
    ctx.metric("eval_loss", loss);
    ctx.metric("perplexity", perplexity);

    // Generation smoke test.
    const promptStart = Math.floor(Math.random() * (data.length - 16));
    const prompt = data.subarray(promptStart, promptStart + 16);
    const generated = model.generate(prompt, p.sampleTokens);
    const sample = tokenizer
      ? tokenizer.decode(generated)
      : `[token ids] ${generated.join(" ")}`;
    ctx.log(`eval loss=${loss.toFixed(4)} ppl=${perplexity.toFixed(1)}`);
    ctx.log(`sample: ${sample.slice(0, 200).replace(/\n/g, "⏎")}`);

    const report: EvalReport = { loss, perplexity, sample };
    return { report, model: handle };
  }
}

export const evaluationDescriptor = DESCRIPTOR;
