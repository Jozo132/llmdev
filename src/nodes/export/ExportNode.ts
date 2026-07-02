/**
 * ExportNode — serializes the trained model to a compact binary checkpoint:
 * a small JSON header + raw little-endian fp32 weights. ~4MB for 1M params,
 * trivially loadable from JS (Float32Array) or Python (np.fromfile).
 */
import { writeFile } from "node:fs/promises";
import { statSync } from "node:fs";
import path from "node:path";
import type {
  ExportedArtifact, NodeDescriptor, NodeParams, NodeRunContext, PipelineNode,
  TokenizerHandle, TrainedModelHandle,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "export.binary",
  label: "Export Checkpoint",
  category: "export",
  inputs: [
    { name: "model", dataType: "model", required: true },
    { name: "tokenizer", dataType: "tokenizer" },
  ],
  outputs: [{ name: "artifact", dataType: "artifact" }],
  paramSchema: [
    { key: "outFile", label: "Output file", type: "string", default: "exports/tinylm-poc" },
  ],
};

export class ExportNode implements PipelineNode {
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
    const tokenizer = inputs.tokenizer as TokenizerHandle | undefined;
    if (!handle) throw new Error("ExportNode requires 'model' input");
    const base = path.join(ctx.artifactsDir, String(this.params.outFile));

    const header = {
      format: "tinylm-v1",
      config: handle.config,
      paramCount: handle.paramCount,
      finalLoss: handle.finalLoss,
      stepsCompleted: handle.stepsCompleted,
      dtype: "float32-le",
      exportedAt: new Date().toISOString(),
    };
    await writeFile(`${base}.json`, JSON.stringify(header, null, 2));
    await writeFile(
      `${base}.weights.bin`,
      Buffer.from(handle.weights.buffer, handle.weights.byteOffset, handle.weights.byteLength)
    );
    // Persist the tokenizer next to the checkpoint so the Chat Sandbox and
    // LibraryManager can run standalone inference on this artifact.
    if (tokenizer) {
      await writeFile(`${base}.tokenizer.json`, JSON.stringify(tokenizer.toJSON()));
    }

    const bytes = statSync(`${base}.weights.bin`).size;
    ctx.log(`Exported ${base}.{json,weights.bin} — ${(bytes / 1024 / 1024).toFixed(2)}MB`);
    ctx.metric("export_bytes", bytes);

    const artifact: ExportedArtifact = { path: `${base}.weights.bin`, bytes, format: "tinylm-v1" };
    return { artifact };
  }
}

export const exportDescriptor = DESCRIPTOR;
