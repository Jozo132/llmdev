/**
 * ExportNode — serializes the trained model to a compact binary checkpoint:
 * a small JSON header + raw little-endian fp32 weights (4 bytes/param),
 * trivially loadable from JS (Float32Array) or Python (np.fromfile).
 */
import { writeFile } from "node:fs/promises";
import { statSync } from "node:fs";
import path from "node:path";
import { sliceTensors, writeGguf, writeSafetensors } from "../../core/Exporters.js";
import { TinyLM } from "../../ml/model.js";
import type {
  ExportedArtifact, NodeDescriptor, NodeParams, NodeRunContext, PipelineNode,
  TokenizerHandle, TrainedModelHandle,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "export.binary",
  label: "Export Checkpoint",
  category: "export",
  theory:
    "Serializes the trained flat weight buffer into interchange containers. " +
    "GGUF (llama.cpp/Ollama/LM Studio) is a self-describing binary: metadata " +
    "key-values + tensor directory + 32-byte-aligned tensor data — alignment " +
    "enables zero-copy mmap loading. safetensors is the PyTorch-ecosystem " +
    "equivalent: a JSON tensor directory with byte ranges, immune to pickle " +
    "code-execution vulnerabilities. Both store fp32 here; quantized GGUF " +
    "types (Q4_K, Q8_0) shrink files 4–8× at inference-quality cost.",
  inputs: [
    { name: "model", dataType: "model", required: true },
    { name: "tokenizer", dataType: "tokenizer" },
  ],
  outputs: [{ name: "artifact", dataType: "artifact" }],
  paramSchema: [
    { key: "outFile", label: "Output file", type: "string", default: "exports/checkpoint" },
    { key: "formats", label: "Formats", type: "string", default: "bin,gguf,safetensors",
      description: "Comma-separated: bin | gguf | safetensors",
      theory: "bin = raw fp32 dump (fastest reload in llmdev). gguf = external " +
        "runner container. safetensors = HuggingFace/PyTorch loading via " +
        "safetensors.torch.load_file().",
      range: "any subset of bin,gguf,safetensors" },
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
    const formats = String(this.params.formats ?? "bin")
      .split(",").map((f) => f.trim().toLowerCase()).filter(Boolean);

    const tensors = sliceTensors(handle.weights, TinyLM.tensorLayout(handle.config));
    let primary: ExportedArtifact | null = null;
    const emit = (p: string) => {
      const bytes = statSync(p).size;
      ctx.log(`exported ${p} — ${(bytes / 1024 / 1024).toFixed(2)}MB`);
      ctx.metric("export_bytes", bytes, { file: p });
      if (!primary) primary = { path: p, bytes, format: path.extname(p).slice(1) || "bin" };
    };

    if (formats.includes("bin")) {
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
      emit(`${base}.weights.bin`);
    }
    if (formats.includes("gguf")) {
      await writeGguf(`${base}.gguf`, handle.config, tensors, {
        "llmdev-tinylm.final_loss": { t: "f32", v: handle.finalLoss },
      });
      emit(`${base}.gguf`);
    }
    if (formats.includes("safetensors")) {
      await writeSafetensors(`${base}.safetensors`, tensors, {
        final_loss: String(handle.finalLoss),
        param_count: String(handle.paramCount),
      });
      emit(`${base}.safetensors`);
    }
    // Persist the tokenizer next to the checkpoint so the Chat Sandbox and
    // LibraryManager can run standalone inference on this artifact.
    if (tokenizer) {
      await writeFile(`${base}.tokenizer.json`, JSON.stringify(tokenizer.toJSON()));
    }

    if (!primary) throw new Error(`No valid formats in "${this.params.formats}"`);
    return { artifact: primary };
  }
}

export const exportDescriptor = DESCRIPTOR;
