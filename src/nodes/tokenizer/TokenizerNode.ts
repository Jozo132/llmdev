/**
 * TokenizerNode — consumes the upstream text-stream, trains a byte-level BPE
 * on a bounded sample, then encodes the stream and packs tokens DIRECTLY into
 * a compact binary shard (.bin of uint16), flushing incrementally so memory
 * and disk stay tiny.  2 bytes/token ⇒ 10M tokens ≈ 20MB.
 */
import { createWriteStream, statSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { ByteBpeTokenizer } from "../../ml/tokenizer.js";
import type {
  NodeDescriptor, NodeParams, NodeRunContext, PipelineNode, TokenFileRef,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "tokenizer.byteBpe",
  label: "Byte-BPE Tokenizer",
  category: "tokenizer",
  theory:
    "Byte-Pair Encoding: start from raw bytes (256 symbols, zero OOV) and " +
    "greedily merge the most frequent adjacent pair until the vocab target. " +
    "Compression ratio directly multiplies effective context: 3.5 chars/token " +
    "means a 2048-token window sees ~7KB of code. Output is packed uint16 " +
    "(2 bytes/token) — the .bin shard IS the training set.",
  inputs: [{ name: "text", dataType: "text-stream", required: true }],
  outputs: [
    { name: "tokens", dataType: "token-file" },
    { name: "tokenizer", dataType: "tokenizer" },
  ],
  paramSchema: [
    { key: "vocabSize", label: "Vocab size", type: "number", default: 8192,
      theory: "Bigger vocab ⇒ better compression but V·d embedding params grow " +
        "linearly — at small d the embedding dominates the whole model. " +
        "uint16 packing caps V at 65535.",
      range: "4096–32768 for ≤10M-param models" },
    { key: "trainSampleChars", label: "BPE train sample (chars)", type: "number", default: 400_000,
      theory: "Merges are learned from this in-RAM sample only. Larger = more " +
        "representative merges but BPE training is O(vocab × sample).",
      range: "100K–2M chars" },
    { key: "maxTokens", label: "Token budget (shard cap)", type: "number", default: 2_000_000,
      description: "Hard cap on emitted tokens — 2M tokens = 4MB on disk",
      theory: "Disk-budget guardrail: tokens × 2 bytes on disk. Also bounds an " +
        "epoch: steps × batch × ctx tokens sampled per run.",
      range: "≤ 50M under the 100GB plan" },
    { key: "outFile", label: "Output shard", type: "string", default: "tokens/js-poc.bin" },
  ],
};

export class TokenizerNode implements PipelineNode {
  readonly descriptor = DESCRIPTOR;
  params: NodeParams;

  constructor(params: NodeParams = {}) {
    this.params = {
      ...Object.fromEntries(DESCRIPTOR.paramSchema.map((p) => [p.key, p.default])),
      ...params,
    };
  }

  async run(inputs: Record<string, unknown>, ctx: NodeRunContext) {
    const text = inputs.text as AsyncIterable<string>;
    if (!text) throw new Error("TokenizerNode requires a 'text' input stream");
    const p = this.params as {
      vocabSize: number; trainSampleChars: number; maxTokens: number; outFile: string;
    };

    // Phase 1 — buffer a bounded sample and the docs it came from (RAM only).
    ctx.log(`Collecting ≤${p.trainSampleChars} chars to train BPE (vocab ${p.vocabSize})…`);
    const bufferedDocs: string[] = [];
    let sample = "";
    const iterator = text[Symbol.asyncIterator]();
    while (sample.length < p.trainSampleChars) {
      const { value, done } = await iterator.next();
      if (done) break;
      bufferedDocs.push(value);
      sample += value.slice(0, Math.max(0, p.trainSampleChars - sample.length));
    }
    const tokenizer = new ByteBpeTokenizer();
    const t0 = performance.now();
    tokenizer.train(sample, p.vocabSize);
    ctx.log(`BPE trained: vocab=${tokenizer.vocabSize} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

    // Phase 2 — encode buffered docs + remaining stream straight to .bin.
    const outPath = path.join(ctx.artifactsDir, p.outFile);
    const ws = createWriteStream(outPath);
    let written = 0;

    const writeTokens = async (ids: Uint16Array): Promise<boolean> => {
      const room = p.maxTokens - written;
      if (room <= 0) return false;
      const slice = ids.length > room ? ids.subarray(0, room) : ids;
      // uint16 little-endian, written incrementally — never a giant buffer.
      const buf = Buffer.from(slice.buffer, slice.byteOffset, slice.length * 2);
      if (!ws.write(buf)) await new Promise<void>((r) => ws.once("drain", () => r()));
      written += slice.length;
      ctx.metric("tokens_written", written);
      return written < p.maxTokens;
    };

    const encodeDoc = async (doc: string): Promise<boolean> => {
      // Encode in 8KB chunks so BPE stays fast on long files.
      for (let i = 0; i < doc.length; i += 8192) {
        if (ctx.signal.aborted) return false;
        if (!(await writeTokens(tokenizer.encode(doc.slice(i, i + 8192))))) return false;
      }
      return true;
    };

    let keepGoing = true;
    for (const doc of bufferedDocs) {
      if (!keepGoing || ctx.signal.aborted) break;
      keepGoing = await encodeDoc(doc);
    }
    while (keepGoing && !ctx.signal.aborted) {
      const { value, done } = await iterator.next();
      if (done) break;
      keepGoing = await encodeDoc(value);
    }
    await new Promise<void>((resolve, reject) => ws.end((err?: Error) => (err ? reject(err) : resolve())));

    // Persist tokenizer next to the shard for later decode/export.
    await writeFile(outPath.replace(/\.bin$/, ".tokenizer.json"), JSON.stringify(tokenizer.toJSON()));

    const bytes = statSync(outPath).size;
    ctx.log(`Shard: ${outPath} — ${written} tokens, ${(bytes / 1024 / 1024).toFixed(2)}MB on disk`);

    const ref: TokenFileRef = { path: outPath, tokens: written, vocabSize: tokenizer.vocabSize };
    return { tokens: ref, tokenizer };
  }
}

export const tokenizerDescriptor = DESCRIPTOR;
