/**
 * TokenizerNode — consumes the upstream text-stream, trains a byte-level BPE
 * on a bounded sample, then encodes the stream and packs tokens DIRECTLY into
 * a compact binary shard (.bin of uint16), flushing incrementally so memory
 * and disk stay tiny.  2 bytes/token ⇒ 10M tokens ≈ 20MB.
 */
import { createHash } from "node:crypto";
import { createWriteStream, existsSync, statSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { ByteBpeTokenizer } from "../../ml/tokenizer.js";
import { trainBpeMerges } from "../../ml/bpeParallel.js";
import { getDatasetDB, type ShardWriter } from "../../core/DatasetDB.js";
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
    { key: "threads", label: "BPE worker threads", type: "number", default: 0,
      description: "0 = all available CPU cores",
      theory: "Pair-frequency counting and merge application are split into " +
        "equal chunks across a worker_threads pool; deltas stream back per " +
        "merge, so scaling is near-linear in cores.",
      range: "0–CPU count" },
    { key: "outFile", label: "Output shard", type: "string", default: "tokens/shard.bin" },
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
      vocabSize: number; trainSampleChars: number; maxTokens: number;
      threads: number; outFile: string;
    };

    const outPath = path.join(ctx.artifactsDir, p.outFile);
    const tokPath = outPath.replace(/\.bin$/, ".tokenizer.json");
    const cachePath = outPath.replace(/\.bin$/, ".cache.json");

    // ── Cache layer: hash the dataset identity + tokenization params. ──────
    // Upstream ingestion nodes attach a `sourceKey` to the text stream.
    const sourceKey = (text as { sourceKey?: string }).sourceKey ?? "unkeyed";
    const cacheKey = createHash("sha256").update(JSON.stringify({
      sourceKey, vocabSize: p.vocabSize,
      trainSampleChars: p.trainSampleChars, maxTokens: p.maxTokens,
    })).digest("hex");

    try {
      if (existsSync(cachePath) && existsSync(outPath) && existsSync(tokPath)) {
        const manifest = JSON.parse(await readFile(cachePath, "utf8")) as
          { key: string; tokens: number };
        if (manifest.key === cacheKey) {
          const cached = ByteBpeTokenizer.fromJSON(JSON.parse(await readFile(tokPath, "utf8")));
          ctx.metric("tokens_written", manifest.tokens);
          ctx.metric("node_progress", 1);
          ctx.log(`BPE cache hit (${cacheKey.slice(0, 12)}…) — reusing ` +
                  `${manifest.tokens} tokens from ${outPath}, skipping training`);
          const ref: TokenFileRef = {
            path: outPath, tokens: manifest.tokens, vocabSize: cached.vocabSize,
          };
          return { tokens: ref, tokenizer: cached };
        }
      }
    } catch { /* corrupt cache manifest → retrain */ }

    // ── SQLite warehouse hit: pre-tokenized shards shared across model shapes.
    // The decode_map rides in the datasets row, so the tokenizer rehydrates
    // with zero reprocessing and the .bin materializes straight from BLOBs.
    const db = getDatasetDB(ctx.artifactsDir);
    const stored = db.findByKey(cacheKey);
    if (stored) {
      const cached = ByteBpeTokenizer.fromJSON(
        db.decodeMap(stored.id) as { merges: Array<[number, number]> }
      );
      const tokens = db.loadTokens(stored.id);
      await writeFile(outPath, Buffer.from(tokens.buffer, tokens.byteOffset, tokens.byteLength));
      await writeFile(tokPath, JSON.stringify(cached.toJSON()));
      await writeFile(cachePath, JSON.stringify({
        key: cacheKey, tokens: tokens.length, vocabSize: cached.vocabSize,
        createdAt: new Date().toISOString(),
      }));
      ctx.metric("tokens_written", tokens.length);
      ctx.metric("node_progress", 1);
      ctx.log(`Warehouse hit (dataset #${stored.id}, ${cacheKey.slice(0, 12)}…) — ` +
              `materialized ${tokens.length} pre-tokenized tokens from SQLite`);
      const ref: TokenFileRef = { path: outPath, tokens: tokens.length, vocabSize: cached.vocabSize };
      return { tokens: ref, tokenizer: cached };
    }

    // Phase 1 — buffer a bounded sample and the docs it came from (RAM only).
    ctx.log(`Collecting ≤${p.trainSampleChars} chars to train BPE (vocab ${p.vocabSize})…`);
    ctx.metric("node_progress", 0.02);
    const bufferedDocs: string[] = [];
    let sample = "";
    const iterator = text[Symbol.asyncIterator]();
    while (sample.length < p.trainSampleChars) {
      if (ctx.signal.aborted) throw new Error("Cancelled while sampling");
      const { value, done } = await iterator.next();
      if (done) break;
      bufferedDocs.push(value);
      sample += value.slice(0, Math.max(0, p.trainSampleChars - sample.length));
    }

    // Multi-threaded merge training with live merge-count streaming.
    const t0 = performance.now();
    let lastEmit = 0;
    const merges = await trainBpeMerges(sample, p.vocabSize, {
      threads: p.threads > 0 ? p.threads : undefined,
      signal: ctx.signal,
      onMerge: (done, total) => {
        const now = performance.now();
        if (now - lastEmit > 100 || done === total) {
          lastEmit = now;
          ctx.metric("bpe_merges", done, { total });
          ctx.metric("node_progress", 0.05 + 0.55 * (done / total));
        }
      },
    });
    if (ctx.signal.aborted) throw new Error("Cancelled during BPE training — workers terminated");
    const tokenizer = ByteBpeTokenizer.fromJSON({ merges });
    ctx.log(`BPE trained: vocab=${tokenizer.vocabSize} in ${((performance.now() - t0) / 1000).toFixed(1)}s (parallel pair counting)`);

    // Phase 2 — encode buffered docs + remaining stream straight to .bin,
    // mirroring every chunk into the SQLite warehouse (batched transactions).
    const ws = createWriteStream(outPath);
    let written = 0;
    const shardWriter: ShardWriter = db.createWriter({
      key: cacheKey,
      name: path.basename(p.outFile, ".bin"),
      source: { sourceKey, vocabSize: p.vocabSize, maxTokens: p.maxTokens },
      vocabSize: tokenizer.vocabSize,
      decodeMap: tokenizer.toJSON(),
    });

    const writeTokens = async (ids: Uint16Array): Promise<boolean> => {
      const room = p.maxTokens - written;
      if (room <= 0) return false;
      const slice = ids.length > room ? ids.subarray(0, room) : ids;
      // uint16 little-endian, written incrementally — never a giant buffer.
      const buf = Buffer.from(slice.buffer, slice.byteOffset, slice.length * 2);
      if (!ws.write(buf)) await new Promise<void>((r) => ws.once("drain", () => r()));
      shardWriter.append(slice);
      written += slice.length;
      const now = performance.now();
      if (now - lastEmit > 100) {
        lastEmit = now;
        ctx.metric("tokens_written", written);
        ctx.metric("node_progress", 0.6 + 0.4 * Math.min(1, written / p.maxTokens));
      }
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
    await writeFile(tokPath, JSON.stringify(tokenizer.toJSON()));

    // Write the cache manifest + commit the warehouse rows ONLY on a complete
    // run — a cancelled run rolls the dataset back so no partial shards leak.
    if (!ctx.signal.aborted) {
      shardWriter.commit();
      await writeFile(cachePath, JSON.stringify({
        key: cacheKey, tokens: written, vocabSize: tokenizer.vocabSize,
        createdAt: new Date().toISOString(),
      }));
    } else {
      shardWriter.abort();
    }

    ctx.metric("tokens_written", written);
    ctx.metric("node_progress", 1);
    const bytes = statSync(outPath).size;
    ctx.log(`Shard: ${outPath} — ${written} tokens, ${(bytes / 1024 / 1024).toFixed(2)}MB on disk`);

    const ref: TokenFileRef = { path: outPath, tokens: written, vocabSize: tokenizer.vocabSize };
    return { tokens: ref, tokenizer };
  }
}

export const tokenizerDescriptor = DESCRIPTOR;
