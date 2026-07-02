/**
 * Exporters — serialize trained weights into standard interchange formats.
 *
 * GGUF v3 (little-endian) — the llama.cpp/Ollama/LM Studio container:
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ u32 magic "GGUF" · u32 version=3                         │
 *   │ u64 tensor_count · u64 metadata_kv_count                 │
 *   │ KV pairs: string key · u32 value-type · value            │
 *   │ tensor infos: name · u32 n_dims · u64 dims[] · u32 dtype │
 *   │               · u64 offset (relative to data section)    │
 *   │ padding → general.alignment (32)                         │
 *   │ tensor data (each tensor aligned to 32)                  │
 *   └──────────────────────────────────────────────────────────┘
 *
 * safetensors — the PyTorch-ecosystem format:
 *   u64 header_len · JSON header { name: {dtype, shape, data_offsets}, … }
 *   · raw tensor bytes.
 *
 * NOTE: external runners can parse these files (format-valid), but executing
 * them requires a runner that implements the "llmdev-tinylm" architecture —
 * general.architecture is set honestly so llama.cpp fails loudly, not weirdly.
 */
import { writeFile } from "node:fs/promises";
import type { ModelConfig } from "./types.js";

export interface NamedTensor {
  name: string;
  shape: number[];       // row-major
  data: Float32Array;
}

// ── GGUF v3 ───────────────────────────────────────────────────────────────────

const GGUF_MAGIC = 0x46554747; // "GGUF" LE
const GGUF_VERSION = 3;
const ALIGNMENT = 32;
// GGUF metadata value types
const T_U32 = 4, T_F32 = 6, T_STR = 8, T_U64 = 10;
const GGML_F32 = 0; // tensor dtype

class ByteWriter {
  private chunks: Buffer[] = [];
  private len = 0;

  get length(): number { return this.len; }
  push(b: Buffer): void { this.chunks.push(b); this.len += b.length; }
  u32(v: number): void { const b = Buffer.alloc(4); b.writeUInt32LE(v); this.push(b); }
  u64(v: number | bigint): void { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); this.push(b); }
  f32(v: number): void { const b = Buffer.alloc(4); b.writeFloatLE(v); this.push(b); }
  str(s: string): void { const bytes = Buffer.from(s, "utf8"); this.u64(bytes.length); this.push(bytes); }
  padTo(alignment: number): void {
    const rem = this.len % alignment;
    if (rem) this.push(Buffer.alloc(alignment - rem));
  }
  concat(): Buffer { return Buffer.concat(this.chunks); }
}

type MetaValue = { t: "u32"; v: number } | { t: "f32"; v: number } | { t: "str"; v: string } | { t: "u64"; v: number };

export async function writeGguf(
  path: string,
  config: ModelConfig,
  tensors: NamedTensor[],
  extraMeta: Record<string, MetaValue> = {}
): Promise<number> {
  const meta: Record<string, MetaValue> = {
    "general.architecture": { t: "str", v: "llmdev-tinylm" },
    "general.name": { t: "str", v: "llmdev PoC checkpoint" },
    "general.alignment": { t: "u32", v: ALIGNMENT },
    "general.file_type": { t: "u32", v: 0 }, // ALL_F32
    "llmdev-tinylm.vocab_size": { t: "u32", v: config.vocabSize },
    "llmdev-tinylm.embedding_length": { t: "u32", v: config.dModel },
    "llmdev-tinylm.context_length": { t: "u32", v: config.contextLength },
    "llmdev-tinylm.feed_forward_length": { t: "u32", v: config.hiddenDim },
    "llmdev-tinylm.mixer": { t: "str", v: config.mixer },
    "llmdev-tinylm.loss": { t: "str", v: config.loss },
    ...(config.nLayers ? { "llmdev-tinylm.block_count": { t: "u32" as const, v: config.nLayers } } : {}),
    ...(config.nHeads ? { "llmdev-tinylm.attention.head_count": { t: "u32" as const, v: config.nHeads } } : {}),
    ...(config.kvHeads ? { "llmdev-tinylm.attention.head_count_kv": { t: "u32" as const, v: config.kvHeads } } : {}),
    ...extraMeta,
  };

  const w = new ByteWriter();
  w.u32(GGUF_MAGIC);
  w.u32(GGUF_VERSION);
  w.u64(tensors.length);
  w.u64(Object.keys(meta).length);

  for (const [key, val] of Object.entries(meta)) {
    w.str(key);
    switch (val.t) {
      case "u32": w.u32(T_U32); w.u32(val.v); break;
      case "f32": w.u32(T_F32); w.f32(val.v); break;
      case "u64": w.u32(T_U64); w.u64(val.v); break;
      case "str": w.u32(T_STR); w.str(val.v); break;
    }
  }

  // Tensor infos — offsets are relative to the (aligned) data section start.
  let dataOffset = 0;
  for (const t of tensors) {
    w.str(t.name);
    w.u32(t.shape.length);
    // GGUF dims are stored fastest-varying first (reverse of row-major shape).
    for (const dim of [...t.shape].reverse()) w.u64(dim);
    w.u32(GGML_F32);
    w.u64(dataOffset);
    dataOffset += t.data.byteLength;
    dataOffset = Math.ceil(dataOffset / ALIGNMENT) * ALIGNMENT;
  }

  w.padTo(ALIGNMENT);
  for (const t of tensors) {
    w.push(Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength));
    w.padTo(ALIGNMENT);
  }

  const buf = w.concat();
  await writeFile(path, buf);
  return buf.length;
}

// ── safetensors ───────────────────────────────────────────────────────────────

export async function writeSafetensors(
  path: string,
  tensors: NamedTensor[],
  metadata: Record<string, string> = {}
): Promise<number> {
  const header: Record<string, unknown> = {
    __metadata__: { format: "llmdev-tinylm", ...metadata },
  };
  let offset = 0;
  for (const t of tensors) {
    header[t.name] = {
      dtype: "F32",
      shape: t.shape,
      data_offsets: [offset, offset + t.data.byteLength],
    };
    offset += t.data.byteLength;
  }
  let headerJson = JSON.stringify(header);
  // Pad header with spaces to 8-byte alignment (common practice, spec-legal).
  headerJson += " ".repeat((8 - ((8 + headerJson.length) % 8)) % 8);
  const headerBytes = Buffer.from(headerJson, "utf8");
  const lenBuf = Buffer.alloc(8);
  lenBuf.writeBigUInt64LE(BigInt(headerBytes.length));

  const buf = Buffer.concat([
    lenBuf,
    headerBytes,
    ...tensors.map((t) => Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength)),
  ]);
  await writeFile(path, buf);
  return buf.length;
}

/** Slice a flat weight buffer into named tensors per a layout table. */
export function sliceTensors(
  weights: Float32Array,
  layout: Array<{ name: string; offset: number; shape: number[] }>
): NamedTensor[] {
  return layout.map(({ name, offset, shape }) => {
    const size = shape.reduce((a, b) => a * b, 1);
    return { name, shape, data: weights.subarray(offset, offset + size) };
  });
}
