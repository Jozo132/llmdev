/**
 * LibraryManager — evolutionary model version library.
 *
 * Discovers trained checkpoints (artifacts/exports/*.json) as base models and
 * manages derived variants under artifacts/library/<id>/:
 *
 *   Clone & Modify → a new variant directory holding a model.json with the
 *   overridden hyperparameters (e.g. swap the mixer for an experimental
 *   layer). When the parameter shapes are unchanged the weights are a
 *   SYMBOLIC LINK to the parent checkpoint (zero disk cost — 100GB budget);
 *   training then materializes real weights atomically (tmp file + rename).
 *
 * Variants can train concurrently (each run yields to the event loop), and
 * every step emits a "metric" event tagged with the variantId so the webapp
 * can cross-benchmark live sparklines.
 */
import { EventEmitter } from "node:events";
import {
  copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync,
  rmSync, symlinkSync, unlinkSync, writeFileSync, lstatSync, readlinkSync,
} from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { TinyLM } from "../ml/model.js";
import { ByteBpeTokenizer } from "../ml/tokenizer.js";
import type { ModelConfig } from "./types.js";

export interface BenchmarkPoint {
  step: number;
  loss: number;
  ts: number;
}

export interface ModelVariant {
  id: string;
  name: string;
  source: "export" | "clone";
  parentId?: string;
  config: ModelConfig;
  paramCount: number;
  weightsPath: string;
  tokenizerPath?: string;
  createdAt: string;
  finalLoss?: number;
  history: BenchmarkPoint[];
  training: boolean;
}

export interface VariantMetric {
  variantId: string;
  step: number;
  loss: number;
  tokensPerSec: number;
}

const paramCountOf = (c: ModelConfig): number => TinyLM.paramCountFor(c);

const slugify = (name: string): string =>
  name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

/** Every file an exported checkpoint may own on disk. */
const EXPORT_EXTS = [".json", ".weights.bin", ".tokenizer.json", ".gguf", ".safetensors"];

export class LibraryManager extends EventEmitter {
  private readonly exportsDir: string;
  private readonly libraryDir: string;
  private readonly tokensDir: string;
  private readonly trainingIds = new Set<string>();
  private readonly aborts = new Map<string, AbortController>();

  constructor(artifactsDir: string) {
    super();
    this.exportsDir = path.join(artifactsDir, "exports");
    this.libraryDir = path.join(artifactsDir, "library");
    this.tokensDir = path.join(artifactsDir, "tokens");
    mkdirSync(this.libraryDir, { recursive: true });
  }

  // ── Discovery ──────────────────────────────────────────────────────────────

  list(): ModelVariant[] {
    const variants: ModelVariant[] = [];

    // Base models: exported checkpoints (tinylm-v1 headers).
    if (existsSync(this.exportsDir)) {
      for (const f of readdirSync(this.exportsDir)) {
        if (!f.endsWith(".json") || f.endsWith(".tokenizer.json")) continue;
        try {
          const header = JSON.parse(readFileSync(path.join(this.exportsDir, f), "utf8"));
          if (header.format !== "tinylm-v1") continue;
          const base = f.replace(/\.json$/, "");
          const tokPath = path.join(this.exportsDir, `${base}.tokenizer.json`);
          variants.push({
            id: `export:${base}`,
            name: base,
            source: "export",
            config: header.config,
            paramCount: header.paramCount,
            weightsPath: path.join(this.exportsDir, `${base}.weights.bin`),
            tokenizerPath: existsSync(tokPath) ? tokPath : this.fallbackTokenizer(),
            createdAt: header.exportedAt ?? "",
            finalLoss: header.finalLoss,
            history: [],
            training: false,
          });
        } catch { /* skip unreadable headers */ }
      }
    }

    // Cloned variants.
    for (const dir of readdirSync(this.libraryDir)) {
      const metaPath = path.join(this.libraryDir, dir, "model.json");
      if (!existsSync(metaPath)) continue;
      try {
        const meta = JSON.parse(readFileSync(metaPath, "utf8")) as ModelVariant;
        meta.training = this.trainingIds.has(meta.id);
        meta.weightsPath = path.join(this.libraryDir, dir, "weights.bin");
        variants.push(meta);
      } catch { /* skip corrupt variants */ }
    }
    return variants;
  }

  private fallbackTokenizer(): string | undefined {
    if (!existsSync(this.tokensDir)) return undefined;
    const t = readdirSync(this.tokensDir).find((f) => f.endsWith(".tokenizer.json"));
    return t ? path.join(this.tokensDir, t) : undefined;
  }

  private defaultShard(): string {
    if (existsSync(this.tokensDir)) {
      const bin = readdirSync(this.tokensDir).find((f) => f.endsWith(".bin"));
      if (bin) return path.join(this.tokensDir, bin);
    }
    throw new Error("No token shard found — run the ingestion pipeline first");
  }

  // ── Create / Delete / Rename ─────────────────────────────────────────

  /** Create a blank variant from config — weights are random-init at first train. */
  create(name: string, overrides: Partial<ModelConfig> = {}): ModelVariant {
    const trimmed = name.trim();
    if (!trimmed) throw new Error("Model name is required");
    const config: ModelConfig = {
      vocabSize: 8192, dModel: 120, contextLength: 64, hiddenDim: 256,
      mixer: "causal-mean", loss: "cross-entropy", ...overrides,
    };
    const id = `${slugify(trimmed) || "model"}-${Date.now().toString(36).slice(-4)}`;
    const dir = path.join(this.libraryDir, id);
    mkdirSync(dir, { recursive: true });
    const variant: ModelVariant = {
      id,
      name: trimmed,
      source: "clone",
      config,
      paramCount: paramCountOf(config),
      weightsPath: path.join(dir, "weights.bin"),
      tokenizerPath: this.fallbackTokenizer(),
      createdAt: new Date().toISOString(),
      history: [],
      training: false,
    };
    writeFileSync(path.join(dir, "model.json"), JSON.stringify(variant, null, 2));
    this.emit("library", this.list());
    return variant;
  }

  /** Delete a variant AND its bytes on disk; materializes dependent symlinks first. */
  delete(id: string): void {
    if (this.trainingIds.has(id)) {
      throw new Error(`"${id}" is training — stop it before deleting`);
    }
    const all = this.list();
    const variant = all.find((v) => v.id === id);
    if (!variant) throw new Error(`Unknown variant: ${id}`);

    // Children may symlink this variant's weights (zero-copy clones) — copy
    // real bytes into them before the parent disappears.
    for (const child of all) {
      if (child.parentId !== id) continue;
      try {
        if (existsSync(child.weightsPath) && lstatSync(child.weightsPath).isSymbolicLink() &&
            existsSync(variant.weightsPath)) {
          unlinkSync(child.weightsPath);
          copyFileSync(variant.weightsPath, child.weightsPath);
        }
      } catch { /* best-effort materialization */ }
    }

    if (variant.source === "clone") {
      rmSync(path.join(this.libraryDir, id), { recursive: true, force: true });
    } else {
      const base = path.join(this.exportsDir, variant.name);
      for (const ext of EXPORT_EXTS) {
        if (existsSync(base + ext)) unlinkSync(base + ext);
      }
    }
    this.emit("library", this.list());
  }

  /** Rename a variant; export renames move every checkpoint file atomically. */
  rename(id: string, newName: string): string {
    if (this.trainingIds.has(id)) {
      throw new Error(`"${id}" is training — stop it before renaming`);
    }
    const name = newName.trim();
    if (!name) throw new Error("Model name is required");
    const variant = this.list().find((v) => v.id === id);
    if (!variant) throw new Error(`Unknown variant: ${id}`);

    let newId = id;
    if (variant.source === "clone") {
      const metaPath = path.join(this.libraryDir, id, "model.json");
      const meta = JSON.parse(readFileSync(metaPath, "utf8")) as ModelVariant;
      meta.name = name;
      writeFileSync(metaPath, JSON.stringify(meta, null, 2));
    } else {
      const slug = slugify(name) || "model";
      const oldBase = path.join(this.exportsDir, variant.name);
      const newBase = path.join(this.exportsDir, slug);
      if (slug !== variant.name && existsSync(`${newBase}.json`)) {
        throw new Error(`An export named "${slug}" already exists`);
      }
      for (const ext of EXPORT_EXTS) {
        if (existsSync(oldBase + ext)) renameSync(oldBase + ext, newBase + ext);
      }
      newId = `export:${slug}`;
      // Re-point clones: parentId, tokenizerPath, and weight symlinks.
      for (const dir of readdirSync(this.libraryDir)) {
        const metaPath = path.join(this.libraryDir, dir, "model.json");
        if (!existsSync(metaPath)) continue;
        try {
          const meta = JSON.parse(readFileSync(metaPath, "utf8")) as ModelVariant;
          let dirty = false;
          if (meta.parentId === id) { meta.parentId = newId; dirty = true; }
          if (meta.tokenizerPath === `${oldBase}.tokenizer.json`) {
            meta.tokenizerPath = `${newBase}.tokenizer.json`;
            dirty = true;
          }
          if (dirty) writeFileSync(metaPath, JSON.stringify(meta, null, 2));
          const wPath = path.join(this.libraryDir, dir, "weights.bin");
          if (existsSync(wPath) && lstatSync(wPath).isSymbolicLink() &&
              readlinkSync(wPath) === path.resolve(`${oldBase}.weights.bin`)) {
            unlinkSync(wPath);
            symlinkSync(path.resolve(`${newBase}.weights.bin`), wPath);
          }
        } catch { /* skip corrupt variants */ }
      }
    }
    this.emit("library", this.list());
    return newId;
  }

  // ── Clone & Modify ─────────────────────────────────────────────────────────

  clone(sourceId: string, name: string, overrides: Partial<ModelConfig> = {}): ModelVariant {
    const source = this.list().find((v) => v.id === sourceId);
    if (!source) throw new Error(`Unknown source variant: ${sourceId}`);

    const slug = slugify(name) || "variant";
    const id = `${slug}-${Date.now().toString(36).slice(-4)}`;
    const dir = path.join(this.libraryDir, id);
    mkdirSync(dir, { recursive: true });

    const config: ModelConfig = { ...source.config, ...overrides };
    const paramCount = paramCountOf(config);
    const weightsPath = path.join(dir, "weights.bin");

    // Shape-compatible clone ⇒ symlink parent weights (zero bytes on disk).
    // Shape change (new dims) ⇒ fresh random init at first training.
    if (paramCount === source.paramCount && existsSync(source.weightsPath)) {
      try {
        symlinkSync(path.resolve(source.weightsPath), weightsPath);
      } catch {
        copyFileSync(source.weightsPath, weightsPath); // atomic-copy fallback
      }
    }

    const variant: ModelVariant = {
      id,
      name,
      source: "clone",
      parentId: sourceId,
      config,
      paramCount,
      weightsPath,
      tokenizerPath: source.tokenizerPath,
      createdAt: new Date().toISOString(),
      finalLoss: source.finalLoss,
      history: [],
      training: false,
    };
    writeFileSync(path.join(dir, "model.json"), JSON.stringify(variant, null, 2));
    this.emit("library", this.list());
    return variant;
  }

  // ── Concurrent training ────────────────────────────────────────────────────

  isTraining(id: string): boolean {
    return this.trainingIds.has(id);
  }

  stopTraining(id: string): void {
    this.aborts.get(id)?.abort();
  }

  async train(variantId: string, opts: { steps?: number; batchSize?: number; lr?: number } = {}): Promise<void> {
    const variant = this.list().find((v) => v.id === variantId);
    if (!variant) throw new Error(`Unknown variant: ${variantId}`);
    if (variant.source === "export") throw new Error("Clone the base model first, then train the clone");
    if (this.trainingIds.has(variantId)) throw new Error(`${variantId} is already training`);

    const steps = opts.steps ?? 30;
    const batchSize = opts.batchSize ?? 4;
    const lr = opts.lr ?? 0.003;

    const abort = new AbortController();
    this.aborts.set(variantId, abort);
    this.trainingIds.add(variantId);
    this.emit("library", this.list());

    try {
      const raw = await readFile(this.defaultShard());
      const data = new Uint16Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 2));
      const windowLen = variant.config.contextLength;

      const model = new TinyLM(variant.config);
      // Warm-start from existing weights when shapes line up.
      if (existsSync(variant.weightsPath)) {
        const w = await readFile(variant.weightsPath);
        const f32 = new Float32Array(w.buffer, w.byteOffset, Math.floor(w.byteLength / 4));
        if (f32.length === model.paramCount) model.params.set(f32);
      }
      model.backend.syncE();

      let finalLoss = NaN;
      for (let step = 1; step <= steps; step++) {
        if (abort.signal.aborted) break;
        const batch = Array.from({ length: batchSize }, () => {
          const start = Math.floor(Math.random() * (data.length - windowLen));
          return data.subarray(start, start + windowLen);
        });
        const t0 = performance.now();
        const { loss, tokensProcessed } = model.step(batch, lr);
        finalLoss = loss;
        const metric: VariantMetric = {
          variantId,
          step,
          loss,
          tokensPerSec: tokensProcessed / ((performance.now() - t0) / 1000),
        };
        this.emit("metric", metric);
        variant.history.push({ step, loss, ts: Date.now() });
        await new Promise((r) => setImmediate(r)); // keep WS + other trainers live
      }

      // Materialize weights atomically: write tmp, then rename over the
      // (possibly symlinked) weights file — the parent stays untouched.
      const dir = path.dirname(variant.weightsPath);
      const tmp = path.join(dir, `.weights.tmp-${process.pid}`);
      await writeFile(tmp, Buffer.from(model.params.buffer, model.params.byteOffset, model.params.byteLength));
      if (existsSync(variant.weightsPath) && lstatSync(variant.weightsPath).isSymbolicLink()) {
        unlinkSync(variant.weightsPath);
      }
      renameSync(tmp, variant.weightsPath);

      variant.finalLoss = finalLoss;
      variant.history = variant.history.slice(-200); // bounded benchmark history
      writeFileSync(path.join(dir, "model.json"), JSON.stringify({ ...variant, training: false }, null, 2));
      model.backend.dispose(); // release device buffers promptly
    } finally {
      this.trainingIds.delete(variantId);
      this.aborts.delete(variantId);
      this.emit("library", this.list());
    }
  }

  // ── Chat inference support ─────────────────────────────────────────────────

  async loadForInference(variantId: string): Promise<{ model: TinyLM; tokenizer: ByteBpeTokenizer }> {
    const variant = this.list().find((v) => v.id === variantId);
    if (!variant) throw new Error(`Unknown variant: ${variantId}`);
    if (!existsSync(variant.weightsPath)) {
      throw new Error(`Variant "${variant.name}" has no trained weights yet`);
    }
    const model = new TinyLM(variant.config);
    const w = await readFile(variant.weightsPath);
    const f32 = new Float32Array(w.buffer, w.byteOffset, Math.floor(w.byteLength / 4));
    if (f32.length !== model.paramCount) {
      throw new Error(`Weight shape mismatch: ${f32.length} vs ${model.paramCount}`);
    }
    model.params.set(f32);
    model.backend.syncE();

    if (!variant.tokenizerPath || !existsSync(variant.tokenizerPath)) {
      throw new Error(`No tokenizer found for "${variant.name}"`);
    }
    const tokenizer = ByteBpeTokenizer.fromJSON(
      JSON.parse(await readFile(variant.tokenizerPath, "utf8"))
    );
    return { model, tokenizer };
  }
}
