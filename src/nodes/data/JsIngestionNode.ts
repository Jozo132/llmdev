/**
 * JsIngestionNode — DataIngestionNode implementation targeting a small, clean
 * JavaScript subset on Hugging Face (default: codeparrot/github-code-clean,
 * MIT-licensed JavaScript config; bigcode/starcoderdata works too with HF_TOKEN).
 *
 * DISK POLICY: uses the Hugging Face *streaming* rows API
 * (datasets-server.huggingface.co) to page documents over HTTP on the fly.
 * NOTHING raw is ever written to disk — documents are yielded as an async
 * text stream that the downstream TokenizerNode consumes and packs straight
 * into a compact uint16 .bin, honoring the 100GB budget.
 */
import type { NodeDescriptor, NodeParams, NodeRunContext, PipelineNode } from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "data.jsIngestion",
  label: "Code Dataset Streamer (HF)",
  category: "data",
  theory:
    "Data quality upper-bounds model quality. This node streams documents " +
    "page-by-page from the Hugging Face rows API — backpressure-driven, so " +
    "nothing raw ever touches disk (100GB budget). Multi-language mode " +
    "interleaves JS/TS/C/C++/Python/Go/Bash corpuses with a per-language " +
    "quota. Dataset scale intuition: Chinchilla-optimal training wants ≈20 " +
    "tokens per parameter, so a 10M model wants ~200M tokens; smaller runs " +
    "simply undertrain.",
  inputs: [],
  outputs: [{ name: "text", dataType: "text-stream" }],
  paramSchema: [
    { key: "dataset", label: "HF Dataset", type: "string", default: "codeparrot/github-code-clean" },
    { key: "languages", label: "Languages (CSV)", type: "string", default: "",
      description: "e.g. javascript,typescript,c,cpp,python,go,bash — overrides Config/Subset",
      theory: "Each language maps to a github-code-clean config using the " +
        "dataset's exact case-sensitive identifier (e.g. \"JavaScript-mit\", " +
        "\"C++-mit\"); documents stream with an equal per-language quota so " +
        "no single corpus dominates the token distribution.",
      range: "any of javascript,typescript,c,cpp,python,go,bash" },
    { key: "license", label: "License subset", type: "string", default: "mit",
      description: "Reserved — github-code-clean configs are per-language only" },
    { key: "config", label: "Config/Subset", type: "string", default: "javascript",
      description: "Single-config fallback when Languages is empty" },
    { key: "split", label: "Split", type: "string", default: "train" },
    { key: "contentField", label: "Text field", type: "string", default: "code" },
    { key: "maxDocs", label: "Max documents", type: "number", default: 500,
      theory: "Hard cap on streamed documents — the disk/compute budget knob. " +
        "More docs ⇒ more unique tokens ⇒ less memorization per epoch.",
      range: "100–10000 at experimental scale" },
    { key: "maxDocChars", label: "Max chars/doc", type: "number", default: 20000,
      theory: "Truncates pathological files; keeps the token distribution from " +
        "being dominated by single giant bundles.",
      range: "5000–100000" },
    { key: "pageSize", label: "Rows per request", type: "number", default: 100,
      range: "1–100 (API hard limit)" },
  ],
};

const API = "https://datasets-server.huggingface.co/rows";
const HF_HUB = "https://huggingface.co";

/** Transient-failure policy: 5 attempts, 2s→4→8→16→32s exponential backoff. */
const MAX_RETRIES = 5;
const BACKOFF_BASE_MS = 2000;

/**
 * Raw-fallback extraction: printable text runs of at least this many chars are
 * treated as document candidates when regex-scanning raw parquet byte streams.
 */
const MIN_RAW_DOC_CHARS = 256;
const PRINTABLE_RUN = /[\t\n\r\u0020-\u007E]{256,}/g;

/** Abort-aware sleep — cancellation cuts the backoff wait short. */
const sleep = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve) => {
    const t = setTimeout(done, ms);
    function done() {
      clearTimeout(t);
      signal.removeEventListener("abort", done);
      resolve();
    }
    signal.addEventListener("abort", done, { once: true });
  });

/**
 * Canonical github-code-clean config identifiers — the dataset's parquet
 * export index is CASE-SENSITIVE and only publishes the exact
 * "{Language}-mit" identifiers below; lowercase forms 404/400 against both
 * the rows API and the parquet export index.
 */
const LANGUAGE_CONFIGS: Record<string, string> = {
  javascript: "JavaScript-mit", js: "JavaScript-mit",
  typescript: "TypeScript-mit", ts: "TypeScript-mit",
  c: "C-mit",
  cpp: "C++-mit", "c++": "C++-mit",
  python: "Python-mit", py: "Python-mit",
  go: "GO-mit",
  bash: "Shell-mit", sh: "Shell-mit", shell: "Shell-mit",
};

function resolveConfigs(languages: string, _license: string, fallback: string): string[] {
  const langs = languages.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
  if (!langs.length) {
    // Normalize legacy "{Language}-{license}" fallbacks to the true config id.
    const base = fallback.trim().toLowerCase().replace(/-(mit|all)$/, "");
    return [LANGUAGE_CONFIGS[base] ?? base];
  }
  return langs.map((l) => {
    const canonical = LANGUAGE_CONFIGS[l];
    if (!canonical) {
      throw new Error(`Unsupported language "${l}" — use one of ${Object.keys(LANGUAGE_CONFIGS).join(", ")}`);
    }
    return canonical;
  });
}

export class JsIngestionNode implements PipelineNode {
  readonly descriptor = DESCRIPTOR;
  params: NodeParams;

  constructor(params: NodeParams = {}) {
    this.params = {
      ...Object.fromEntries(DESCRIPTOR.paramSchema.map((p) => [p.key, p.default])),
      ...params,
    };
  }

  async run(_inputs: Record<string, unknown>, ctx: NodeRunContext) {
    const p = this.params as {
      dataset: string; languages: string; license: string; config: string;
      split: string; contentField: string;
      maxDocs: number; maxDocChars: number; pageSize: number;
    };
    const configs = resolveConfigs(p.languages, p.license, p.config);
    ctx.log(`Streaming ${p.dataset} [${configs.join(" + ")}/${p.split}] — no raw data hits disk.`);

    // Lazy async generator: HTTP pages are fetched only as the downstream
    // tokenizer pulls documents, so backpressure is free. The sourceKey
    // identifies the dataset slice for the downstream BPE cache layer.
    const stream = this.streamDocs(p, configs, ctx);
    const sourceKey = JSON.stringify({
      dataset: p.dataset, configs, split: p.split,
      contentField: p.contentField, maxDocs: p.maxDocs, maxDocChars: p.maxDocChars,
    });
    return { text: Object.assign(stream, { sourceKey }) };
  }

  private async *streamDocs(
    p: { dataset: string; split: string; contentField: string;
         maxDocs: number; maxDocChars: number; pageSize: number },
    configs: string[],
    ctx: NodeRunContext
  ): AsyncGenerator<string> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;

    // Equal per-language quota keeps the mixed corpus balanced.
    const perConfig = Math.ceil(p.maxDocs / configs.length);
    let docs = 0;

    for (const config of configs) {
      let offset = 0;
      let configDocs = 0;
      let retries = 0; // transient-error budget, reset on every healthy page
      let useFallback = false; // rows API gave up — switch to raw parquet streaming

      while (docs < p.maxDocs && configDocs < perConfig && !ctx.signal.aborted) {
        const length = Math.min(p.pageSize, 100); // API hard limit: 100 rows
        const url =
          `${API}?dataset=${encodeURIComponent(p.dataset)}` +
          `&config=${encodeURIComponent(config)}` +
          `&split=${encodeURIComponent(p.split)}` +
          `&offset=${offset}&length=${length}`;

        let res: Response;
        try {
          res = await fetch(url, { headers, signal: ctx.signal });
        } catch (err) {
          if (ctx.signal.aborted) return;
          // Network hiccup (reset, DNS, timeout) — retry with backoff.
          if (retries < MAX_RETRIES) {
            const delay = BACKOFF_BASE_MS * 2 ** retries++;
            ctx.log(`HF request failed (${(err as Error).message}) — retry ${retries}/${MAX_RETRIES} in ${delay / 1000}s`);
            await sleep(delay, ctx.signal);
            continue;
          }
          ctx.log(`HF rows API unreachable after ${MAX_RETRIES} retries — falling back to raw parquet streaming.`);
          useFallback = true;
          break;
        }
        if (res.status === 429) {
          ctx.log("Rate-limited by HF — backing off 5s");
          await sleep(5000, ctx.signal);
          continue;
        }
        // ANY other non-200 (404 cache miss, 500, 501 Not Implemented, …):
        // the datasets-server cache layer is unreliable — don't burn retries,
        // forcefully side-step it via the raw parquet export immediately.
        if (!res.ok) {
          ctx.log(`[ingest] Rows API returned ${res.status} for ${config}; forcefully switching to raw Parquet streaming fallback`);
          useFallback = true;
          break;
        }
        retries = 0; // healthy page — reset the backoff budget

        const body = (await res.json()) as { rows?: Array<{ row: Record<string, unknown> }> };
        const rows = body.rows ?? [];
        if (rows.length === 0) break; // end of split for this language

        for (const { row } of rows) {
          if (docs >= p.maxDocs || configDocs >= perConfig || ctx.signal.aborted) break;
          const raw = row[p.contentField];
          if (typeof raw !== "string" || raw.length < 64) continue; // skip junk
          docs++;
          configDocs++;
          if (docs % 100 === 0) ctx.metric("docs_streamed", docs);
          yield raw.slice(0, p.maxDocChars);
        }
        offset += rows.length;
      }

      // ── Raw parquet fallback: bypass the volatile datasets-server cache ──
      if (useFallback && !ctx.signal.aborted) {
        for await (const doc of this.streamParquetRaw(p.dataset, config, p.split, headers, ctx)) {
          if (docs >= p.maxDocs || configDocs >= perConfig || ctx.signal.aborted) break;
          docs++;
          configDocs++;
          if (docs % 100 === 0) ctx.metric("docs_streamed", docs);
          yield doc.slice(0, p.maxDocChars);
        }
      }

      ctx.log(`└ ${config}: ${configDocs} documents`);
      if (docs >= p.maxDocs || ctx.signal.aborted) break;
    }
    ctx.log(`Ingestion complete: ${docs} documents streamed across ${configs.length} corpus(es).`);
    ctx.metric("docs_streamed", docs);
  }

  /**
   * Raw streaming fallback — resolves the dataset's auto-converted parquet
   * index straight from the Hub API (`/api/datasets/<id>/parquet`), then
   * streams each parquet file over standard chunked HTTP. Documents are
   * recovered by regex-scanning decoded chunks for long printable text runs
   * (parquet plain-encoded string pages carry the payload verbatim), with a
   * carry-over tail so runs spanning chunk boundaries are never split. This
   * never touches the datasets-server cache layer and writes nothing to disk.
   */
  private async *streamParquetRaw(
    dataset: string,
    config: string,
    split: string,
    headers: Record<string, string>,
    ctx: NodeRunContext
  ): AsyncGenerator<string> {
    // Config identifiers are CASE-SENSITIVE ("JavaScript-mit", "C++-mit", …) —
    // never lowercase them. Split segments are lowercase by convention.
    const cfg = config.trim();
    const spl = split.trim().toLowerCase();
    // Per-config endpoint first: /api/datasets/<id>/parquet/<config>/<split>
    // returns a plain JSON array of shard URLs. The config segment must be
    // percent-encoded (e.g. "C++-mit" -> "C%2B%2B-mit") or the server drops
    // the request with a 400.
    let files: string[] = [];
    const scopedUrl = `${HF_HUB}/api/datasets/${dataset}/parquet/${encodeURIComponent(cfg)}/${encodeURIComponent(spl)}`;
    try {
      const scoped = await fetch(scopedUrl, { headers, signal: ctx.signal });
      if (scoped.ok) {
        const body = (await scoped.json()) as unknown;
        if (Array.isArray(body)) files = body.filter((u): u is string => typeof u === "string");
      } else {
        ctx.log(`[ingest] Scoped parquet endpoint ${scoped.status} for ${cfg}/${spl} — defaulting to root export index.`);
      }
    } catch (err) {
      if (ctx.signal.aborted) return;
      ctx.log(`[ingest] Scoped parquet endpoint failed (${(err as Error).message}) — defaulting to root export index.`);
    }
    if (!files.length) {
      // Root index fallback — flat array of entries, each carrying its own
      // case-sensitive `config` (and usually `split`) tag plus a download
      // `url`. Filter by exact config match rather than assuming a nested
      // object shape.
      const idxRes = await fetch(`${HF_HUB}/api/datasets/${dataset}/parquet`, { headers, signal: ctx.signal });
      if (!idxRes.ok) {
        throw new Error(`Parquet index ${idxRes.status} for ${dataset} — both rows API and raw fallback failed.`);
      }
      const index = (await idxRes.json()) as unknown;
      const entries = Array.isArray(index) ? (index as Record<string, unknown>[]) : [];
      // Normalized form ("javascript-mit") tolerates hyphenation/case drift
      // between the exact config id and whatever the manifest actually ships.
      const norm = (s: unknown) => String(s ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
      const cfgNorm = norm(cfg);
      const matches = (rec: Record<string, unknown>): boolean => {
        if (rec.config === cfg) return true;
        return norm(rec.config) === cfgNorm;
      };
      const matched = entries.filter((item): item is Record<string, unknown> => {
        if (!item || typeof item !== "object") return false;
        const rec = item as Record<string, unknown>;
        if (!matches(rec)) return false;
        return !("split" in rec) || rec.split === spl || norm(rec.split) === norm(spl);
      });
      if (!matched.length && entries.length) {
        ctx.log(`[ingest] No manifest entry matched config "${cfg}" — sample entry: ${JSON.stringify(entries[0])}`);
      }
      files = matched
        .map((item) => item.url)
        .filter((u): u is string => typeof u === "string");
    }
    if (!files.length) {
      ctx.log(`Parquet index has no files for ${cfg}/${spl} — skipping config.`);
      return;
    }
    ctx.log(`Raw fallback: streaming ${files.length} parquet shard(s) for ${cfg}/${spl}.`);

    const decoder = new TextDecoder("latin1");
    for (const fileUrl of files) {
      if (ctx.signal.aborted) return;
      let res: Response;
      try {
        res = await fetch(fileUrl, { headers, signal: ctx.signal });
      } catch (err) {
        if (ctx.signal.aborted) return;
        ctx.log(`Parquet shard fetch failed (${(err as Error).message}) — skipping shard.`);
        continue;
      }
      if (!res.ok || !res.body) {
        ctx.log(`Parquet shard ${res.status} — skipping shard.`);
        continue;
      }

      const reader = res.body.getReader();
      let carry = ""; // tail of the previous chunk — runs may span chunk edges
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done || ctx.signal.aborted) break;
          const text = carry + decoder.decode(value, { stream: true });
          // Hold any printable suffix — an in-progress run may continue in
          // the next chunk; only the prefix before it is safe to scan.
          const tail = /[\t\n\r\u0020-\u007E]+$/.exec(text);
          const holdFrom = tail ? tail.index : text.length;
          carry = text.slice(holdFrom);
          for (const m of text.slice(0, holdFrom).matchAll(PRINTABLE_RUN)) {
            yield m[0];
          }
          if (carry.length > 4_000_000) { // pathological run — flush + reset
            yield carry;
            carry = "";
          }
        }
      } finally {
        void reader.cancel().catch(() => {});
      }
      if (carry.length >= MIN_RAW_DOC_CHARS) yield carry;
    }
  }
}

export const jsIngestionDescriptor = DESCRIPTOR;
