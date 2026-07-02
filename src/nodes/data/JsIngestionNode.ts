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
      theory: "Each language maps to a github-code-clean config " +
        "({Language}-{license}); documents stream with an equal per-language " +
        "quota so no single corpus dominates the token distribution.",
      range: "any of javascript,typescript,c,cpp,python,go,bash" },
    { key: "license", label: "License subset", type: "string", default: "mit",
      description: "mit | all — suffix for language configs" },
    { key: "config", label: "Config/Subset", type: "string", default: "JavaScript-mit",
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

/** Transient-failure policy: 5 attempts, 2s→4→8→16→32s exponential backoff. */
const MAX_RETRIES = 5;
const BACKOFF_BASE_MS = 2000;

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

/** Canonical github-code-clean language names for the scaling corpus set. */
const LANGUAGE_CONFIGS: Record<string, string> = {
  javascript: "JavaScript", js: "JavaScript",
  typescript: "TypeScript", ts: "TypeScript",
  c: "C",
  cpp: "C++", "c++": "C++",
  python: "Python", py: "Python",
  go: "Go",
  bash: "Shell", sh: "Shell", shell: "Shell",
};

function resolveConfigs(languages: string, license: string, fallback: string): string[] {
  const langs = languages.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
  if (!langs.length) return [fallback];
  return langs.map((l) => {
    const canonical = LANGUAGE_CONFIGS[l];
    if (!canonical) {
      throw new Error(`Unsupported language "${l}" — use one of ${Object.keys(LANGUAGE_CONFIGS).join(", ")}`);
    }
    return `${canonical}-${license}`;
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
          throw new Error(`HF streaming request failed after ${MAX_RETRIES} retries: ${(err as Error).message}`);
        }
        if (res.status === 429) {
          ctx.log("Rate-limited by HF — backing off 5s");
          await sleep(5000, ctx.signal);
          continue;
        }
        // Transient upstream errors (502/503/504 gateway blips) — retry with
        // exponential backoff instead of failing the whole pipeline.
        if (res.status >= 500) {
          if (retries < MAX_RETRIES) {
            const delay = BACKOFF_BASE_MS * 2 ** retries++;
            ctx.log(`HF rows API ${res.status} [${config}] (transient) — retry ${retries}/${MAX_RETRIES} in ${delay / 1000}s`);
            await sleep(delay, ctx.signal);
            continue;
          }
          throw new Error(
            `HF rows API ${res.status} [${config}] still failing after ${MAX_RETRIES} retries — ` +
            `the datasets-server is likely down, try again later.`
          );
        }
        if (!res.ok) {
          throw new Error(
            `HF rows API ${res.status} [${config}]: ${await res.text().then((t) => t.slice(0, 300))}` +
            ` — check dataset/config names or set HF_TOKEN for gated sets.`
          );
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
      ctx.log(`└ ${config}: ${configDocs} documents`);
      if (docs >= p.maxDocs || ctx.signal.aborted) break;
    }
    ctx.log(`Ingestion complete: ${docs} documents streamed across ${configs.length} corpus(es).`);
    ctx.metric("docs_streamed", docs);
  }
}

export const jsIngestionDescriptor = DESCRIPTOR;
