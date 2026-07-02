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
  label: "JS Dataset Streamer (HF)",
  category: "data",
  inputs: [],
  outputs: [{ name: "text", dataType: "text-stream" }],
  paramSchema: [
    { key: "dataset", label: "HF Dataset", type: "string", default: "codeparrot/github-code-clean" },
    { key: "config", label: "Config/Subset", type: "string", default: "JavaScript-mit",
      description: "e.g. JavaScript-mit for github-code-clean, javascript for starcoderdata" },
    { key: "split", label: "Split", type: "string", default: "train" },
    { key: "contentField", label: "Text field", type: "string", default: "code" },
    { key: "maxDocs", label: "Max documents", type: "number", default: 500 },
    { key: "maxDocChars", label: "Max chars/doc", type: "number", default: 20000 },
    { key: "pageSize", label: "Rows per request", type: "number", default: 100 },
  ],
};

const API = "https://datasets-server.huggingface.co/rows";

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
      dataset: string; config: string; split: string; contentField: string;
      maxDocs: number; maxDocChars: number; pageSize: number;
    };
    ctx.log(`Streaming ${p.dataset} [${p.config}/${p.split}] — no raw data hits disk.`);

    // Lazy async generator: HTTP pages are fetched only as the downstream
    // tokenizer pulls documents, so backpressure is free.
    const stream = this.streamDocs(p, ctx);
    return { text: stream };
  }

  private async *streamDocs(
    p: { dataset: string; config: string; split: string; contentField: string;
         maxDocs: number; maxDocChars: number; pageSize: number },
    ctx: NodeRunContext
  ): AsyncGenerator<string> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;

    let offset = 0;
    let docs = 0;

    while (docs < p.maxDocs && !ctx.signal.aborted) {
      const length = Math.min(p.pageSize, 100); // API hard limit: 100 rows
      const url =
        `${API}?dataset=${encodeURIComponent(p.dataset)}` +
        `&config=${encodeURIComponent(p.config)}` +
        `&split=${encodeURIComponent(p.split)}` +
        `&offset=${offset}&length=${length}`;

      let res: Response;
      try {
        res = await fetch(url, { headers, signal: ctx.signal });
      } catch (err) {
        if (ctx.signal.aborted) return;
        throw new Error(`HF streaming request failed: ${(err as Error).message}`);
      }
      if (res.status === 429) {
        ctx.log("Rate-limited by HF — backing off 5s");
        await new Promise((r) => setTimeout(r, 5000));
        continue;
      }
      if (!res.ok) {
        throw new Error(
          `HF rows API ${res.status}: ${await res.text().then((t) => t.slice(0, 300))}` +
          ` — check dataset/config names or set HF_TOKEN for gated sets.`
        );
      }

      const body = (await res.json()) as { rows?: Array<{ row: Record<string, unknown> }> };
      const rows = body.rows ?? [];
      if (rows.length === 0) break; // end of split

      for (const { row } of rows) {
        if (docs >= p.maxDocs || ctx.signal.aborted) return;
        const raw = row[p.contentField];
        if (typeof raw !== "string" || raw.length < 64) continue; // skip junk
        docs++;
        if (docs % 100 === 0) ctx.metric("docs_streamed", docs);
        yield raw.slice(0, p.maxDocChars);
      }
      offset += rows.length;
    }
    ctx.log(`Ingestion complete: ${docs} documents streamed.`);
    ctx.metric("docs_streamed", docs);
  }
}

export const jsIngestionDescriptor = DESCRIPTOR;
