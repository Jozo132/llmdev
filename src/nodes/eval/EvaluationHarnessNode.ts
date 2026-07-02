/**
 * EvaluationHarnessNode — advanced model measurement across two benchmarks:
 *
 *  1. CODING NEXT-TOKEN PERPLEXITY — cross-entropy over a HELD-OUT tail slice
 *     of the token stream (never sampled by the trainer, which draws random
 *     windows from the head region). ppl = exp(mean CE); untrained baseline
 *     is the vocab size. Forward passes run through the ComputeBackend seam,
 *     so CPU and CUDA builds are measured identically.
 *
 *  2. MCP / TOOL-CALLING FUNCTIONAL ACCURACY — 100 deterministic structural
 *     diagnostic prompts (file listing, reads, searches, command execution…).
 *     Each generation is scored on two independent axes:
 *       · syntax  — did the model emit a clean <call_mcp_tool>…</call_mcp_tool>
 *                   block containing parseable JSON-RPC 2.0?
 *       · schema  — do method/params/arguments match the expected tool schema
 *                   (correct tool name, all required argument keys, correct
 *                   primitive types)?
 *     The functional score is the fraction of prompts passing BOTH axes.
 *
 * All results stream as eval_metrics into the analytics multi-series charts.
 */
import { readFile } from "node:fs/promises";
import { TinyLM } from "../../ml/model.js";
import type {
  NodeDescriptor, NodeParams, NodeRunContext, PipelineNode,
  TokenFileRef, TokenizerHandle, TrainedModelHandle,
} from "../../core/types.js";

const DESCRIPTOR: NodeDescriptor = {
  type: "eval.harness",
  label: "Evaluation Harness",
  category: "eval",
  theory:
    "Two-benchmark measurement harness. (1) Held-out next-token perplexity " +
    "on the verification tail slice the optimizer never saw — exp(mean " +
    "cross-entropy), the model's effective branching factor on unseen code. " +
    "(2) MCP tool-calling functional accuracy — 100 structural diagnostic " +
    "prompts scored deterministically for exact <call_mcp_tool> JSON-RPC " +
    "syntax extraction and argument schema validation. Both stream into the " +
    "analytics charts, so architecture variants are directly comparable.",
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
    { key: "holdoutFrac", label: "Held-out slice", type: "number", default: 0.1,
      theory: "Fraction of the token stream reserved as the verification " +
        "slice (the tail). Trainers sample the head, so these windows are " +
        "genuinely unseen.",
      range: "0.02–0.2" },
    { key: "evalBatches", label: "Eval batches", type: "number", default: 8,
      theory: "More batches ⇒ tighter loss estimate (stderr ∝ 1/√n).",
      range: "4–64" },
    { key: "batchSize", label: "Batch size", type: "number", default: 4 },
    { key: "mcpPrompts", label: "MCP diagnostic prompts", type: "number", default: 100,
      theory: "Structural tool-calling probes per run. 100 gives 1% score " +
        "resolution; the collection is deterministic so runs are comparable.",
      range: "10–100" },
    { key: "mcpGenTokens", label: "Gen tokens/prompt", type: "number", default: 96,
      range: "48–256" },
  ],
};

// ── MCP diagnostic prompt collection ─────────────────────────────────────────

interface McpDiagnostic {
  prompt: string;
  tool: string;
  /** Required argument key → expected typeof. */
  argSchema: Record<string, "string" | "number" | "boolean">;
}

/** Deterministic 100-prompt structural diagnostic collection. */
function buildMcpDiagnostics(count: number): McpDiagnostic[] {
  const dirs = ["src", "src/core", "src/ml", "webapp/src", "scripts", "artifacts", "pipelines", "src/nodes", "src/server", "docs"];
  const files = ["package.json", "README.md", "src/cli.ts", "tsconfig.json", "src/core/Engine.ts", "src/ml/model.ts", "webapp/src/App.vue", "pipelines/poc-js-1m.json", "src/server/index.ts", "scripts/serve.mjs"];
  const patterns = ["registerNode", "TODO", "vocabSize", "AbortSignal", "export class", "import type", "contextLength", "better-sqlite3", "WebSocket", "Float32Array"];
  const commands = ["npm test", "npm run build", "git status", "ls -la", "node --version", "npm run poc", "git log --oneline -5", "npx tsc --noEmit", "npm run serve", "df -h"];
  const templates: Array<(i: number) => McpDiagnostic> = [
    (i) => ({
      prompt: `List all files in the "${dirs[i % dirs.length]}" directory using the MCP list_files tool.`,
      tool: "list_files",
      argSchema: { path: "string" },
    }),
    (i) => ({
      prompt: `Read the contents of "${files[i % files.length]}" via the MCP read_file tool.`,
      tool: "read_file",
      argSchema: { path: "string" },
    }),
    (i) => ({
      prompt: `Search the workspace for "${patterns[i % patterns.length]}" with the MCP search tool.`,
      tool: "search",
      argSchema: { query: "string" },
    }),
    (i) => ({
      prompt: `Execute the shell command "${commands[i % commands.length]}" through the MCP run_command tool.`,
      tool: "run_command",
      argSchema: { command: "string" },
    }),
    (i) => ({
      prompt: `Fetch training status for model variant "variant-${i}" using the MCP model_status tool.`,
      tool: "model_status",
      argSchema: { variantId: "string" },
    }),
  ];
  return Array.from({ length: count }, (_, i) => templates[i % templates.length](i));
}

// ── Deterministic output scoring ─────────────────────────────────────────────

const CALL_BLOCK = /<call_mcp_tool>\s*([\s\S]*?)\s*<\/call_mcp_tool>/;

interface McpScore {
  syntaxOk: boolean; // clean block + parseable JSON-RPC 2.0 envelope
  schemaOk: boolean; // tool name + argument keys/types match expectations
}

export function scoreMcpOutput(output: string, expected: McpDiagnostic): McpScore {
  const match = CALL_BLOCK.exec(output);
  if (!match) return { syntaxOk: false, schemaOk: false };

  let rpc: unknown;
  try {
    rpc = JSON.parse(match[1]);
  } catch {
    return { syntaxOk: false, schemaOk: false };
  }
  const env = rpc as {
    jsonrpc?: unknown; method?: unknown;
    params?: { name?: unknown; arguments?: unknown };
  };
  const syntaxOk =
    env !== null && typeof env === "object" &&
    env.jsonrpc === "2.0" &&
    env.method === "tools/call" &&
    typeof env.params === "object" && env.params !== null;
  if (!syntaxOk) return { syntaxOk: false, schemaOk: false };

  const args = env.params!.arguments as Record<string, unknown> | undefined;
  const schemaOk =
    env.params!.name === expected.tool &&
    typeof args === "object" && args !== null &&
    Object.entries(expected.argSchema).every(([key, ty]) => typeof args[key] === ty);
  return { syntaxOk: true, schemaOk };
}

// ── Node ─────────────────────────────────────────────────────────────────────

export class EvaluationHarnessNode implements PipelineNode {
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
    if (!handle || !tokenRef) throw new Error("EvaluationHarnessNode requires 'model' and 'tokens'");
    const p = this.params as {
      holdoutFrac: number; evalBatches: number; batchSize: number;
      mcpPrompts: number; mcpGenTokens: number;
    };

    // Rehydrate the model around the trained flat weight buffer; forward math
    // runs through the ComputeBackend seam (js-cpu or CUDA transparently).
    const model = new TinyLM(handle.config);
    model.params.set(handle.weights);

    // ── Benchmark 1: coding next-token perplexity on the held-out tail ──────
    const raw = await readFile(tokenRef.path);
    const data = new Uint16Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 2));
    const windowLen = handle.config.contextLength;
    const holdoutStart = Math.max(0, Math.floor(data.length * (1 - p.holdoutFrac)));
    const holdout = data.subarray(holdoutStart);
    if (holdout.length < windowLen + 1) {
      throw new Error(`Held-out slice too small (${holdout.length} tokens) for context ${windowLen}`);
    }

    let totalLoss = 0;
    let batches = 0;
    for (let b = 0; b < p.evalBatches; b++) {
      if (ctx.signal.aborted) break;
      await ctx.waitIfPaused();
      const batch = Array.from({ length: p.batchSize }, () => {
        const start = Math.floor(Math.random() * (holdout.length - windowLen));
        return holdout.subarray(start, start + windowLen);
      });
      totalLoss += model.evalLoss(batch);
      batches++;
      ctx.metric("node_progress", (0.5 * (b + 1)) / p.evalBatches);
    }
    const loss = batches ? totalLoss / batches : NaN;
    const perplexity = Math.exp(loss);
    ctx.metric("eval_loss", loss);
    ctx.metric("eval_perplexity", perplexity);
    ctx.log(`held-out [${holdout.length} tok tail] loss=${loss.toFixed(4)} ppl=${perplexity.toFixed(1)}`);

    // ── Benchmark 2: MCP tool-calling functional accuracy ───────────────────
    let syntaxHits = 0;
    let schemaHits = 0;
    let functionalHits = 0;
    let probed = 0;

    if (tokenizer) {
      const diagnostics = buildMcpDiagnostics(Math.max(1, Math.min(100, p.mcpPrompts)));
      for (let i = 0; i < diagnostics.length; i++) {
        if (ctx.signal.aborted) break;
        await ctx.waitIfPaused();
        const diag = diagnostics[i];
        const promptIds = tokenizer.encode(diag.prompt).slice(-Math.max(1, windowLen - p.mcpGenTokens));
        const generated = model.generate(promptIds, p.mcpGenTokens); // greedy ⇒ deterministic
        const output = tokenizer.decode(generated.slice(promptIds.length));
        const score = scoreMcpOutput(output, diag);
        probed++;
        if (score.syntaxOk) syntaxHits++;
        if (score.schemaOk) schemaHits++;
        if (score.syntaxOk && score.schemaOk) functionalHits++;
        ctx.metric("node_progress", 0.5 + (0.5 * (i + 1)) / diagnostics.length);
        if ((i + 1) % 10 === 0) {
          // Stream rolling accuracy so the multi-series chart animates live.
          ctx.metric("tool_syntax_accuracy", (100 * syntaxHits) / probed);
          ctx.metric("tool_schema_accuracy", (100 * schemaHits) / probed);
          ctx.metric("tool_accuracy", (100 * functionalHits) / probed);
        }
      }
    } else {
      ctx.log("No tokenizer wired — skipping MCP tool-calling benchmark.");
    }

    const toolSyntaxPct = probed ? (100 * syntaxHits) / probed : 0;
    const toolSchemaPct = probed ? (100 * schemaHits) / probed : 0;
    const toolFunctionalPct = probed ? (100 * functionalHits) / probed : 0;
    ctx.metric("tool_syntax_accuracy", toolSyntaxPct);
    ctx.metric("tool_schema_accuracy", toolSchemaPct);
    ctx.metric("tool_accuracy", toolFunctionalPct);
    if (probed) {
      ctx.log(
        `MCP harness [${probed} prompts]: syntax=${toolSyntaxPct.toFixed(1)}% ` +
        `schema=${toolSchemaPct.toFixed(1)}% functional=${toolFunctionalPct.toFixed(1)}%`
      );
    }

    const report = {
      loss, perplexity,
      toolSyntaxPct, toolSchemaPct, toolFunctionalPct,
      promptsProbed: probed,
      sample: "",
    };
    return { report, model: handle };
  }
}

export const evaluationHarnessDescriptor = DESCRIPTOR;
