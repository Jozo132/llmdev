/**
 * MCP-style JSON-RPC 2.0 handler + local tool registry.
 * The Chat Sandbox parses tool-call payloads out of model output (or slash
 * commands) and dispatches them here over the WebSocket bridge.
 */
import { execFile } from "node:child_process";
import type { LibraryManager } from "../core/LibraryManager.js";

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: number | string | null;
  method: string;
  params?: Record<string, unknown>;
}

interface ToolDef {
  name: string;
  description: string;
  inputSchema: { type: "object"; properties: Record<string, unknown>; required?: string[] };
  run(args: Record<string, unknown>): Promise<unknown>;
}

const sh = (cmd: string, args: string[]): Promise<string> =>
  new Promise((resolve) => {
    execFile(cmd, args, { timeout: 5000 }, (err, stdout) => resolve(err ? "" : stdout.trim()));
  });

export function createMcpHandler(library: LibraryManager) {
  const tools: ToolDef[] = [
    {
      name: "get_time",
      description: "Current server time (ISO 8601).",
      inputSchema: { type: "object", properties: {} },
      run: async () => ({ now: new Date().toISOString() }),
    },
    {
      name: "disk_usage",
      description: "Artifact disk usage vs. the 100GB budget.",
      inputSchema: { type: "object", properties: {} },
      run: async () => ({
        artifacts: await sh("du", ["-sh", process.env.LLMDEV_ARTIFACTS_DIR ?? "artifacts"]),
        budgetGb: Number(process.env.LLMDEV_DISK_BUDGET_GB ?? 100),
      }),
    },
    {
      name: "list_models",
      description: "List all checkpoints/variants in the local model library.",
      inputSchema: { type: "object", properties: {} },
      run: async () =>
        library.list().map(({ id, name, paramCount, finalLoss, training }) => ({
          id, name, paramCount, finalLoss, training,
        })),
    },
  ];

  return async function handle(payload: unknown): Promise<unknown> {
    const req = payload as JsonRpcRequest;
    const reply = (result: unknown) => ({ jsonrpc: "2.0", id: req?.id ?? null, result });
    const fail = (code: number, message: string) => ({
      jsonrpc: "2.0", id: req?.id ?? null, error: { code, message },
    });

    if (!req || req.jsonrpc !== "2.0" || typeof req.method !== "string") {
      return fail(-32600, "Invalid JSON-RPC 2.0 request");
    }
    switch (req.method) {
      case "initialize":
        return reply({
          protocolVersion: "2025-03-26",
          serverInfo: { name: "llmdev-bridge", version: "0.1.0" },
          capabilities: { tools: {} },
        });
      case "tools/list":
        return reply({
          tools: tools.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })),
        });
      case "tools/call": {
        const name = req.params?.name as string;
        const tool = tools.find((t) => t.name === name);
        if (!tool) return fail(-32602, `Unknown tool: ${name}`);
        try {
          const result = await tool.run((req.params?.arguments as Record<string, unknown>) ?? {});
          return reply({ content: [{ type: "text", text: JSON.stringify(result, null, 2) }] });
        } catch (err) {
          return fail(-32000, err instanceof Error ? err.message : String(err));
        }
      }
      default:
        return fail(-32601, `Method not found: ${req.method}`);
    }
  };
}
