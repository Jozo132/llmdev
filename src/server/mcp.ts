/**
 * MCP bridge — JSON-RPC 2.0 handler, local tool registry, AND a native
 * Model Context Protocol *client* that connects to real external MCP servers
 * over stdio child processes (filesystem tools, fetch utilities, git tools…).
 *
 * Wire format (stdio transport): newline-delimited JSON-RPC 2.0 frames on the
 * child's stdin/stdout, per the MCP stdio transport spec.
 *
 *   connectToMcpServer(name, cmd, args)
 *     └─ spawn → initialize → notifications/initialized → tools/list
 *        └─ discovered tools are injected into the backend registry under
 *           "<server>.<tool>" — instantly visible to the ChatSandbox tool
 *           pane and callable from model output <call_mcp_tool> blocks.
 *
 * tools/call responses (local or external) additionally carry a
 * `contextBlock` — the result wrapped in <mcp_tool_response>…</mcp_tool_response>
 * ready to be appended to the model's generation context window.
 */
import { execFile, spawn, type ChildProcess } from "node:child_process";
import type { LibraryManager } from "../core/LibraryManager.js";

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: number | string | null;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id?: number | string | null;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

export interface McpToolSchema {
  type: "object";
  properties: Record<string, unknown>;
  required?: string[];
}

interface ToolDef {
  name: string;
  description: string;
  inputSchema: McpToolSchema;
  /** Which external server owns this tool; undefined ⇒ built-in local tool. */
  server?: string;
  run(args: Record<string, unknown>): Promise<unknown>;
}

const REQUEST_TIMEOUT_MS = 15_000;
const PROTOCOL_VERSION = "2025-03-26";

// ── Stdio MCP client ─────────────────────────────────────────────────────────

/**
 * A live JSON-RPC 2.0 client bound to one external MCP server child process.
 * Requests are correlated by id; frames are newline-delimited JSON on stdout.
 */
export class StdioMcpClient {
  readonly name: string;
  private proc: ChildProcess | null = null;
  private nextId = 1;
  private buffer = "";
  private readonly pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: NodeJS.Timeout }
  >();
  private readonly log: (msg: string) => void;

  constructor(name: string, log: (msg: string) => void = console.log) {
    this.name = name;
    this.log = log;
  }

  get alive(): boolean {
    return this.proc !== null && this.proc.exitCode === null;
  }

  /** Spawn the server process and run the MCP initialization handshake. */
  async start(command: string, args: string[]): Promise<void> {
    this.proc = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"] });
    this.proc.stdout!.setEncoding("utf8");
    this.proc.stdout!.on("data", (chunk: string) => this.onData(chunk));
    this.proc.stderr!.setEncoding("utf8");
    this.proc.stderr!.on("data", (chunk: string) =>
      this.log(`[mcp:${this.name}] ${chunk.trim()}`)
    );
    this.proc.on("exit", (code) => {
      this.log(`[mcp:${this.name}] server exited (code ${code})`);
      for (const [, p] of this.pending) {
        clearTimeout(p.timer);
        p.reject(new Error(`MCP server "${this.name}" exited`));
      }
      this.pending.clear();
      this.proc = null;
    });
    // Fail fast when the executable is missing.
    await new Promise<void>((resolve, reject) => {
      this.proc!.once("spawn", () => resolve());
      this.proc!.once("error", (err) => reject(new Error(`spawn ${command}: ${err.message}`)));
    });

    // ── Initialization handshake ──
    await this.request("initialize", {
      protocolVersion: PROTOCOL_VERSION,
      clientInfo: { name: "llmdev", version: "0.1.0" },
      capabilities: {},
    });
    this.notify("notifications/initialized", {});
  }

  /** Send a request frame and await the correlated response result. */
  request(method: string, params: Record<string, unknown>): Promise<unknown> {
    if (!this.alive) return Promise.reject(new Error(`MCP server "${this.name}" is not running`));
    const id = this.nextId++;
    const frame = JSON.stringify({ jsonrpc: "2.0", id, method, params });
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP "${this.name}" ${method} timed out after ${REQUEST_TIMEOUT_MS / 1000}s`));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timer });
      this.proc!.stdin!.write(frame + "\n");
    });
  }

  /** Fire-and-forget notification (no id ⇒ no response expected). */
  notify(method: string, params: Record<string, unknown>): void {
    if (!this.alive) return;
    this.proc!.stdin!.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
  }

  close(): void {
    this.proc?.kill();
    this.proc = null;
  }

  /** Newline-delimited JSON framing: split stdout into complete frames. */
  private onData(chunk: string): void {
    this.buffer += chunk;
    let nl: number;
    while ((nl = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, nl).trim();
      this.buffer = this.buffer.slice(nl + 1);
      if (!line) continue;
      let msg: JsonRpcResponse;
      try {
        msg = JSON.parse(line) as JsonRpcResponse;
      } catch {
        this.log(`[mcp:${this.name}] unparseable frame: ${line.slice(0, 120)}`);
        continue;
      }
      if (typeof msg.id === "number" && this.pending.has(msg.id)) {
        const p = this.pending.get(msg.id)!;
        this.pending.delete(msg.id);
        clearTimeout(p.timer);
        if (msg.error) p.reject(new Error(`${msg.error.code}: ${msg.error.message}`));
        else p.resolve(msg.result);
      }
      // Server-initiated requests/notifications are ignored (logging-only client).
    }
  }
}

// ── Handler + registry ───────────────────────────────────────────────────────

const sh = (cmd: string, args: string[]): Promise<string> =>
  new Promise((resolve) => {
    execFile(cmd, args, { timeout: 5000 }, (err, stdout) => resolve(err ? "" : stdout.trim()));
  });

/** Wrap a tool result into the model-facing context block. */
const toContextBlock = (toolName: string, text: string): string =>
  `<mcp_tool_response tool="${toolName}">\n${text}\n</mcp_tool_response>`;

export interface McpHandler {
  (payload: unknown): Promise<unknown>;
  /** Spawn + handshake an external stdio MCP server, then sync its tools. */
  connectToMcpServer(name: string, command: string, args: string[]): Promise<number>;
  disconnectMcpServer(name: string): void;
  listServers(): Array<{ name: string; alive: boolean; tools: number }>;
}

export function createMcpHandler(library: LibraryManager, log: (msg: string) => void = console.log): McpHandler {
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

  const clients = new Map<string, StdioMcpClient>();

  /**
   * Dynamic tool syncing: pull the server's tools/list schema and inject each
   * tool into the shared registry as "<server>.<tool>" — dispatch maps the
   * JSON-RPC payload straight onto the live stdio child process.
   */
  async function syncServerTools(client: StdioMcpClient): Promise<number> {
    const listed = (await client.request("tools/list", {})) as {
      tools?: Array<{ name: string; description?: string; inputSchema?: McpToolSchema }>;
    };
    const discovered = listed?.tools ?? [];
    // Drop any previously-synced tools from this server (reconnect case).
    for (let i = tools.length - 1; i >= 0; i--) {
      if (tools[i].server === client.name) tools.splice(i, 1);
    }
    for (const t of discovered) {
      tools.push({
        name: `${client.name}.${t.name}`,
        description: t.description ?? `${t.name} (via MCP server "${client.name}")`,
        inputSchema: t.inputSchema ?? { type: "object", properties: {} },
        server: client.name,
        run: async (args) => client.request("tools/call", { name: t.name, arguments: args }),
      });
    }
    log(`[mcp:${client.name}] synced ${discovered.length} external tool(s) into the registry`);
    return discovered.length;
  }

  async function connectToMcpServer(name: string, command: string, args: string[]): Promise<number> {
    clients.get(name)?.close();
    const client = new StdioMcpClient(name, log);
    await client.start(command, args);
    clients.set(name, client);
    return syncServerTools(client);
  }

  function disconnectMcpServer(name: string): void {
    clients.get(name)?.close();
    clients.delete(name);
    for (let i = tools.length - 1; i >= 0; i--) {
      if (tools[i].server === name) tools.splice(i, 1);
    }
  }

  const handle = async function handle(payload: unknown): Promise<unknown> {
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
          protocolVersion: PROTOCOL_VERSION,
          serverInfo: { name: "llmdev-bridge", version: "0.1.0" },
          capabilities: { tools: {} },
        });
      case "tools/list":
        return reply({
          tools: tools.map(({ name, description, inputSchema, server }) => ({
            name, description, inputSchema, ...(server ? { server } : {}),
          })),
        });
      case "tools/call": {
        const name = req.params?.name as string;
        const tool = tools.find((t) => t.name === name);
        if (!tool) return fail(-32602, `Unknown tool: ${name}`);
        try {
          const result = await tool.run((req.params?.arguments as Record<string, unknown>) ?? {});
          // External servers already answer in MCP content shape; local tools
          // are wrapped. Both get a model-ready <mcp_tool_response> block.
          const content = (result as { content?: Array<{ type: string; text?: string }> })?.content
            ?? [{ type: "text", text: JSON.stringify(result, null, 2) }];
          const text = content.map((c) => c.text ?? "").join("\n");
          return reply({ content, contextBlock: toContextBlock(name, text) });
        } catch (err) {
          return fail(-32000, err instanceof Error ? err.message : String(err));
        }
      }
      // ── external server management (usable straight from the chat pane) ──
      case "servers/connect": {
        const { name, command, args } = (req.params ?? {}) as { name?: string; command?: string; args?: string[] };
        if (!name || !command) return fail(-32602, "servers/connect requires { name, command, args? }");
        try {
          const n = await connectToMcpServer(name, command, args ?? []);
          return reply({ connected: name, tools: n });
        } catch (err) {
          return fail(-32000, err instanceof Error ? err.message : String(err));
        }
      }
      case "servers/disconnect": {
        const name = req.params?.name as string;
        if (!name) return fail(-32602, "servers/disconnect requires { name }");
        disconnectMcpServer(name);
        return reply({ disconnected: name });
      }
      case "servers/list":
        return reply({
          servers: [...clients.entries()].map(([name, c]) => ({
            name, alive: c.alive, tools: tools.filter((t) => t.server === name).length,
          })),
        });
      default:
        return fail(-32601, `Method not found: ${req.method}`);
    }
  } as McpHandler;

  handle.connectToMcpServer = connectToMcpServer;
  handle.disconnectMcpServer = disconnectMcpServer;
  handle.listServers = () =>
    [...clients.entries()].map(([name, c]) => ({
      name, alive: c.alive, tools: tools.filter((t) => t.server === name).length,
    }));
  return handle;
}
