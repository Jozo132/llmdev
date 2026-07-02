/**
 * MessageBroker — decouples the compute coordinator from the UI transport.
 *
 * HORIZONTAL SCALING SEAM: one HTTP server multiplexes three standardized
 * planes, so the coordinator process, the UI, and remote compute workers are
 * independently replaceable:
 *
 *   WS  /            UI clients — the full ClientMessage/ServerMessage protocol
 *   WS  /worker      remote worker machines — register with capabilities,
 *                    receive dispatched task envelopes, stream events back
 *   REST /api/*      stateless control plane (state, library, datasets,
 *                    POST /api/op for any protocol operation)
 *
 * The broker owns ZERO engine logic: every operation funnels through the
 * injected OpHandler, so the compute coordinator can live in this process
 * today and behind a socket tomorrow without touching transport code.
 */
import http, { type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { WebSocketServer, WebSocket } from "ws";
import type { ClientMessage, ServerMessage } from "./protocol.js";

export interface WorkerHandle {
  id: string;
  capabilities: Record<string, unknown>;
  connectedAt: string;
  ws: WebSocket;
}

export type OpHandler = (msg: ClientMessage, reply: (msg: ServerMessage) => void) => void;

export interface BrokerOptions {
  port: number;
  /** Compute-coordinator dispatch — shared by WS frames and REST POSTs. */
  onOp: OpHandler;
  /** Snapshot providers for GET /api/<name> routes. */
  restGet?: Record<string, () => unknown>;
  /** Called for each newly connected UI client to sync initial state. */
  onUiConnect?: (send: (msg: ServerMessage) => void) => void;
}

const MAX_BODY = 1 << 20; // 1MB POST cap

export class MessageBroker {
  private readonly uiClients = new Set<WebSocket>();
  private readonly workers = new Map<string, WorkerHandle>();
  private readonly server: http.Server;
  private readonly uiWss = new WebSocketServer({ noServer: true });
  private readonly workerWss = new WebSocketServer({ noServer: true });
  private readonly opts: BrokerOptions;

  constructor(opts: BrokerOptions) {
    this.opts = opts;
    this.server = http.createServer((req, res) => this.handleRest(req, res));

    // Route upgrades by path: / → UI plane, /worker → compute plane.
    this.server.on("upgrade", (req, socket, head) => {
      const path = (req.url ?? "/").split("?")[0];
      if (path === "/worker") {
        this.workerWss.handleUpgrade(req, socket, head, (ws) => this.acceptWorker(ws));
      } else {
        this.uiWss.handleUpgrade(req, socket, head, (ws) => this.acceptUi(ws));
      }
    });
  }

  listen(): void {
    this.server.listen(this.opts.port);
  }

  // ── UI plane ───────────────────────────────────────────────────────────────

  broadcast(msg: ServerMessage): void {
    const data = JSON.stringify(msg);
    for (const ws of this.uiClients) {
      if (ws.readyState === WebSocket.OPEN) ws.send(data);
    }
  }

  private acceptUi(ws: WebSocket): void {
    this.uiClients.add(ws);
    const send = (msg: ServerMessage) => ws.send(JSON.stringify(msg));
    this.opts.onUiConnect?.(send);
    ws.on("message", (raw) => {
      let msg: ClientMessage;
      try {
        msg = JSON.parse(String(raw)) as ClientMessage;
      } catch {
        return send({ ev: "error", message: "Invalid JSON" });
      }
      try {
        this.opts.onOp(msg, send);
      } catch (err) {
        send({ ev: "error", message: err instanceof Error ? err.message : String(err) });
      }
    });
    ws.on("close", () => this.uiClients.delete(ws));
  }

  // ── Worker plane (horizontal scaling) ──────────────────────────────────────

  listWorkers(): Array<Omit<WorkerHandle, "ws">> {
    return [...this.workers.values()].map(({ id, capabilities, connectedAt }) => ({
      id, capabilities, connectedAt,
    }));
  }

  /** Forward a task envelope to one registered remote worker. */
  dispatchToWorker(workerId: string, task: unknown): void {
    const w = this.workers.get(workerId);
    if (!w || w.ws.readyState !== WebSocket.OPEN) {
      throw new Error(`Worker ${workerId} is not connected`);
    }
    w.ws.send(JSON.stringify({ type: "task", task }));
  }

  private acceptWorker(ws: WebSocket): void {
    let handle: WorkerHandle | null = null;
    ws.on("message", (raw) => {
      let msg: { type: string; id?: string; capabilities?: Record<string, unknown>; payload?: unknown };
      try {
        msg = JSON.parse(String(raw));
      } catch {
        return;
      }
      if (msg.type === "register") {
        handle = {
          id: msg.id ?? randomUUID().slice(0, 8),
          capabilities: msg.capabilities ?? {},
          connectedAt: new Date().toISOString(),
          ws,
        };
        this.workers.set(handle.id, handle);
        ws.send(JSON.stringify({ type: "registered", id: handle.id }));
        this.broadcast({ ev: "workers", workers: this.listWorkers() });
      } else if (msg.type === "event" && handle) {
        // Mirror worker telemetry (metrics, task results) to every UI client.
        this.broadcast({ ev: "worker_event", workerId: handle.id, payload: msg.payload });
      }
    });
    ws.on("close", () => {
      if (handle) {
        this.workers.delete(handle.id);
        this.broadcast({ ev: "workers", workers: this.listWorkers() });
      }
    });
  }

  // ── REST control plane ─────────────────────────────────────────────────────

  private handleRest(req: IncomingMessage, res: ServerResponse): void {
    const path = (req.url ?? "/").split("?")[0];
    const json = (code: number, body: unknown) => {
      res.writeHead(code, { "Content-Type": "application/json" });
      res.end(JSON.stringify(body));
    };

    if (req.method === "GET") {
      if (path === "/api/health") {
        return json(200, { ok: true, workers: this.workers.size, uptimeSec: process.uptime() });
      }
      if (path === "/api/workers") return json(200, this.listWorkers());
      const name = path.startsWith("/api/") ? path.slice(5) : "";
      const provider = this.opts.restGet?.[name];
      if (provider) {
        try {
          return json(200, provider());
        } catch (err) {
          return json(500, { error: err instanceof Error ? err.message : String(err) });
        }
      }
      return json(404, { error: `Unknown route ${path}` });
    }

    if (req.method === "POST" && path === "/api/op") {
      let body = "";
      req.on("data", (c: Buffer) => {
        body += c;
        if (body.length > MAX_BODY) req.destroy();
      });
      req.on("end", () => {
        try {
          const msg = JSON.parse(body) as ClientMessage;
          const replies: ServerMessage[] = [];
          this.opts.onOp(msg, (m) => replies.push(m));
          json(200, { accepted: true, replies });
        } catch (err) {
          json(400, { error: err instanceof Error ? err.message : String(err) });
        }
      });
      return;
    }

    json(405, { error: "Method not allowed" });
  }
}
