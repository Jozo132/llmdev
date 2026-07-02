/**
 * DatasetDB — centralized pre-tokenized dataset warehouse on better-sqlite3.
 *
 * VERTICAL SCALING: the connection runs in WAL (Write-Ahead Logging) mode so
 * readers (training loops streaming batches) NEVER block on writers (batched
 * shard commits) — GPU-bound tensor loops stay hot while ingestion appends.
 * Cache/mmap pragmas are tuned so hot shards are served from page cache with
 * zero syscall overhead.
 *
 * SCHEMA:
 *   datasets  — one row per tokenized corpus: identity key (sha256 of source
 *               + tokenizer params), vocab size, token count, and the
 *               decode_map (vocabulary id → string-byte mapping, i.e. the BPE
 *               merge table) needed to detokenize without any external file.
 *   shards    — ordered uint16 token arrays as binary BLOBs (2 bytes/token).
 *               Any model shape pulls the same pre-tokenized stream with zero
 *               reprocessing overhead; shard granularity keeps single rows
 *               small enough for incremental batched transactions.
 *
 *   projects     — visual IDE workspaces (id, name, updated_at).
 *   graph_nodes  — per-project canvas topology: node type, layout x/y, and
 *                  the full hyperparameter bag as a JSON blob.
 *   graph_edges  — per-project port-level wiring between nodes.
 *   Graph saves run as a single REPLACE transaction so a reload always sees
 *   a consistent snapshot of the last committed layout.
 */
import Database from "better-sqlite3";
import { mkdirSync } from "node:fs";
import path from "node:path";

export interface DatasetMeta {
  id: number;
  key: string;            // content-addressed identity (cache key)
  name: string;
  source: string;         // JSON: HF dataset/config/split/languages…
  vocabSize: number;
  tokenCount: number;
  createdAt: string;
}

export interface ProjectMeta {
  id: string;
  name: string;
  updatedAt: number;
}

export interface GraphNodeRow {
  id: string;
  type: string;
  x: number;
  y: number;
  params: Record<string, unknown>;
}

export interface GraphEdgeRow {
  id: string;
  from: { node: string; port: string };
  to: { node: string; port: string };
}

export interface ProjectGraph {
  project: ProjectMeta;
  nodes: GraphNodeRow[];
  edges: GraphEdgeRow[];
}

export interface ShardWriter {
  /** Append one uint16 token chunk (committed in batched transactions). */
  append(tokens: Uint16Array): void;
  /** Flush pending rows and finalize the dataset row. Returns total tokens. */
  commit(): number;
  /** Roll back everything written by this writer (e.g. on cancellation). */
  abort(): void;
}

const DEFAULT_BATCH_ROWS = 64; // shard rows per transaction

export class DatasetDB {
  private readonly db: Database.Database;

  constructor(artifactsDir: string, file = "warehouse.db") {
    mkdirSync(artifactsDir, { recursive: true });
    this.db = new Database(path.join(artifactsDir, file));
    // ── WAL + performance-tuned cache allocations ──────────────────────────
    this.db.pragma("journal_mode = WAL");        // readers never block writers
    this.db.pragma("synchronous = NORMAL");      // fsync at checkpoint, not per-commit
    this.db.pragma("cache_size = -65536");       // 64MB page cache
    this.db.pragma("mmap_size = 268435456");     // 256MB mmap window (zero-copy reads)
    this.db.pragma("temp_store = MEMORY");
    this.db.pragma("wal_autocheckpoint = 4096");

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS datasets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        key         TEXT NOT NULL UNIQUE,
        name        TEXT NOT NULL,
        source      TEXT NOT NULL DEFAULT '{}',
        vocab_size  INTEGER NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        decode_map  TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
      );
      CREATE TABLE IF NOT EXISTS shards (
        dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
        seq         INTEGER NOT NULL,
        n_tokens    INTEGER NOT NULL,
        tokens      BLOB NOT NULL,
        PRIMARY KEY (dataset_id, seq)
      ) WITHOUT ROWID;
      CREATE INDEX IF NOT EXISTS idx_datasets_key ON datasets(key);
      CREATE TABLE IF NOT EXISTS projects (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        updated_at  INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS graph_nodes (
        project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        node_id     TEXT NOT NULL,
        type        TEXT NOT NULL,
        x           REAL NOT NULL DEFAULT 0,
        y           REAL NOT NULL DEFAULT 0,
        properties  TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (project_id, node_id)
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS graph_edges (
        project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        edge_id         TEXT NOT NULL,
        source_node_id  TEXT NOT NULL,
        source_port     TEXT NOT NULL,
        target_node_id  TEXT NOT NULL,
        target_port     TEXT NOT NULL,
        PRIMARY KEY (project_id, edge_id)
      ) WITHOUT ROWID;
    `);
    this.db.pragma("foreign_keys = ON");
  }

  // ── Lookup ─────────────────────────────────────────────────────────────────

  findByKey(key: string): DatasetMeta | null {
    const row = this.db
      .prepare(
        `SELECT id, key, name, source, vocab_size AS vocabSize,
                token_count AS tokenCount, created_at AS createdAt
           FROM datasets WHERE key = ?`
      )
      .get(key) as DatasetMeta | undefined;
    return row ?? null;
  }

  list(): DatasetMeta[] {
    return this.db
      .prepare(
        `SELECT id, key, name, source, vocab_size AS vocabSize,
                token_count AS tokenCount, created_at AS createdAt
           FROM datasets ORDER BY id`
      )
      .all() as DatasetMeta[];
  }

  /** decode_map: vocabulary id → string bytes (BPE merge table JSON). */
  decodeMap(datasetId: number): unknown {
    const row = this.db
      .prepare(`SELECT decode_map AS m FROM datasets WHERE id = ?`)
      .get(datasetId) as { m: string } | undefined;
    if (!row) throw new Error(`No dataset ${datasetId}`);
    return JSON.parse(row.m);
  }

  /**
   * Materialize the full pre-tokenized stream as one contiguous Uint16Array.
   * WAL readers run lock-free, so this never stalls a concurrent writer.
   */
  loadTokens(datasetId: number): Uint16Array {
    const rows = this.db
      .prepare(`SELECT n_tokens, tokens FROM shards WHERE dataset_id = ? ORDER BY seq`)
      .all(datasetId) as Array<{ n_tokens: number; tokens: Buffer }>;
    const total = rows.reduce((s, r) => s + r.n_tokens, 0);
    const out = new Uint16Array(total);
    let off = 0;
    for (const r of rows) {
      out.set(new Uint16Array(r.tokens.buffer, r.tokens.byteOffset, r.n_tokens), off);
      off += r.n_tokens;
    }
    return out;
  }

  /** Random-access batch sampling without materializing the whole stream. */
  loadShard(datasetId: number, seq: number): Uint16Array | null {
    const row = this.db
      .prepare(`SELECT n_tokens, tokens FROM shards WHERE dataset_id = ? AND seq = ?`)
      .get(datasetId, seq) as { n_tokens: number; tokens: Buffer } | undefined;
    if (!row) return null;
    return new Uint16Array(row.tokens.buffer, row.tokens.byteOffset, row.n_tokens);
  }

  shardCount(datasetId: number): number {
    const r = this.db
      .prepare(`SELECT COUNT(*) AS n FROM shards WHERE dataset_id = ?`)
      .get(datasetId) as { n: number };
    return r.n;
  }

  // ── Ingestion ──────────────────────────────────────────────────────────────

  /**
   * Open a batched shard writer. Rows accumulate in memory and are committed
   * `batchRows` at a time inside a single transaction — the write pattern WAL
   * is optimized for. An existing dataset with the same key is replaced.
   */
  createWriter(
    meta: { key: string; name: string; source: Record<string, unknown>; vocabSize: number; decodeMap: unknown },
    batchRows = DEFAULT_BATCH_ROWS
  ): ShardWriter {
    const db = this.db;
    db.prepare(`DELETE FROM datasets WHERE key = ?`).run(meta.key);
    const { lastInsertRowid } = db
      .prepare(
        `INSERT INTO datasets (key, name, source, vocab_size, token_count, decode_map)
         VALUES (?, ?, ?, ?, 0, ?)`
      )
      .run(meta.key, meta.name, JSON.stringify(meta.source), meta.vocabSize, JSON.stringify(meta.decodeMap));
    const datasetId = Number(lastInsertRowid);

    const insert = db.prepare(
      `INSERT INTO shards (dataset_id, seq, n_tokens, tokens) VALUES (?, ?, ?, ?)`
    );
    const flushTx = db.transaction((rows: Array<{ seq: number; tokens: Uint16Array }>) => {
      for (const r of rows) {
        insert.run(datasetId, r.seq, r.tokens.length,
          Buffer.from(r.tokens.buffer, r.tokens.byteOffset, r.tokens.length * 2));
      }
    });

    let pending: Array<{ seq: number; tokens: Uint16Array }> = [];
    let seq = 0;
    let total = 0;
    let closed = false;

    return {
      append: (tokens: Uint16Array): void => {
        if (closed || tokens.length === 0) return;
        // Copy: the caller may reuse/transfer its buffer between appends.
        pending.push({ seq: seq++, tokens: tokens.slice() });
        total += tokens.length;
        if (pending.length >= batchRows) {
          flushTx(pending);
          pending = [];
        }
      },
      commit: (): number => {
        if (closed) return total;
        closed = true;
        if (pending.length) flushTx(pending);
        pending = [];
        db.prepare(`UPDATE datasets SET token_count = ? WHERE id = ?`).run(total, datasetId);
        return total;
      },
      abort: (): void => {
        if (closed) return;
        closed = true;
        pending = [];
        db.prepare(`DELETE FROM datasets WHERE id = ?`).run(datasetId);
      },
    };
  }

  delete(key: string): void {
    this.db.prepare(`DELETE FROM datasets WHERE key = ?`).run(key);
  }

  // ── Project / visual-graph persistence ────────────────────────────────────

  listProjects(): ProjectMeta[] {
    return this.db
      .prepare(`SELECT id, name, updated_at AS updatedAt FROM projects ORDER BY updated_at DESC`)
      .all() as ProjectMeta[];
  }

  /**
   * Atomically replace the full topology state (nodes + edges + layout) for a
   * project. One transaction ⇒ a concurrent reload never observes a
   * half-written graph.
   */
  saveGraph(projectId: string, name: string, nodes: GraphNodeRow[], edges: GraphEdgeRow[]): void {
    const db = this.db;
    const upsertProject = db.prepare(
      `INSERT INTO projects (id, name, updated_at) VALUES (?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at`
    );
    const insNode = db.prepare(
      `INSERT INTO graph_nodes (project_id, node_id, type, x, y, properties) VALUES (?, ?, ?, ?, ?, ?)`
    );
    const insEdge = db.prepare(
      `INSERT INTO graph_edges (project_id, edge_id, source_node_id, source_port, target_node_id, target_port)
       VALUES (?, ?, ?, ?, ?, ?)`
    );
    db.transaction(() => {
      upsertProject.run(projectId, name, Date.now());
      db.prepare(`DELETE FROM graph_nodes WHERE project_id = ?`).run(projectId);
      db.prepare(`DELETE FROM graph_edges WHERE project_id = ?`).run(projectId);
      for (const n of nodes) {
        insNode.run(projectId, n.id, n.type, n.x, n.y, JSON.stringify(n.params ?? {}));
      }
      for (const e of edges) {
        insEdge.run(projectId, e.id, e.from.node, e.from.port, e.to.node, e.to.port);
      }
    })();
  }

  loadGraph(projectId: string): ProjectGraph | null {
    const project = this.db
      .prepare(`SELECT id, name, updated_at AS updatedAt FROM projects WHERE id = ?`)
      .get(projectId) as ProjectMeta | undefined;
    if (!project) return null;
    const nodes = (this.db
      .prepare(`SELECT node_id AS id, type, x, y, properties FROM graph_nodes WHERE project_id = ?`)
      .all(projectId) as Array<{ id: string; type: string; x: number; y: number; properties: string }>)
      .map((r) => ({ id: r.id, type: r.type, x: r.x, y: r.y, params: JSON.parse(r.properties) as Record<string, unknown> }));
    const edges = (this.db
      .prepare(
        `SELECT edge_id AS id, source_node_id AS sn, source_port AS sp,
                target_node_id AS tn, target_port AS tp
           FROM graph_edges WHERE project_id = ?`
      )
      .all(projectId) as Array<{ id: string; sn: string; sp: string; tn: string; tp: string }>)
      .map((r) => ({ id: r.id, from: { node: r.sn, port: r.sp }, to: { node: r.tn, port: r.tp } }));
    return { project, nodes, edges };
  }

  renameProject(projectId: string, name: string): void {
    this.db.prepare(`UPDATE projects SET name = ?, updated_at = ? WHERE id = ?`).run(name, Date.now(), projectId);
  }

  deleteProject(projectId: string): void {
    this.db.prepare(`DELETE FROM projects WHERE id = ?`).run(projectId);
  }

  close(): void {
    this.db.pragma("wal_checkpoint(TRUNCATE)");
    this.db.close();
  }
}

// Shared per-process handle (one WAL connection serves all nodes).
let shared: DatasetDB | null = null;
export function getDatasetDB(artifactsDir: string): DatasetDB {
  if (!shared) shared = new DatasetDB(artifactsDir);
  return shared;
}
