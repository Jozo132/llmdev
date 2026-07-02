/**
 * bpeParallel — multi-threaded byte-BPE merge training over worker_threads.
 *
 * Each worker owns one contiguous chunk of the byte stream. Rounds are
 * delta-based: after the initial full pair count, a merge broadcast makes
 * every worker rewrite its chunk in place and reply ONLY with the pair-count
 * deltas around the merge sites (classic incremental BPE), so per-round
 * message traffic is tiny. The coordinator aggregates deltas into a global
 * count map + lazy max-heap and picks the next best pair in O(log n).
 *
 * Cancellation: the AbortSignal is polled every round; on abort all workers
 * are terminated instantly (no draining).
 */
import os from "node:os";
import { Worker } from "node:worker_threads";

export interface BpeTrainOptions {
  /** Worker count; defaults to available cores (capped by sample size). */
  threads?: number;
  /** Cooperative cancellation — terminates all workers immediately. */
  signal?: AbortSignal;
  /** Called after every accepted merge with (done, total). */
  onMerge?: (done: number, total: number) => void;
}

/** Worker body (CJS, eval'd) — pure JS so it runs identically under tsx/node. */
const WORKER_SRC = `
const { parentPort, workerData } = require("node:worker_threads");
let ids = workerData; // Uint16Array chunk of token ids (< 65536)
let len = ids.length;

function pack(m) {
  const out = new Float64Array(m.size * 2);
  let j = 0;
  for (const [k, c] of m) { out[j++] = k; out[j++] = c; }
  return out;
}

parentPort.on("message", (msg) => {
  if (msg.op === "count") {
    const m = new Map();
    for (let i = 0; i < len - 1; i++) {
      const k = ids[i] * 65536 + ids[i + 1];
      m.set(k, (m.get(k) || 0) + 1);
    }
    const out = pack(m);
    parentPort.postMessage(out, [out.buffer]);
    return;
  }
  // op === "merge": rewrite (a,b) -> newId in place; reply with count deltas.
  const { a, b, newId } = msg;
  const delta = new Map();
  const bump = (x, y, d) => {
    if (x < 0 || y < 0) return;
    const k = x * 65536 + y;
    delta.set(k, (delta.get(k) || 0) + d);
  };
  let w = 0;
  for (let i = 0; i < len; i++) {
    if (i < len - 1 && ids[i] === a && ids[i + 1] === b) {
      const prev = w > 0 ? ids[w - 1] : -1;   // output-side neighbour
      const next = i + 2 < len ? ids[i + 2] : -1; // input-side neighbour
      bump(prev, a, -1); bump(prev, newId, +1);
      bump(a, b, -1);
      bump(b, next, -1); bump(newId, next, +1);
      ids[w++] = newId;
      i++;
    } else {
      ids[w++] = ids[i];
    }
  }
  len = w;
  const out = pack(delta);
  parentPort.postMessage(out, [out.buffer]);
});
`;

/** Lazy max-heap keyed by count; stale entries are discarded on peek. */
class MaxHeap {
  private c: number[] = []; // counts
  private k: number[] = []; // pair keys

  get size(): number { return this.c.length; }

  push(count: number, key: number): void {
    let i = this.c.push(count) - 1;
    this.k.push(key);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.c[p] >= this.c[i]) break;
      [this.c[p], this.c[i]] = [this.c[i], this.c[p]];
      [this.k[p], this.k[i]] = [this.k[i], this.k[p]];
      i = p;
    }
  }

  peek(): [number, number] { return [this.c[0], this.k[0]]; }

  pop(): void {
    const last = this.c.length - 1;
    this.c[0] = this.c[last]; this.k[0] = this.k[last];
    this.c.pop(); this.k.pop();
    let i = 0;
    for (;;) {
      const l = 2 * i + 1, r = l + 1;
      let m = i;
      if (l < this.c.length && this.c[l] > this.c[m]) m = l;
      if (r < this.c.length && this.c[r] > this.c[m]) m = r;
      if (m === i) break;
      [this.c[m], this.c[i]] = [this.c[i], this.c[m]];
      [this.k[m], this.k[i]] = [this.k[i], this.k[m]];
      i = m;
    }
  }
}

/**
 * Train BPE merges on `sample` across CPU threads.
 * Returns the ordered merge table ([left, right] producing id 256+i).
 */
export async function trainBpeMerges(
  sample: string,
  targetVocab: number,
  opts: BpeTrainOptions = {}
): Promise<Array<[number, number]>> {
  if (targetVocab > 65535) throw new Error("uint16 vocab cap is 65535");
  const bytes = new TextEncoder().encode(sample);
  const hw = os.availableParallelism?.() ?? os.cpus().length;
  const threads = Math.max(1, Math.min(opts.threads || hw, hw, Math.ceil(bytes.length / 65536)));

  const workers: Worker[] = [];
  const chunk = Math.ceil(bytes.length / threads);
  for (let t = 0; t < threads; t++) {
    const part = bytes.subarray(t * chunk, Math.min(bytes.length, (t + 1) * chunk));
    const idsChunk = new Uint16Array(part.length);
    idsChunk.set(part);
    workers.push(new Worker(WORKER_SRC, {
      eval: true, workerData: idsChunk, transferList: [idsChunk.buffer],
    }));
  }

  const ask = (w: Worker, msg: unknown): Promise<Float64Array> =>
    new Promise((resolve, reject) => {
      const onMsg = (m: Float64Array) => { cleanup(); resolve(m); };
      const onErr = (e: Error) => { cleanup(); reject(e); };
      const onExit = () => { cleanup(); reject(new Error("BPE worker terminated")); };
      const cleanup = () => {
        w.off("message", onMsg); w.off("error", onErr); w.off("exit", onExit);
      };
      w.once("message", onMsg); w.once("error", onErr); w.once("exit", onExit);
      w.postMessage(msg);
    });

  const merges: Array<[number, number]> = [];
  const counts = new Map<number, number>();
  const heap = new MaxHeap();
  const absorb = (packed: Float64Array): void => {
    for (let i = 0; i < packed.length; i += 2) {
      const key = packed[i];
      const c = (counts.get(key) ?? 0) + packed[i + 1];
      if (c > 0) { counts.set(key, c); heap.push(c, key); }
      else counts.delete(key);
    }
  };
  const total = Math.max(0, targetVocab - 256);

  try {
    (await Promise.all(workers.map((w) => ask(w, { op: "count" })))).forEach(absorb);
    while (merges.length < total) {
      if (opts.signal?.aborted) break;
      // Lazy heap top — discard entries whose count is stale.
      let bestKey = -1, bestCount = 0;
      while (heap.size) {
        const [c, k] = heap.peek();
        if (counts.get(k) === c) { bestKey = k; bestCount = c; break; }
        heap.pop();
      }
      if (bestKey < 0 || bestCount < 2) break; // merges must save ≥1 token
      const a = Math.floor(bestKey / 65536);
      const b = bestKey % 65536;
      const newId = 256 + merges.length;
      merges.push([a, b]);
      // Workers rewrite chunks + return deltas; (a,b) zeroes out naturally.
      (await Promise.all(workers.map((w) => ask(w, { op: "merge", a, b, newId })))).forEach(absorb);
      opts.onMerge?.(merges.length, total);
    }
  } finally {
    await Promise.all(workers.map((w) => w.terminate()));
  }
  return merges;
}
