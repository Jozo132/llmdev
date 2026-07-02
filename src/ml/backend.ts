/**
 * ComputeBackend — the seam between TinyLM and hardware.
 *
 * The tied output head (logits = E·y and its backward) dominates cost:
 * O(V·d) per token vs O(d·h) for the MLP. Both backends implement identical
 * semantics; "cuda" pins E and its gradient accumulator in GPU memory via the
 * N-API bridge (src/native), so per-token traffic is just y/logits vectors.
 *
 * createBackend() auto-selects: CUDA when the addon is built AND a GPU is
 * visible, otherwise the pure-JS CPU implementation. Force with
 * LLMDEV_BACKEND=js|cuda.
 */
import { createRequire } from "node:module";

export interface ComputeBackend {
  readonly name: string;
  /** Bind the (live) embedding matrix view [V×d]. Called once per model. */
  init(E: Float32Array, V: number, d: number): void;
  /** Push updated E to the device after an optimizer step (no-op on CPU). */
  syncE(): void;
  /** logits[i] = dot(E[i,:], y) */
  logitsForward(y: Float32Array, logits: Float32Array): void;
  /** dY[j] += Σ_i dLogits[i]·E[i,j]; internally accumulate gE[i,j] += dLogits[i]·y[j] */
  logitsBackward(y: Float32Array, dLogits: Float32Array, dY: Float32Array): void;
  /** Drain the internal grad-E accumulator into the host buffer (+=) and clear. */
  flushGradE(gE: Float32Array): void;
  dispose(): void;
}

// ── CPU reference implementation ─────────────────────────────────────────────

class JsBackend implements ComputeBackend {
  readonly name = "js-cpu";
  private E!: Float32Array;
  private gE!: Float32Array;
  private V = 0;
  private d = 0;

  init(E: Float32Array, V: number, d: number): void {
    this.E = E; // live view into the model's flat param buffer
    this.V = V;
    this.d = d;
    this.gE = new Float32Array(V * d);
  }

  syncE(): void {}

  logitsForward(y: Float32Array, logits: Float32Array): void {
    const { E, V, d } = this;
    for (let i = 0; i < V; i++) {
      let acc = 0;
      const off = i * d;
      for (let j = 0; j < d; j++) acc += y[j] * E[off + j];
      logits[i] = acc;
    }
  }

  logitsBackward(y: Float32Array, dLogits: Float32Array, dY: Float32Array): void {
    const { E, gE, V, d } = this;
    for (let i = 0; i < V; i++) {
      const dl = dLogits[i];
      if (dl === 0) continue;
      const off = i * d;
      for (let j = 0; j < d; j++) {
        dY[j] += dl * E[off + j];
        gE[off + j] += dl * y[j];
      }
    }
  }

  flushGradE(gE: Float32Array): void {
    for (let i = 0; i < this.gE.length; i++) gE[i] += this.gE[i];
    this.gE.fill(0);
  }

  dispose(): void {}
}

// ── CUDA backend via the N-API bridge ────────────────────────────────────────

interface NativeAddon {
  cudaAvailable(): boolean;
  deviceName(): string | null;
  createContext(E: Float32Array, V: number, d: number): unknown;
  releaseContext(ctx: unknown): void;
  syncE(ctx: unknown, E: Float32Array): void;
  logitsForward(ctx: unknown, y: Float32Array, logits: Float32Array): void;
  logitsBackward(ctx: unknown, y: Float32Array, dLogits: Float32Array, dY: Float32Array): void;
  flushGradE(ctx: unknown, gE: Float32Array): void;
  attnLastForward(x: Float32Array, T: number, d: number, out: Float32Array): boolean;
  sgemm(A: Float32Array, B: Float32Array, C: Float32Array, M: number, K: number, N: number): void;
}

const require_ = createRequire(import.meta.url);
let addon: NativeAddon | null = null;
try {
  // NOTE: resolved relative to src/ml/ — run via tsx (dev default).
  addon = require_("../native/index.cjs") as NativeAddon | null;
} catch {
  addon = null;
}

export function cudaAvailable(): boolean {
  return !!addon && addon.cudaAvailable();
}

export function cudaDeviceName(): string | null {
  return cudaAvailable() ? addon!.deviceName() : null;
}

/** Generic GPU SGEMM for custom nodes/layers; throws when CUDA is absent. */
export function gpuSgemm(
  A: Float32Array, B: Float32Array, C: Float32Array, M: number, K: number, N: number
): void {
  if (!cudaAvailable()) throw new Error("CUDA addon not available");
  addon!.sgemm(A, B, C, M, K, N);
}

/**
 * Flash-attention hook for SoftmaxAttentionMixer: tiled KV-sliced online
 * softmax on-device. Returns false (⇒ CPU streaming fallback) when the addon
 * is missing or the model is too wide for the kernel's shared-memory budget.
 */
export function attnLastForwardGpu(
  x: Float32Array, T: number, d: number, out: Float32Array
): boolean {
  if (!cudaAvailable()) return false;
  try {
    return addon!.attnLastForward(x, T, d, out);
  } catch {
    return false;
  }
}

class CudaBackend implements ComputeBackend {
  readonly name = "cuda";
  private ctx: unknown = null;
  private E!: Float32Array;

  init(E: Float32Array, V: number, d: number): void {
    this.E = E;
    this.ctx = addon!.createContext(E, V, d);
  }

  syncE(): void {
    addon!.syncE(this.ctx, this.E);
  }

  logitsForward(y: Float32Array, logits: Float32Array): void {
    addon!.logitsForward(this.ctx, y, logits);
  }

  logitsBackward(y: Float32Array, dLogits: Float32Array, dY: Float32Array): void {
    addon!.logitsBackward(this.ctx, y, dLogits, dY);
  }

  flushGradE(gE: Float32Array): void {
    addon!.flushGradE(this.ctx, gE);
  }

  dispose(): void {
    // Immediate device-memory release (cancel_training) — idempotent; the GC
    // finalizer reclaims the remaining host-side struct later.
    if (this.ctx) addon!.releaseContext(this.ctx);
    this.ctx = null;
  }
}

export function createBackend(prefer?: "js" | "cuda"): ComputeBackend {
  const want = prefer ?? (process.env.LLMDEV_BACKEND as "js" | "cuda" | undefined);
  if (want === "js") return new JsBackend();
  if (cudaAvailable()) return new CudaBackend();
  if (want === "cuda") {
    throw new Error("LLMDEV_BACKEND=cuda but the addon is not built / no GPU visible");
  }
  return new JsBackend();
}
