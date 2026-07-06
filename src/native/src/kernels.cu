// kernels.cu — custom CUDA forward/backward kernels for llmdev.
// Compiled by nvcc via the binding.gyp .cu rule targeting Blackwell
// (compute_120/sm_120, RTX 5060 Ti) with a PTX fallback for JIT.
//
// Exposed through a flat extern "C" ABI so the N-API wrapper (addon.cc)
// stays free of CUDA headers.

#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),    \
              __FILE__, __LINE__);                                           \
    }                                                                        \
  } while (0)

// Kernel launches fail SILENTLY (e.g. cudaErrorNoKernelImageForDevice when
// the binary lacks SASS/PTX for the running GPU) unless explicitly checked —
// the subsequent cudaMemcpy would still succeed and return stale zeros,
// which starves the whole backward pass of gradient. Always check.
#define CUDA_CHECK_LAUNCH()                                                  \
  do {                                                                       \
    cudaError_t _e = cudaGetLastError();                                     \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA kernel launch error %s at %s:%d\n",              \
              cudaGetErrorString(_e), __FILE__, __LINE__);                   \
    }                                                                        \
  } while (0)

// Persistent GPU context for one model:
// the embedding matrix E [V×d] lives on-device across the whole training run,
// the E-gradient accumulates on-device, and each transformer layer owns a set
// of DISCRETE device weight buffers (allocated per layer index, never as one
// monolithic slab — depth changes cannot fragment or force a full realloc).
struct LlmCtx {
  float* dE;    // device E            [V*d]
  float* dGE;   // device grad-E accum [V*d]
  float* dY;    // device y / dY       [d]
  float* dLog;  // device logits/dLog  [V]
  int V;
  int d;
  bool freed;   // explicit-release guard (cancel_training) vs GC finalizer
  // ── per-layer transformer weights (arrays of discrete device pointers) ──
  int nLayers;                 // 0 until llm_ctx_alloc_buffers is called
  int hidden;                  // MLP hidden dim h
  float** w_attention_queries; // [nLayers] → device [d*d]
  float** w_attention_keys;    // [nLayers] → device [d*d]
  float** w_attention_values;  // [nLayers] → device [d*d]
  float** w_mlp_gate;          // [nLayers] → device [d*h]
  float** w_mlp_down;          // [nLayers] → device [h*d]
};

// ── Kernels ──────────────────────────────────────────────────────────────────

// logits[i] = dot(E[i,:], y) — one block per vocab row, warp-strided reduce.
__global__ void k_logits_forward(const float* __restrict__ E,
                                 const float* __restrict__ y,
                                 float* __restrict__ logits, int V, int d) {
  int row = blockIdx.x;
  if (row >= V) return;
  float acc = 0.f;
  for (int j = threadIdx.x; j < d; j += blockDim.x) acc += E[row * (long)d + j] * y[j];
  // block reduction
  __shared__ float sh[256];
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) logits[row] = sh[0];
}

// gE[i,j] += dLogits[i] * y[j]  (rank-1 update into the on-device accumulator)
__global__ void k_grad_e_acc(float* __restrict__ gE,
                             const float* __restrict__ dLogits,
                             const float* __restrict__ y, int V, int d) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = (long)V * d;
  for (; idx < total; idx += (long)gridDim.x * blockDim.x) {
    int i = (int)(idx / d);
    int j = (int)(idx % d);
    gE[idx] += dLogits[i] * y[j];
  }
}

// dY[j] = Σ_i dLogits[i] * E[i,j] — one thread per output dim, loop over V.
__global__ void k_dy(const float* __restrict__ E,
                     const float* __restrict__ dLogits,
                     float* __restrict__ dY, int V, int d) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  float acc = 0.f;
  for (int i = 0; i < V; i++) acc += dLogits[i] * E[i * (long)d + j];
  dY[j] = acc;
}

// Generic tiled SGEMM: C[M×N] = A[M×K] · B[K×N] — building block for the 10M
// architecture's full-matrix forward/backward passes.
#define TILE 16
__global__ void k_sgemm(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, int M, int K, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;
  for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
    int ak = t * TILE + threadIdx.x;
    int bk = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * (long)K + ak] : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * (long)N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * (long)N + col] = acc;
}

// LoRA accumulate variant: C[M×N] += alpha · A[M×K]·B[K×N].
// Used for the low-rank adapter contribution h += (α/r)·(X·A)·B where the
// tiny [T×r] activation X·A is precomputed on the host and B ∈ R^{r×d}.
__global__ void k_sgemm_acc(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int K, int N, float alpha) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;
  for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
    int ak = t * TILE + threadIdx.x;
    int bk = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * (long)K + ak] : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * (long)N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * (long)N + col] += alpha * acc;
}

// ── GPU-native stochastic exploration for LoRA adapter pools ───────────────
// Stateless PCG-style hash: no host RNG state, no random buffers crossing PCIe.
__device__ __forceinline__ unsigned int pcg_hash(unsigned int input) {
  unsigned int state = input * 747796405u + 2891336453u;
  unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__device__ __forceinline__ float u01(unsigned int seed) {
  return ((pcg_hash(seed) >> 8) + 1.0f) * 0x1.0p-24f;
}

__device__ __forceinline__ float pseudo_gaussian(unsigned int seed) {
  const float u1 = fmaxf(u01(seed), 1e-7f);
  const float u2 = u01(seed ^ 0x9e3779b9u);
  return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

__global__ void k_stochastic_mutate(const float* __restrict__ base,
                                    float* __restrict__ population,
                                    int paramCount, int populationSize,
                                    unsigned int iterationStep, float sigma) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = (long)paramCount * populationSize;
  for (; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const int candidate = (int)(idx / paramCount);
    const int weightIdx = (int)(idx % paramCount);
    const unsigned int seed = pcg_hash((unsigned int)weightIdx + iterationStep * 1337u +
                                       (unsigned int)candidate * 0x85ebca6bu);
    population[idx] = base[weightIdx] + sigma * pseudo_gaussian(seed);
  }
}

__global__ void k_reduce_fittest(const float* __restrict__ losses,
                                 int* __restrict__ order,
                                 int populationSize) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int i = 0; i < populationSize; i++) order[i] = i;
  for (int i = 0; i < populationSize - 1; i++) {
    int best = i;
    for (int j = i + 1; j < populationSize; j++) {
      if (losses[order[j]] < losses[order[best]]) best = j;
    }
    int tmp = order[i];
    order[i] = order[best];
    order[best] = tmp;
  }
}

__global__ void k_gather_survivors(const float* __restrict__ population,
                                   const int* __restrict__ order,
                                   float* __restrict__ survivors,
                                   int paramCount, int survivorCount) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = (long)paramCount * survivorCount;
  for (; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const int s = (int)(idx / paramCount);
    const int w = (int)(idx % paramCount);
    survivors[idx] = population[(long)order[s] * paramCount + w];
  }
}

__global__ void k_replenish_population(float* __restrict__ population,
                                       const float* __restrict__ survivors,
                                       int paramCount, int populationSize,
                                       int survivorCount, unsigned int iterationStep,
                                       float sigma) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = (long)paramCount * populationSize;
  for (; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const int candidate = (int)(idx / paramCount);
    const int weightIdx = (int)(idx % paramCount);
    const int survivor = candidate % survivorCount;
    float value = survivors[(long)survivor * paramCount + weightIdx];
    if (candidate >= survivorCount) {
      const unsigned int seed = pcg_hash((unsigned int)weightIdx + iterationStep * 7331u +
                                         (unsigned int)candidate * 0xc2b2ae35u);
      value += sigma * 0.35f * pseudo_gaussian(seed);
    }
    population[idx] = value;
  }
}

__global__ void k_blend_best(float* __restrict__ base,
                             const float* __restrict__ best,
                             int paramCount, float blend) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  for (; idx < paramCount; idx += (long)gridDim.x * blockDim.x) {
    base[idx] = base[idx] * (1.0f - blend) + best[idx] * blend;
  }
}

// ── Flash-style attention: last-query row, tiled KV slicing ───────────────
// Computes out = softmax(q·K^T/√d)·V for q = x[T-1], K = V = x, WITHOUT ever
// materializing the T×T score matrix: keys stream through a shared-memory
// tile (ATTN_TILE rows) with an online-softmax running max/denominator — the
// same memory-slicing trick as FlashAttention. Peak device memory is
// O(T·d + tile) regardless of context length, so ctx=4096 fits easily in the
// RTX 5060 Ti's VRAM.
#define ATTN_TILE 64
#define ATTN_MAX_D 1024
__global__ void k_attn_last_forward(const float* __restrict__ x, int T, int d,
                                    float* __restrict__ out) {
  // Single block; threads cooperate across the d dimension.
  __shared__ float sQ[ATTN_MAX_D];
  __shared__ float sScores[ATTN_TILE];
  __shared__ float sRunMax, sDenom, sCorr;
  extern __shared__ float sAcc[]; // [d] accumulator in dynamic shared mem

  const float scale = rsqrtf((float)d);
  const long qOff = (long)(T - 1) * d;
  for (int k = threadIdx.x; k < d; k += blockDim.x) {
    sQ[k] = x[qOff + k];
    sAcc[k] = 0.f;
  }
  if (threadIdx.x == 0) { sRunMax = -1e30f; sDenom = 0.f; }
  __syncthreads();

  for (int tile = 0; tile < T; tile += ATTN_TILE) {
    int rows = min(ATTN_TILE, T - tile);
    // 1) scores for this KV slice (one thread per row)
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
      const float* key = x + (long)(tile + r) * d;
      float s = 0.f;
      for (int k = 0; k < d; k++) s += sQ[k] * key[k];
      sScores[r] = s * scale;
    }
    __syncthreads();
    // 2) online-softmax rescale (thread 0 finds tile max, computes correction)
    if (threadIdx.x == 0) {
      float tileMax = sRunMax;
      for (int r = 0; r < rows; r++) tileMax = fmaxf(tileMax, sScores[r]);
      sCorr = expf(sRunMax - tileMax);
      sRunMax = tileMax;
      float dAdd = 0.f;
      for (int r = 0; r < rows; r++) {
        sScores[r] = expf(sScores[r] - tileMax);
        dAdd += sScores[r];
      }
      sDenom = sDenom * sCorr + dAdd;
    }
    __syncthreads();
    // 3) rescale accumulator + add weighted values (threads split d)
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
      float acc = sAcc[k] * sCorr;
      for (int r = 0; r < rows; r++) acc += sScores[r] * x[(long)(tile + r) * d + k];
      sAcc[k] = acc;
    }
    __syncthreads();
  }

  const float inv = 1.f / sDenom;
  for (int k = threadIdx.x; k < d; k += blockDim.x) out[qOff + k] = sAcc[k] * inv;
}

// ── extern "C" ABI consumed by addon.cc ─────────────────────────────────────

extern "C" {

int llm_cuda_available() {
  int n = 0;
  return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 1 : 0;
}

int llm_device_name(char* buf, int len) {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0;
  snprintf(buf, len, "%s (sm_%d%d, %.1fGB)", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1073741824.0);
  return 1;
}

void* llm_ctx_create(const float* E, int V, int d) {
  LlmCtx* ctx = new LlmCtx{nullptr, nullptr, nullptr, nullptr, V, d, false,
                           0, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
  size_t ed = (size_t)V * d * sizeof(float);
  CUDA_CHECK(cudaMalloc(&ctx->dE, ed));
  CUDA_CHECK(cudaMalloc(&ctx->dGE, ed));
  CUDA_CHECK(cudaMalloc(&ctx->dY, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ctx->dLog, V * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(ctx->dE, E, ed, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(ctx->dGE, 0, ed));
  return ctx;
}

// Allocate the multi-layer weight arena: loop through nLayers and give every
// layer index its own discrete cudaMalloc'd buffers — w_attention_queries[l],
// w_attention_keys[l], w_attention_values[l], w_mlp_gate[l], w_mlp_down[l].
// Idempotent-safe: re-allocation with a new depth frees the old arrays first.
void llm_ctx_alloc_buffers(void* p, int nLayers, int d, int h) {
  LlmCtx* ctx = (LlmCtx*)p;
  if (ctx->freed) return;
  if (ctx->nLayers > 0) {
    for (int l = 0; l < ctx->nLayers; l++) {
      cudaFree(ctx->w_attention_queries[l]);
      cudaFree(ctx->w_attention_keys[l]);
      cudaFree(ctx->w_attention_values[l]);
      cudaFree(ctx->w_mlp_gate[l]);
      cudaFree(ctx->w_mlp_down[l]);
    }
    delete[] ctx->w_attention_queries;
    delete[] ctx->w_attention_keys;
    delete[] ctx->w_attention_values;
    delete[] ctx->w_mlp_gate;
    delete[] ctx->w_mlp_down;
  }
  ctx->nLayers = nLayers;
  ctx->hidden = h;
  ctx->w_attention_queries = new float*[nLayers];
  ctx->w_attention_keys = new float*[nLayers];
  ctx->w_attention_values = new float*[nLayers];
  ctx->w_mlp_gate = new float*[nLayers];
  ctx->w_mlp_down = new float*[nLayers];
  size_t dd = (size_t)d * d * sizeof(float);
  size_t dh = (size_t)d * h * sizeof(float);
  for (int l = 0; l < nLayers; l++) {
    CUDA_CHECK(cudaMalloc(&ctx->w_attention_queries[l], dd));
    CUDA_CHECK(cudaMalloc(&ctx->w_attention_keys[l], dd));
    CUDA_CHECK(cudaMalloc(&ctx->w_attention_values[l], dd));
    CUDA_CHECK(cudaMalloc(&ctx->w_mlp_gate[l], dh));
    CUDA_CHECK(cudaMalloc(&ctx->w_mlp_down[l], dh));
  }
}

// Push one layer's weight matrices host→device (after each optimizer step).
void llm_ctx_sync_layer(void* p, int layer, const float* wq, const float* wk,
                        const float* wv, const float* wg, const float* wd) {
  LlmCtx* ctx = (LlmCtx*)p;
  if (ctx->freed || layer < 0 || layer >= ctx->nLayers) return;
  size_t dd = (size_t)ctx->d * ctx->d * sizeof(float);
  size_t dh = (size_t)ctx->d * ctx->hidden * sizeof(float);
  CUDA_CHECK(cudaMemcpy(ctx->w_attention_queries[layer], wq, dd, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->w_attention_keys[layer], wk, dd, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->w_attention_values[layer], wv, dd, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->w_mlp_gate[layer], wg, dh, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->w_mlp_down[layer], wd, dh, cudaMemcpyHostToDevice));
}

// Explicit GPU release for cancel_training: frees device buffers immediately
// but keeps the struct alive for the eventual GC finalizer. Idempotent.
void llm_ctx_release_buffers(void* p) {
  LlmCtx* ctx = (LlmCtx*)p;
  if (ctx->freed) return;
  cudaFree(ctx->dE);
  cudaFree(ctx->dGE);
  cudaFree(ctx->dY);
  cudaFree(ctx->dLog);
  ctx->dE = ctx->dGE = ctx->dY = ctx->dLog = nullptr;
  for (int l = 0; l < ctx->nLayers; l++) {
    cudaFree(ctx->w_attention_queries[l]);
    cudaFree(ctx->w_attention_keys[l]);
    cudaFree(ctx->w_attention_values[l]);
    cudaFree(ctx->w_mlp_gate[l]);
    cudaFree(ctx->w_mlp_down[l]);
  }
  if (ctx->nLayers > 0) {
    delete[] ctx->w_attention_queries;
    delete[] ctx->w_attention_keys;
    delete[] ctx->w_attention_values;
    delete[] ctx->w_mlp_gate;
    delete[] ctx->w_mlp_down;
    ctx->w_attention_queries = ctx->w_attention_keys = ctx->w_attention_values = nullptr;
    ctx->w_mlp_gate = ctx->w_mlp_down = nullptr;
    ctx->nLayers = 0;
  }
  ctx->freed = true;
}

void llm_ctx_destroy(void* p) {
  llm_ctx_release_buffers(p);
  delete (LlmCtx*)p;
}

// Re-upload E after the host-side optimizer step.
void llm_ctx_sync_e(void* p, const float* E) {
  LlmCtx* ctx = (LlmCtx*)p;
  if (ctx->freed) return;
  CUDA_CHECK(cudaMemcpy(ctx->dE, E, (size_t)ctx->V * ctx->d * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void llm_ctx_logits_forward(void* p, const float* y, float* logits) {
  LlmCtx* ctx = (LlmCtx*)p;
  if (ctx->freed) return;
  CUDA_CHECK(cudaMemcpy(ctx->dY, y, ctx->d * sizeof(float), cudaMemcpyHostToDevice));
  k_logits_forward<<<ctx->V, 256>>>(ctx->dE, ctx->dY, ctx->dLog, ctx->V, ctx->d);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(logits, ctx->dLog, ctx->V * sizeof(float), cudaMemcpyDeviceToHost));
}

void llm_ctx_logits_backward(void* p, const float* y, const float* dLogits, float* dY) {
  LlmCtx* ctx = (LlmCtx*)p;
  if (ctx->freed) return;
  CUDA_CHECK(cudaMemcpy(ctx->dY, y, ctx->d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->dLog, dLogits, ctx->V * sizeof(float), cudaMemcpyHostToDevice));
  long total = (long)ctx->V * ctx->d;
  int blocks = (int)((total + 255) / 256);
  if (blocks > 65535) blocks = 65535;
  k_grad_e_acc<<<blocks, 256>>>(ctx->dGE, ctx->dLog, ctx->dY, ctx->V, ctx->d);
  CUDA_CHECK_LAUNCH();
  k_dy<<<(ctx->d + 127) / 128, 128>>>(ctx->dE, ctx->dLog, ctx->dY, ctx->V, ctx->d);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(dY, ctx->dY, ctx->d * sizeof(float), cudaMemcpyDeviceToHost));
}

// Drain the on-device gradient accumulator into the host grad buffer (+=).
void llm_ctx_flush_grad_e(void* p, float* gE) {
  LlmCtx* ctx = (LlmCtx*)p;
  if (ctx->freed) return;
  size_t n = (size_t)ctx->V * ctx->d;
  float* host = new float[n];
  CUDA_CHECK(cudaMemcpy(host, ctx->dGE, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++) gE[i] += host[i];
  delete[] host;
  CUDA_CHECK(cudaMemset(ctx->dGE, 0, n * sizeof(float)));
}

// Streaming last-row attention with tiled KV slicing (see kernel above).
// Host x [T*d] in, host out [T*d] (only the final row is written).
int llm_attn_last_forward(const float* x, int T, int d, float* out) {
  if (d > ATTN_MAX_D) return 0; // fall back to CPU for very wide models
  float *dX, *dOut;
  size_t bytes = (size_t)T * d * sizeof(float);
  CUDA_CHECK(cudaMalloc(&dX, bytes));
  CUDA_CHECK(cudaMalloc(&dOut, bytes));
  CUDA_CHECK(cudaMemcpy(dX, x, bytes, cudaMemcpyHostToDevice));
  k_attn_last_forward<<<1, 256, d * sizeof(float)>>>(dX, T, d, dOut);
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error %s at %s:%d\n",
            cudaGetErrorString(launchErr), __FILE__, __LINE__);
    cudaFree(dX);
    cudaFree(dOut);
    return 0; // signal CPU fallback instead of returning stale zeros
  }
  CUDA_CHECK(cudaMemcpy(out + (long)(T - 1) * d, dOut + (long)(T - 1) * d,
                        d * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dX);
  cudaFree(dOut);
  return 1;
}

void llm_sgemm(const float* A, const float* B, float* C, int M, int K, int N) {
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, A, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  dim3 block(TILE, TILE);
  k_sgemm<<<grid, block>>>(dA, dB, dC, M, K, N);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(C, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

// LoRA forward accumulate: C += alpha·A·B (see k_sgemm_acc). Returns 0 on
// launch failure so the caller falls back to the CPU path instead of
// silently keeping a stale C.
int llm_sgemm_acc(const float* A, const float* B, float* C, int M, int K, int N,
                  float alpha) {
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, A, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, C, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice));
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  dim3 block(TILE, TILE);
  k_sgemm_acc<<<grid, block>>>(dA, dB, dC, M, K, N, alpha);
  cudaError_t launchErr = cudaGetLastError();
  int ok = 1;
  if (launchErr != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error %s at %s:%d\n",
            cudaGetErrorString(launchErr), __FILE__, __LINE__);
    ok = 0;
  } else {
    CUDA_CHECK(cudaMemcpy(C, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
  }
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return ok;
}

int llm_es_mutate_population(const float* base, float* population, int paramCount,
                             int populationSize, int iterationStep, float sigma) {
  if (paramCount <= 0 || populationSize <= 0) return 0;
  float *dBase, *dPop;
  const size_t baseBytes = (size_t)paramCount * sizeof(float);
  const size_t popBytes = (size_t)paramCount * populationSize * sizeof(float);
  CUDA_CHECK(cudaMalloc(&dBase, baseBytes));
  CUDA_CHECK(cudaMalloc(&dPop, popBytes));
  CUDA_CHECK(cudaMemcpy(dBase, base, baseBytes, cudaMemcpyHostToDevice));
  int blocks = (int)(((long)paramCount * populationSize + 255) / 256);
  if (blocks > 65535) blocks = 65535;
  k_stochastic_mutate<<<blocks, 256>>>(dBase, dPop, paramCount, populationSize,
                                       (unsigned int)iterationStep, sigma);
  cudaError_t launchErr = cudaGetLastError();
  int ok = 1;
  if (launchErr != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error %s at %s:%d\n",
            cudaGetErrorString(launchErr), __FILE__, __LINE__);
    ok = 0;
  } else {
    CUDA_CHECK(cudaMemcpy(population, dPop, popBytes, cudaMemcpyDeviceToHost));
  }
  cudaFree(dBase);
  cudaFree(dPop);
  return ok;
}

int llm_ctx_reduce_fittest(float* base, float* population, const float* losses,
                           int paramCount, int populationSize, int survivorCount,
                           int iterationStep, float blend, float sigma,
                           float* metrics) {
  if (paramCount <= 0 || populationSize <= 0) return 0;
  survivorCount = max(1, min(survivorCount, populationSize));
  blend = fmaxf(0.0f, fminf(blend, 1.0f));
  float *dBase, *dPop, *dLoss, *dSurvivors;
  int* dOrder;
  const size_t baseBytes = (size_t)paramCount * sizeof(float);
  const size_t popBytes = (size_t)paramCount * populationSize * sizeof(float);
  const size_t lossBytes = (size_t)populationSize * sizeof(float);
  const size_t survivorBytes = (size_t)paramCount * survivorCount * sizeof(float);
  CUDA_CHECK(cudaMalloc(&dBase, baseBytes));
  CUDA_CHECK(cudaMalloc(&dPop, popBytes));
  CUDA_CHECK(cudaMalloc(&dLoss, lossBytes));
  CUDA_CHECK(cudaMalloc(&dOrder, (size_t)populationSize * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dSurvivors, survivorBytes));
  CUDA_CHECK(cudaMemcpy(dBase, base, baseBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dPop, population, popBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dLoss, losses, lossBytes, cudaMemcpyHostToDevice));
  k_reduce_fittest<<<1, 1>>>(dLoss, dOrder, populationSize);
  CUDA_CHECK_LAUNCH();
  int blocksSurvivors = (int)(((long)paramCount * survivorCount + 255) / 256);
  if (blocksSurvivors > 65535) blocksSurvivors = 65535;
  k_gather_survivors<<<blocksSurvivors, 256>>>(dPop, dOrder, dSurvivors,
                                               paramCount, survivorCount);
  CUDA_CHECK_LAUNCH();
  int blocksPop = (int)(((long)paramCount * populationSize + 255) / 256);
  if (blocksPop > 65535) blocksPop = 65535;
  k_replenish_population<<<blocksPop, 256>>>(dPop, dSurvivors, paramCount,
                                             populationSize, survivorCount,
                                             (unsigned int)iterationStep, sigma);
  CUDA_CHECK_LAUNCH();
  int blocksBase = (int)((paramCount + 255) / 256);
  if (blocksBase > 65535) blocksBase = 65535;
  k_blend_best<<<blocksBase, 256>>>(dBase, dSurvivors, paramCount, blend);
  cudaError_t launchErr = cudaGetLastError();
  int ok = 1;
  if (launchErr != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error %s at %s:%d\n",
            cudaGetErrorString(launchErr), __FILE__, __LINE__);
    ok = 0;
  } else {
    int* order = new int[populationSize];
    CUDA_CHECK(cudaMemcpy(order, dOrder, (size_t)populationSize * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(base, dBase, baseBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(population, dPop, popBytes, cudaMemcpyDeviceToHost));
    float mean = 0.0f;
    for (int i = 0; i < populationSize; i++) mean += losses[i];
    mean /= populationSize;
    float variance = 0.0f;
    for (int i = 0; i < populationSize; i++) {
      const float delta = losses[i] - mean;
      variance += delta * delta;
    }
    metrics[0] = losses[order[0]];
    metrics[1] = variance / populationSize;
    metrics[2] = (float)order[0];
    delete[] order;
  }
  cudaFree(dBase);
  cudaFree(dPop);
  cudaFree(dLoss);
  cudaFree(dOrder);
  cudaFree(dSurvivors);
  return ok;
}

}  // extern "C"
