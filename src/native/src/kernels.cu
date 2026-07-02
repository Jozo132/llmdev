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

// Persistent GPU context for one model's tied output head:
// the embedding matrix E [V×d] lives on-device across the whole training run,
// and the E-gradient accumulates on-device — host↔device traffic per token is
// only y (d floats), logits (V floats) and dLogits (V floats).
struct LlmCtx {
  float* dE;    // device E            [V*d]
  float* dGE;   // device grad-E accum [V*d]
  float* dY;    // device y / dY       [d]
  float* dLog;  // device logits/dLog  [V]
  int V;
  int d;
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
  LlmCtx* ctx = new LlmCtx{nullptr, nullptr, nullptr, nullptr, V, d};
  size_t ed = (size_t)V * d * sizeof(float);
  CUDA_CHECK(cudaMalloc(&ctx->dE, ed));
  CUDA_CHECK(cudaMalloc(&ctx->dGE, ed));
  CUDA_CHECK(cudaMalloc(&ctx->dY, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ctx->dLog, V * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(ctx->dE, E, ed, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(ctx->dGE, 0, ed));
  return ctx;
}

void llm_ctx_destroy(void* p) {
  LlmCtx* ctx = (LlmCtx*)p;
  cudaFree(ctx->dE);
  cudaFree(ctx->dGE);
  cudaFree(ctx->dY);
  cudaFree(ctx->dLog);
  delete ctx;
}

// Re-upload E after the host-side optimizer step.
void llm_ctx_sync_e(void* p, const float* E) {
  LlmCtx* ctx = (LlmCtx*)p;
  CUDA_CHECK(cudaMemcpy(ctx->dE, E, (size_t)ctx->V * ctx->d * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void llm_ctx_logits_forward(void* p, const float* y, float* logits) {
  LlmCtx* ctx = (LlmCtx*)p;
  CUDA_CHECK(cudaMemcpy(ctx->dY, y, ctx->d * sizeof(float), cudaMemcpyHostToDevice));
  k_logits_forward<<<ctx->V, 256>>>(ctx->dE, ctx->dY, ctx->dLog, ctx->V, ctx->d);
  CUDA_CHECK(cudaMemcpy(logits, ctx->dLog, ctx->V * sizeof(float), cudaMemcpyDeviceToHost));
}

void llm_ctx_logits_backward(void* p, const float* y, const float* dLogits, float* dY) {
  LlmCtx* ctx = (LlmCtx*)p;
  CUDA_CHECK(cudaMemcpy(ctx->dY, y, ctx->d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->dLog, dLogits, ctx->V * sizeof(float), cudaMemcpyHostToDevice));
  long total = (long)ctx->V * ctx->d;
  int blocks = (int)((total + 255) / 256);
  if (blocks > 65535) blocks = 65535;
  k_grad_e_acc<<<blocks, 256>>>(ctx->dGE, ctx->dLog, ctx->dY, ctx->V, ctx->d);
  k_dy<<<(ctx->d + 127) / 128, 128>>>(ctx->dE, ctx->dLog, ctx->dY, ctx->V, ctx->d);
  CUDA_CHECK(cudaMemcpy(dY, ctx->dY, ctx->d * sizeof(float), cudaMemcpyDeviceToHost));
}

// Drain the on-device gradient accumulator into the host grad buffer (+=).
void llm_ctx_flush_grad_e(void* p, float* gE) {
  LlmCtx* ctx = (LlmCtx*)p;
  size_t n = (size_t)ctx->V * ctx->d;
  float* host = new float[n];
  CUDA_CHECK(cudaMemcpy(host, ctx->dGE, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++) gE[i] += host[i];
  delete[] host;
  CUDA_CHECK(cudaMemset(ctx->dGE, 0, n * sizeof(float)));
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
  CUDA_CHECK(cudaMemcpy(C, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

}  // extern "C"
