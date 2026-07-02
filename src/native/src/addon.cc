// addon.cc — node-addon-api wrapper around the CUDA kernels (kernels.cu).
//
// Zero-copy by construction: JS Float32Array buffers are handed to the CUDA
// layer as raw float* pointers taken directly from the V8 ArrayBuffer backing
// store — no serialization, no intermediate copies on the host side.
//
// A "context" pins the embedding matrix (and its gradient accumulator) in GPU
// memory for the lifetime of a training run; it is exposed to JS as an
// External handle with a finalizer, so leaked contexts are reclaimed by GC.

#include <napi.h>

extern "C" {
int llm_cuda_available();
int llm_device_name(char* buf, int len);
void* llm_ctx_create(const float* E, int V, int d);
void llm_ctx_destroy(void* ctx);
void llm_ctx_release_buffers(void* ctx);
void llm_ctx_sync_e(void* ctx, const float* E);
void llm_ctx_logits_forward(void* ctx, const float* y, float* logits);
void llm_ctx_logits_backward(void* ctx, const float* y, const float* dLogits, float* dY);
void llm_ctx_flush_grad_e(void* ctx, float* gE);
void llm_ctx_alloc_buffers(void* ctx, int nLayers, int d, int h);
void llm_ctx_sync_layer(void* ctx, int layer, const float* wq, const float* wk,
                        const float* wv, const float* wg, const float* wd);
int llm_attn_last_forward(const float* x, int T, int d, float* out);
void llm_sgemm(const float* A, const float* B, float* C, int M, int K, int N);
}

namespace {

float* F32(const Napi::Value& v) {
  return v.As<Napi::Float32Array>().Data();
}

Napi::Value CudaAvailable(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), llm_cuda_available() == 1);
}

Napi::Value DeviceName(const Napi::CallbackInfo& info) {
  char buf[256];
  if (!llm_device_name(buf, sizeof(buf))) return info.Env().Null();
  return Napi::String::New(info.Env(), buf);
}

// createContext(E: Float32Array, V: number, d: number) → External
Napi::Value CreateContext(const Napi::CallbackInfo& info) {
  int V = info[1].As<Napi::Number>().Int32Value();
  int d = info[2].As<Napi::Number>().Int32Value();
  void* ctx = llm_ctx_create(F32(info[0]), V, d);
  return Napi::External<void>::New(info.Env(), ctx,
                                   [](Napi::Env, void* p) { llm_ctx_destroy(p); });
}

// syncE(ctx, E) — push updated weights to the GPU after an optimizer step.
Napi::Value SyncE(const Napi::CallbackInfo& info) {
  llm_ctx_sync_e(info[0].As<Napi::External<void>>().Data(), F32(info[1]));
  return info.Env().Undefined();
}

// logitsForward(ctx, y: Float32Array[d], logits: Float32Array[V])
Napi::Value LogitsForward(const Napi::CallbackInfo& info) {
  llm_ctx_logits_forward(info[0].As<Napi::External<void>>().Data(), F32(info[1]), F32(info[2]));
  return info.Env().Undefined();
}

// logitsBackward(ctx, y[d], dLogits[V], dY[d]) — accumulates grad-E on device.
Napi::Value LogitsBackward(const Napi::CallbackInfo& info) {
  llm_ctx_logits_backward(info[0].As<Napi::External<void>>().Data(), F32(info[1]), F32(info[2]),
                          F32(info[3]));
  return info.Env().Undefined();
}

// flushGradE(ctx, gE: Float32Array[V*d]) — drain device accumulator (+=).
Napi::Value FlushGradE(const Napi::CallbackInfo& info) {
  llm_ctx_flush_grad_e(info[0].As<Napi::External<void>>().Data(), F32(info[1]));
  return info.Env().Undefined();
}

// releaseContext(ctx) — immediate device-buffer free for cancel_training.
// Safe to call repeatedly; the GC finalizer later reclaims the host struct.
Napi::Value ReleaseContext(const Napi::CallbackInfo& info) {
  llm_ctx_release_buffers(info[0].As<Napi::External<void>>().Data());
  return info.Env().Undefined();
}

// allocLayers(ctx, nLayers, d, h) — discrete per-layer device weight buffers:
// w_attention_queries[l], w_attention_keys[l], w_attention_values[l],
// w_mlp_gate[l], w_mlp_down[l] for l ∈ [0, nLayers).
Napi::Value AllocLayers(const Napi::CallbackInfo& info) {
  llm_ctx_alloc_buffers(info[0].As<Napi::External<void>>().Data(),
                        info[1].As<Napi::Number>().Int32Value(),
                        info[2].As<Napi::Number>().Int32Value(),
                        info[3].As<Napi::Number>().Int32Value());
  return info.Env().Undefined();
}

// syncLayer(ctx, layer, wq[d*d], wk[d*d], wv[d*d], wGate[d*h], wDown[h*d])
Napi::Value SyncLayer(const Napi::CallbackInfo& info) {
  llm_ctx_sync_layer(info[0].As<Napi::External<void>>().Data(),
                     info[1].As<Napi::Number>().Int32Value(),
                     F32(info[2]), F32(info[3]), F32(info[4]), F32(info[5]), F32(info[6]));
  return info.Env().Undefined();
}

// attnLastForward(x[T*d], T, d, out[T*d]) → bool (false = use CPU fallback)
Napi::Value AttnLastForward(const Napi::CallbackInfo& info) {
  int ok = llm_attn_last_forward(F32(info[0]),
                                 info[1].As<Napi::Number>().Int32Value(),
                                 info[2].As<Napi::Number>().Int32Value(),
                                 F32(info[3]));
  return Napi::Boolean::New(info.Env(), ok == 1);
}

// sgemm(A[M*K], B[K*N], C[M*N], M, K, N) — generic building block.
Napi::Value Sgemm(const Napi::CallbackInfo& info) {
  llm_sgemm(F32(info[0]), F32(info[1]), F32(info[2]),
            info[3].As<Napi::Number>().Int32Value(),
            info[4].As<Napi::Number>().Int32Value(),
            info[5].As<Napi::Number>().Int32Value());
  return info.Env().Undefined();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("cudaAvailable", Napi::Function::New(env, CudaAvailable));
  exports.Set("deviceName", Napi::Function::New(env, DeviceName));
  exports.Set("createContext", Napi::Function::New(env, CreateContext));
  exports.Set("releaseContext", Napi::Function::New(env, ReleaseContext));
  exports.Set("syncE", Napi::Function::New(env, SyncE));
  exports.Set("logitsForward", Napi::Function::New(env, LogitsForward));
  exports.Set("logitsBackward", Napi::Function::New(env, LogitsBackward));
  exports.Set("flushGradE", Napi::Function::New(env, FlushGradE));
  exports.Set("allocLayers", Napi::Function::New(env, AllocLayers));
  exports.Set("syncLayer", Napi::Function::New(env, SyncLayer));
  exports.Set("attnLastForward", Napi::Function::New(env, AttnLastForward));
  exports.Set("sgemm", Napi::Function::New(env, Sgemm));
  return exports;
}

}  // namespace

NODE_API_MODULE(llmdev_native, Init)
