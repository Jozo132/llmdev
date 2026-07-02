<script setup lang="ts">
/**
 * ChatSandbox — immersive widescreen dual-pane playground.
 *
 *   LEFT (fixed):  model binding, sampling controls (with hyperparameter
 *                  theory), context files injected into the prompt, MCP tool
 *                  configuration, and the composer.
 *   RIGHT (flow):  streaming assistant output token-by-token, fenced code
 *                  blocks with syntax highlighting, parsed JSON-RPC payload
 *                  cards, and the tool-invocation result log.
 */
import { computed, nextTick, reactive, ref, watch, onMounted } from "vue";
import { usePipelineStore, type ChatMessage } from "../stores/pipeline";
import { highlight, segmentMessage } from "../lib/highlight";

const store = usePipelineStore();
const input = ref("");
const scroller = ref<HTMLElement | null>(null);
let rpcId = 1;

// ── Sampling controls with inline theory ─────────────────────────────────────
const sampling = reactive({ maxTokens: 96, temperature: 0.8, topP: 0.95 });
const SAMPLING_THEORY: Record<string, { theory: string; range: string }> = {
  temperature: {
    theory:
      "Divides the logits before softmax: p ∝ exp(z/T). T→0 sharpens toward " +
      "greedy argmax (deterministic, repetitive); T>1 flattens toward uniform " +
      "(creative, incoherent). It rescales the model's confidence, not its knowledge.",
    range: "0 (greedy) · 0.6–0.9 typical · >1.2 chaotic",
  },
  topP: {
    theory:
      "Nucleus sampling: sample only from the smallest token set whose " +
      "cumulative probability ≥ P, cutting the low-probability tail where " +
      "degenerate tokens live. Adapts set size to the model's certainty, " +
      "unlike fixed top-k.",
    range: "0.9–0.98 · 1.0 disables",
  },
  maxTokens: {
    theory:
      "Generation budget. Each token is a full forward pass — latency and " +
      "compute scale linearly. No KV-cache in the PoC, so cost per token also " +
      "grows with accumulated context length.",
    range: "16–512",
  },
};
const openTheory = ref<string | null>(null);

// ── Context files (prepended to every prompt) ────────────────────────────────
const contextFiles = ref<Array<{ name: string; content: string; enabled: boolean }>>([]);
const newFileName = ref("");
const newFileContent = ref("");

function addContextFile() {
  if (!newFileName.value.trim() || !newFileContent.value.trim()) return;
  contextFiles.value.push({
    name: newFileName.value.trim(),
    content: newFileContent.value,
    enabled: true,
  });
  newFileName.value = "";
  newFileContent.value = "";
}

const contextPrefix = computed(() =>
  contextFiles.value
    .filter((f) => f.enabled)
    .map((f) => `// FILE: ${f.name}\n${f.content}\n`)
    .join("\n")
);

// ── MCP tooling ──────────────────────────────────────────────────────────────
onMounted(() => {
  store.connect();
  // Populate the tool catalog once the bridge is up.
  const poll = setInterval(() => {
    if (store.connected) {
      dispatchRpc({ method: "tools/list" }, true);
      clearInterval(poll);
    }
  }, 400);
});

function dispatchRpc(payload: Record<string, unknown>, silent = false) {
  if (!silent) {
    store.chatMessages.push({
      role: "user",
      text: `⚙ JSON-RPC → ${String((payload as { method?: string }).method)}`,
    });
  }
  store.sendMcp({ jsonrpc: "2.0", id: rpcId++, ...payload });
}

function invokeTool(name: string) {
  dispatchRpc({ method: "tools/call", params: { name, arguments: {} } });
}

// ── JSON-RPC extraction from assistant output ────────────────────────────────
interface ParsedRpc {
  payload: Record<string, unknown>;
  pretty: string;
}

function extractJsonRpc(text: string): ParsedRpc[] {
  const found: ParsedRpc[] = [];
  const candidates: string[] = [];
  for (const m of text.matchAll(/```json\s*([\s\S]*?)```/g)) candidates.push(m[1]);
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== "{") continue;
    let depth = 0;
    for (let j = i; j < text.length; j++) {
      if (text[j] === "{") depth++;
      else if (text[j] === "}") {
        depth--;
        if (depth === 0) {
          const chunk = text.slice(i, j + 1);
          if (chunk.includes('"jsonrpc"')) candidates.push(chunk);
          i = j;
          break;
        }
      }
    }
  }
  for (const c of candidates) {
    try {
      const obj = JSON.parse(c);
      if (obj && typeof obj === "object" && (obj.jsonrpc === "2.0" || obj.method)) {
        found.push({ payload: obj, pretty: JSON.stringify(obj, null, 2) });
      }
    } catch { /* not JSON */ }
  }
  return found;
}

const rpcFor = (m: ChatMessage) => (m.role === "assistant" && !m.streaming ? extractJsonRpc(m.text) : []);
const segmentsFor = (m: ChatMessage) => segmentMessage(m.text);

// ── Composer ─────────────────────────────────────────────────────────────────
function submit() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  if (text === "/tools") return dispatchRpc({ method: "tools/list" });
  const toolCmd = text.match(/^\/tool\s+(\S+)\s*(\{.*\})?$/s);
  if (toolCmd) {
    let args = {};
    try { args = toolCmd[2] ? JSON.parse(toolCmd[2]) : {}; } catch { /* {} */ }
    return dispatchRpc({ method: "tools/call", params: { name: toolCmd[1], arguments: args } });
  }

  const prompt = contextPrefix.value ? `${contextPrefix.value}\n${text}` : text;
  store.sendChat(prompt, sampling.maxTokens, sampling.temperature, sampling.topP);
}

watch(
  () => [store.chatMessages.length, store.chatMessages[store.chatMessages.length - 1]?.text],
  async () => {
    await nextTick();
    scroller.value?.scrollTo({ top: scroller.value.scrollHeight });
  }
);
</script>

<template>
  <div class="flex min-h-0 min-w-0 flex-1">
    <!-- ════════ LEFT PANE: input · context · tools ════════ -->
    <aside class="flex w-[26rem] shrink-0 flex-col border-r border-slate-800 bg-panel">
      <div class="flex-1 space-y-5 overflow-y-auto p-4">
        <!-- model binding -->
        <section>
          <h3 class="mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">Model binding</h3>
          <select
            v-model="store.chatVariantId"
            class="w-full rounded border border-slate-700 bg-canvas px-2 py-1.5 font-mono text-xs"
          >
            <option v-for="v in store.library" :key="v.id" :value="v.id">
              {{ v.name }} · {{ (v.paramCount / 1e6).toFixed(1) }}M{{ v.finalLoss ? ` · loss ${v.finalLoss.toFixed(2)}` : "" }}
            </option>
          </select>
        </section>

        <!-- sampling with theory -->
        <section>
          <h3 class="mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">Sampling</h3>
          <div v-for="(cfg, key) in SAMPLING_THEORY" :key="key" class="mb-2">
            <div class="flex items-center gap-1">
              <label class="text-xs text-slate-400">{{ key }}</label>
              <button
                class="flex h-4 w-4 items-center justify-center rounded-full bg-slate-800 text-[9px] text-sky-400 hover:bg-slate-700"
                @click="openTheory = openTheory === key ? null : key"
              >i</button>
              <input
                v-model.number="sampling[key as keyof typeof sampling]"
                type="number" step="any"
                class="ml-auto w-24 rounded border border-slate-700 bg-canvas px-2 py-0.5 text-right font-mono text-xs"
              />
            </div>
            <div v-if="openTheory === key" class="mt-1 rounded border border-sky-900 bg-sky-950/40 p-2 text-[11px] leading-relaxed text-slate-300">
              {{ cfg.theory }}
              <p class="mt-1 font-mono text-[10px] text-amber-300">safe range: {{ cfg.range }}</p>
            </div>
          </div>
        </section>

        <!-- context files -->
        <section>
          <h3 class="mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
            Context files <span class="normal-case text-slate-600">(prepended to prompts)</span>
          </h3>
          <div v-for="(f, i) in contextFiles" :key="i" class="mb-1 flex items-center gap-2 rounded border border-slate-800 px-2 py-1">
            <input v-model="f.enabled" type="checkbox" class="h-3 w-3 accent-emerald-500" />
            <span class="font-mono text-[11px] text-slate-300">{{ f.name }}</span>
            <span class="text-[9px] text-slate-600">{{ f.content.length }} ch</span>
            <button class="ml-auto text-[10px] text-red-400 hover:text-red-300" @click="contextFiles.splice(i, 1)">✕</button>
          </div>
          <input v-model="newFileName" placeholder="filename.js" class="mb-1 w-full rounded border border-slate-700 bg-canvas px-2 py-1 font-mono text-xs" />
          <textarea v-model="newFileContent" rows="3" placeholder="paste file content…" class="mb-1 w-full resize-none rounded border border-slate-700 bg-canvas px-2 py-1 font-mono text-xs" />
          <button class="rounded bg-slate-800 px-2 py-1 text-[11px] hover:bg-slate-700" @click="addContextFile">+ add context file</button>
        </section>

        <!-- MCP tools -->
        <section>
          <h3 class="mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">MCP tools (JSON-RPC 2.0)</h3>
          <div v-for="t in store.mcpTools" :key="t.name" class="mb-1 rounded border border-violet-900/60 px-2 py-1.5">
            <div class="flex items-center">
              <span class="font-mono text-[11px] text-violet-300">{{ t.name }}</span>
              <button
                class="ml-auto rounded bg-violet-800 px-2 py-0.5 text-[10px] font-semibold hover:bg-violet-700"
                @click="invokeTool(t.name)"
              >invoke</button>
            </div>
            <p class="text-[10px] leading-tight text-slate-500">{{ t.description }}</p>
          </div>
          <p class="mt-1 text-[9px] text-slate-600">
            Slash commands: <code>/tools</code> · <code>/tool &lt;name&gt; {json}</code>.
            JSON-RPC envelopes in model output are parsed into dispatchable cards →
          </p>
        </section>
      </div>

      <!-- composer (permanent, bottom-left) -->
      <div class="border-t border-slate-800 p-3">
        <textarea
          v-model="input"
          rows="3"
          placeholder="Message the model…  (Enter to send · Shift+Enter newline)"
          class="w-full resize-none rounded-lg border border-slate-700 bg-canvas px-3 py-2 font-mono text-sm focus:border-sky-600 focus:outline-none"
          @keydown.enter.exact.prevent="submit"
        />
        <div class="mt-2 flex gap-2">
          <button
            v-if="!store.activeChatId"
            class="flex-1 rounded-lg bg-sky-700 py-1.5 text-sm font-semibold hover:bg-sky-600 disabled:opacity-40"
            :disabled="!store.chatVariantId || !input.trim()"
            @click="submit"
          >Send ▸</button>
          <button
            v-else
            class="flex-1 rounded-lg bg-red-800 py-1.5 text-sm font-semibold hover:bg-red-700"
            @click="store.stopChat()"
          >■ Stop generation</button>
        </div>
      </div>
    </aside>

    <!-- ════════ RIGHT PANE: streaming output ════════ -->
    <div ref="scroller" class="min-w-0 flex-1 overflow-y-auto px-8 py-6">
      <div v-if="!store.chatMessages.length" class="mt-24 text-center text-sm text-slate-600">
        <p class="text-2xl">⬡</p>
        <p class="mt-2">Bind a trained checkpoint on the left and start the conversation.</p>
        <p class="mt-1 text-xs">Tokens stream from local inference — small models produce avant-garde JavaScript poetry.</p>
      </div>

      <div v-for="(m, i) in store.chatMessages" :key="i" class="mb-5">
        <!-- user -->
        <div v-if="m.role === 'user'" class="flex justify-end">
          <div class="max-w-[75%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-sky-900/60 px-4 py-2 text-sm">{{ m.text }}</div>
        </div>

        <!-- assistant: prose + highlighted code segments -->
        <div v-else-if="m.role === 'assistant'" class="flex justify-start">
          <div class="w-full max-w-[85%]">
            <p class="mb-0.5 font-mono text-[10px] text-slate-600">{{ m.variantId }}</p>
            <div class="rounded-2xl rounded-bl-sm border border-slate-800 bg-panel px-4 py-3 text-sm">
              <template v-for="(seg, s) in segmentsFor(m)" :key="s">
                <p v-if="seg.kind === 'text'" class="whitespace-pre-wrap font-mono text-slate-200">{{ seg.content }}</p>
                <div v-else class="my-2 overflow-hidden rounded-lg border border-slate-700">
                  <div class="flex items-center bg-slate-900 px-3 py-1">
                    <span class="font-mono text-[10px] uppercase text-slate-500">{{ seg.lang }}</span>
                  </div>
                  <!-- content is HTML-escaped inside highlight() before spans are added -->
                  <pre class="overflow-x-auto bg-black/40 p-3 font-mono text-[12px] leading-relaxed" v-html="highlight(seg.content)" />
                </div>
              </template>
              <span v-if="m.streaming" class="animate-pulse text-emerald-400">▍</span>
            </div>

            <!-- parsed MCP payload cards -->
            <div v-for="(rpc, r) in rpcFor(m)" :key="r" class="mt-2 rounded-lg border border-violet-800 bg-violet-950/40 p-3">
              <p class="mb-1 text-[10px] font-bold uppercase tracking-wider text-violet-300">⚙ JSON-RPC payload detected in output</p>
              <pre class="max-h-40 overflow-auto font-mono text-[11px] text-violet-200">{{ rpc.pretty }}</pre>
              <button
                class="mt-2 rounded bg-violet-700 px-3 py-1 text-[11px] font-semibold hover:bg-violet-600"
                @click="dispatchRpc(rpc.payload)"
              >Dispatch to MCP handler</button>
            </div>
          </div>
        </div>

        <!-- tool invocation log -->
        <div v-else class="flex justify-start">
          <div class="w-full max-w-[85%] rounded-lg border border-emerald-900 bg-emerald-950/40 p-3">
            <p class="mb-1 text-[10px] font-bold uppercase tracking-wider text-emerald-400">tool invocation result</p>
            <pre class="max-h-56 overflow-auto font-mono text-[11px] text-emerald-100">{{ m.text }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
