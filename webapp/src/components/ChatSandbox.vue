<script setup lang="ts">
/**
 * ChatSandbox — universal chat playground for locally trained checkpoints.
 *
 * - Model picker bound to the live library (WS-synced).
 * - Token-by-token streaming inference over the WebSocket bridge.
 * - Structural MCP / tool-calling support:
 *     · assistant output is scanned for JSON-RPC 2.0 envelopes ({"jsonrpc"…})
 *       and fenced ```json blocks — parsed payloads render as tool-call cards
 *       with one-click dispatch to the server's MCP handler;
 *     · `/tool <name> {args}` slash-commands issue tools/call directly
 *       (handy while the toy models are still learning to emit valid JSON);
 *     · `/tools` lists the server's tool registry.
 */
import { computed, nextTick, ref, watch } from "vue";
import { usePipelineStore, type ChatMessage } from "../stores/pipeline";

const store = usePipelineStore();
const input = ref("");
const maxTokens = ref(96);
const temperature = ref(0.8);
const scroller = ref<HTMLElement | null>(null);

let rpcId = 1;

const trainedVariants = computed(() => store.library);

// ── JSON-RPC payload extraction from message text ────────────────────────────
interface ParsedRpc {
  payload: Record<string, unknown>;
  pretty: string;
}

function extractJsonRpc(text: string): ParsedRpc[] {
  const found: ParsedRpc[] = [];
  const candidates: string[] = [];

  // fenced ```json blocks
  for (const m of text.matchAll(/```json\s*([\s\S]*?)```/g)) candidates.push(m[1]);

  // balanced-brace objects containing "jsonrpc"
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
    } catch { /* not valid JSON — ignore */ }
  }
  return found;
}

const rpcFor = (m: ChatMessage) => (m.role === "assistant" && !m.streaming ? extractJsonRpc(m.text) : []);

function dispatchRpc(payload: Record<string, unknown>) {
  store.sendMcp({ jsonrpc: "2.0", id: rpcId++, ...payload });
}

// ── Input handling (chat + slash commands) ───────────────────────────────────
function submit() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  if (text === "/tools") {
    store.chatMessages.push({ role: "user", text });
    dispatchRpc({ method: "tools/list" });
    return;
  }
  const toolCmd = text.match(/^\/tool\s+(\S+)\s*(\{.*\})?$/s);
  if (toolCmd) {
    store.chatMessages.push({ role: "user", text });
    let args = {};
    try { args = toolCmd[2] ? JSON.parse(toolCmd[2]) : {}; } catch { /* default {} */ }
    dispatchRpc({ method: "tools/call", params: { name: toolCmd[1], arguments: args } });
    return;
  }

  store.sendChat(text, maxTokens.value, temperature.value);
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
  <div class="flex min-h-0 flex-1 flex-col">
    <!-- toolbar: model binding + sampling controls -->
    <div class="flex items-center gap-3 border-b border-slate-800 bg-panel/60 px-4 py-2 text-xs">
      <label class="text-slate-500">model</label>
      <select
        v-model="store.chatVariantId"
        class="rounded border border-slate-700 bg-canvas px-2 py-1 font-mono text-xs"
      >
        <option v-for="v in trainedVariants" :key="v.id" :value="v.id">
          {{ v.name }} ({{ (v.paramCount / 1e6).toFixed(1) }}M{{ v.finalLoss ? `, loss ${v.finalLoss.toFixed(2)}` : "" }})
        </option>
      </select>
      <label class="text-slate-500">max tokens</label>
      <input v-model.number="maxTokens" type="number" class="w-16 rounded border border-slate-700 bg-canvas px-1 py-1" />
      <label class="text-slate-500">temp</label>
      <input v-model.number="temperature" type="number" step="0.1" min="0" max="2" class="w-14 rounded border border-slate-700 bg-canvas px-1 py-1" />
      <span class="ml-auto text-slate-600">/tools · /tool &lt;name&gt; {json} · plain text → local inference</span>
    </div>

    <!-- message pane -->
    <div ref="scroller" class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
      <div v-if="!store.chatMessages.length" class="mt-16 text-center text-sm text-slate-600">
        Select a trained checkpoint and start typing.<br />
        <span class="text-xs">Tokens stream straight from local inference — expect avant-garde JavaScript poetry from small models.</span>
      </div>

      <div v-for="(m, i) in store.chatMessages" :key="i" class="mb-4">
        <!-- user -->
        <div v-if="m.role === 'user'" class="flex justify-end">
          <div class="max-w-[70%] rounded-2xl rounded-br-sm bg-sky-900/60 px-4 py-2 text-sm whitespace-pre-wrap">{{ m.text }}</div>
        </div>

        <!-- assistant -->
        <div v-else-if="m.role === 'assistant'" class="flex justify-start">
          <div class="max-w-[80%]">
            <p class="mb-0.5 font-mono text-[10px] text-slate-600">{{ m.variantId }}</p>
            <div class="rounded-2xl rounded-bl-sm border border-slate-800 bg-panel px-4 py-2 font-mono text-sm whitespace-pre-wrap">
              <span>{{ m.text }}</span><span v-if="m.streaming" class="animate-pulse text-emerald-400">▍</span>
            </div>
            <!-- parsed MCP / tool-call payloads -->
            <div v-for="(rpc, r) in rpcFor(m)" :key="r"
                 class="mt-2 rounded border border-violet-800 bg-violet-950/40 p-2">
              <p class="mb-1 text-[10px] font-bold uppercase tracking-wider text-violet-300">
                ⚙ JSON-RPC payload detected
              </p>
              <pre class="max-h-40 overflow-auto font-mono text-[11px] text-violet-200">{{ rpc.pretty }}</pre>
              <button
                class="mt-1 rounded bg-violet-700 px-2 py-0.5 text-[10px] font-semibold hover:bg-violet-600"
                @click="dispatchRpc(rpc.payload)"
              >Dispatch tools/call</button>
            </div>
          </div>
        </div>

        <!-- tool result -->
        <div v-else class="flex justify-start">
          <div class="max-w-[80%] rounded border border-emerald-900 bg-emerald-950/40 p-2">
            <p class="mb-1 text-[10px] font-bold uppercase tracking-wider text-emerald-400">tool result</p>
            <pre class="max-h-48 overflow-auto font-mono text-[11px] text-emerald-100">{{ m.text }}</pre>
          </div>
        </div>
      </div>
    </div>

    <!-- composer -->
    <div class="border-t border-slate-800 bg-panel p-3">
      <div class="flex gap-2">
        <textarea
          v-model="input"
          rows="2"
          placeholder="Message the model…  (Enter to send, Shift+Enter for newline)"
          class="flex-1 resize-none rounded-lg border border-slate-700 bg-canvas px-3 py-2 font-mono text-sm focus:border-sky-600 focus:outline-none"
          @keydown.enter.exact.prevent="submit"
        />
        <button
          v-if="!store.activeChatId"
          class="rounded-lg bg-sky-700 px-4 text-sm font-semibold hover:bg-sky-600 disabled:opacity-40"
          :disabled="!store.chatVariantId || !input.trim()"
          @click="submit"
        >Send</button>
        <button
          v-else
          class="rounded-lg bg-red-800 px-4 text-sm font-semibold hover:bg-red-700"
          @click="store.stopChat()"
        >■ Stop</button>
      </div>
    </div>
  </div>
</template>
