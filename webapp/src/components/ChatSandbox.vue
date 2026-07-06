<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from "vue";
import { usePipelineStore, type ChatMessage } from "../stores/pipeline";
import { highlight, segmentMessage } from "../lib/highlight";

type DiagnosticPart =
  | { kind: "text"; text: string }
  | { kind: "tool_call"; name: string; body: string; inFlight: boolean }
  | { kind: "tool_response"; name: string; body: string };

const store = usePipelineStore();
const input = ref("");
const scroller = ref<HTMLElement | null>(null);
const textarea = ref<HTMLTextAreaElement | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const sidebarOpen = ref(true);
const editingSessionId = ref<string | null>(null);
const editingTitle = ref("");
const openBlocks = reactive<Record<string, boolean>>({});
let rpcId = 1;

const sampling = reactive({ maxTokens: 192, temperature: 0.8, topP: 0.95 });

const activeVariant = computed(() =>
  store.library.find((variant) => variant.id === store.chatVariantId) ?? null
);

const contextWindow = computed(() => {
  const value = activeVariant.value?.config.contextLength;
  return typeof value === "number" ? value : null;
});

const quickStarts = computed(() =>
  store.library.slice(0, 6).map((variant) => ({
    id: variant.id,
    title: variant.name,
    meta: `${(variant.paramCount / 1e6).toFixed(1)}M params${variant.finalLoss ? ` · loss ${variant.finalLoss.toFixed(3)}` : ""}`,
    prompt: `Inspect ${variant.name} and suggest a focused next evaluation.`
  }))
);

onMounted(() => {
  store.connect();
  const poll = setInterval(() => {
    if (!store.connected) return;
    dispatchRpc({ method: "tools/list" }, true);
    clearInterval(poll);
  }, 400);
});

watch(
  () => [store.chatMessages.length, store.chatMessages[store.chatMessages.length - 1]?.text],
  async () => {
    await nextTick();
    scroller.value?.scrollTo({ top: scroller.value.scrollHeight, behavior: "smooth" });
  }
);

watch(input, async () => {
  await nextTick();
  if (!textarea.value) return;
  textarea.value.style.height = "0px";
  textarea.value.style.height = `${Math.min(textarea.value.scrollHeight, 180)}px`;
});

function dispatchRpc(payload: Record<string, unknown>, silent = false) {
  if (!silent) {
    store.chatMessages.push({ role: "system", text: `JSON-RPC -> ${String(payload.method ?? "mcp")}` });
  }
  store.sendMcp({ jsonrpc: "2.0", id: rpcId++, ...payload });
}

function newChat() {
  store.newChat(store.chatVariantId);
  nextTick(() => textarea.value?.focus());
}

function beginRename(sessionId: string, title: string) {
  editingSessionId.value = sessionId;
  editingTitle.value = title;
}

function finishRename() {
  if (!editingSessionId.value) return;
  store.renameChatSession(editingSessionId.value, editingTitle.value);
  editingSessionId.value = null;
  editingTitle.value = "";
}

function deleteSession(sessionId: string) {
  store.deleteChatSession(sessionId);
}

async function onFilesSelected(event: Event) {
  const files = Array.from((event.target as HTMLInputElement).files ?? []);
  for (const file of files) {
    const content = await file.text();
    store.addPendingAttachment({
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      type: file.type,
      content,
    });
  }
  if (fileInput.value) fileInput.value.value = "";
}

function submit() {
  const text = input.value.trim();
  if (!text || !store.chatVariantId) return;
  input.value = "";

  if (text === "/tools") return dispatchRpc({ method: "tools/list" });
  const toolCmd = text.match(/^\/tool\s+(\S+)\s*(\{.*\})?$/s);
  if (toolCmd) {
    let args = {};
    try { args = toolCmd[2] ? JSON.parse(toolCmd[2]) : {}; } catch { args = {}; }
    return dispatchRpc({ method: "tools/call", params: { name: toolCmd[1], arguments: args } });
  }

  store.sendChat(text, sampling.maxTokens, sampling.temperature, sampling.topP);
}

function onComposerKeydown(event: KeyboardEvent) {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  submit();
}

function selectQuickStart(variantId: string, prompt: string) {
  store.chatVariantId = variantId;
  if (!store.activeChatSessionId) store.newChat(variantId);
  input.value = prompt;
  nextTick(() => textarea.value?.focus());
}

function parseDiagnostics(text: string, streaming = false): DiagnosticPart[] {
  const parts: DiagnosticPart[] = [];
  const pattern = /<call_mcp_tool\s+name="([^"]+)">([\s\S]*?)<\/call_mcp_tool>|<mcp_tool_response\s+tool="([^"]+)">([\s\S]*?)<\/mcp_tool_response>/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index == null) continue;
    if (match.index > cursor) parts.push({ kind: "text", text: text.slice(cursor, match.index) });
    if (match[1]) {
      const responseTag = `<mcp_tool_response tool="${match[1]}">`;
      parts.push({ kind: "tool_call", name: match[1], body: match[2].trim(), inFlight: streaming && !text.includes(responseTag) });
    } else {
      parts.push({ kind: "tool_response", name: match[3], body: match[4].trim() });
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) parts.push({ kind: "text", text: text.slice(cursor) });
  return parts.length ? parts : [{ kind: "text", text }];
}

function segmentsFor(text: string) {
  return segmentMessage(text);
}

function blockKey(messageIndex: number, partIndex: number) {
  return `${messageIndex}:${partIndex}`;
}

function isOpen(key: string) {
  return openBlocks[key] ?? true;
}

function toggleBlock(key: string) {
  openBlocks[key] = !isOpen(key);
}

function roleLabel(message: ChatMessage) {
  if (message.role === "assistant") return activeVariant.value?.name ?? message.variantId ?? "assistant";
  if (message.role === "system") return "system";
  if (message.role === "tool") return "tool";
  return "you";
}

function fileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
</script>

<template>
  <div class="chat-shell">
    <aside class="history-sidebar" :class="{ collapsed: !sidebarOpen }">
      <div class="sidebar-top">
        <button class="collapse-button" @click="sidebarOpen = !sidebarOpen">{{ sidebarOpen ? "‹" : "›" }}</button>
        <button v-if="sidebarOpen" class="new-chat-button" @click="newChat">+ New Chat</button>
      </div>

      <div v-if="sidebarOpen" class="session-list">
        <div
          v-for="session in store.chatSessions"
          :key="session.id"
          class="session-item"
          :class="{ active: session.id === store.activeChatSessionId }"
          @click="store.loadChatSession(session.id)"
          @dblclick.stop="beginRename(session.id, session.title)"
        >
          <span class="session-glyph">⌁</span>
          <input
            v-if="editingSessionId === session.id"
            v-model="editingTitle"
            class="session-rename"
            @click.stop
            @keydown.enter.prevent="finishRename"
            @blur="finishRename"
          />
          <span v-else class="session-title">{{ session.title }}</span>
          <button class="session-delete" @click.stop="deleteSession(session.id)">×</button>
        </div>

        <p v-if="!store.chatSessions.length" class="empty-history">No chat history yet.</p>
      </div>
    </aside>

    <main class="conversation-main">
      <header v-if="store.activeChatSessionId" class="conversation-header">
        <div>
          <p class="header-eyebrow">local inference session</p>
          <h2>{{ activeVariant?.name ?? "No model selected" }}</h2>
        </div>
        <div class="header-meta">
          <span>{{ contextWindow ? `${contextWindow} token context` : "context unknown" }}</span>
          <span>{{ store.runtime?.backend ?? "backend ?" }}</span>
          <label class="mcp-toggle">
            <input type="checkbox" :checked="store.mcpTools.length > 0" readonly />
            <span>{{ store.mcpTools.length }} stdio MCP tools</span>
          </label>
        </div>
      </header>

      <section v-if="!store.activeChatSessionId" class="empty-dashboard">
        <div class="empty-copy">
          <p class="header-eyebrow">chat terminal</p>
          <h1>Start a local model session</h1>
          <p>Select a model action, attach context, or open an existing chat from the history sidebar.</p>
        </div>
        <div class="quick-grid">
          <button v-for="card in quickStarts" :key="card.id" class="quick-card" @click="selectQuickStart(card.id, card.prompt)">
            <span>{{ card.title }}</span>
            <small>{{ card.meta }}</small>
          </button>
        </div>
      </section>

      <section ref="scroller" class="message-scroll" :class="{ empty: !store.activeChatSessionId }">
        <article v-for="(message, messageIndex) in store.chatMessages" :key="messageIndex" class="message-row" :class="message.role">
          <div class="message-card">
            <div class="message-label">{{ roleLabel(message) }}</div>

            <div v-if="message.attachments?.length" class="message-attachments">
              <span v-for="file in message.attachments" :key="file.id" class="attachment-chip">{{ file.name }} · {{ fileSize(file.size) }}</span>
            </div>

            <template v-for="(part, partIndex) in parseDiagnostics(message.text, message.streaming)" :key="partIndex">
              <div v-if="part.kind === 'text'" class="message-text">
                <template v-for="(segment, segmentIndex) in segmentsFor(part.text)" :key="segmentIndex">
                  <p v-if="segment.kind === 'text'" class="plain-text">{{ segment.content }}</p>
                  <div v-else class="code-block">
                    <div class="code-title">{{ segment.lang || "code" }}</div>
                    <pre v-html="highlight(segment.content)" />
                  </div>
                </template>
              </div>

              <div v-else-if="part.kind === 'tool_call'" class="tool-console">
                <button class="tool-console-head" @click="toggleBlock(blockKey(messageIndex, partIndex))">
                  <span class="spinner" :class="{ active: part.inFlight }" />
                  <span>call {{ part.name }}</span>
                  <small>{{ part.inFlight ? "running" : "ready" }}</small>
                </button>
                <pre v-if="isOpen(blockKey(messageIndex, partIndex))">{{ part.body }}</pre>
              </div>

              <div v-else class="tool-response">
                <button class="tool-console-head" @click="toggleBlock(blockKey(messageIndex, partIndex))">
                  <span>response {{ part.name }}</span>
                  <small>read-only</small>
                </button>
                <pre v-if="isOpen(blockKey(messageIndex, partIndex))">{{ part.body }}</pre>
              </div>
            </template>

            <span v-if="message.streaming" class="stream-caret" />
          </div>
        </article>
      </section>

      <form class="composer-dock" @submit.prevent="submit">
        <div v-if="store.pendingAttachments.length" class="pending-files">
          <span v-for="file in store.pendingAttachments" :key="file.id" class="file-pill">
            {{ file.name }} <small>{{ fileSize(file.size) }}</small>
            <button type="button" @click="store.removePendingAttachment(file.id)">×</button>
          </span>
        </div>

        <div class="composer-row">
          <input ref="fileInput" type="file" multiple class="hidden-input" @change="onFilesSelected" />
          <button type="button" class="icon-button" @click="fileInput?.click()">⌘</button>
          <textarea
            ref="textarea"
            v-model="input"
            rows="1"
            placeholder="Message the model..."
            @keydown="onComposerKeydown"
          />
          <button type="submit" class="send-button" :disabled="!input.trim() || !store.chatVariantId || !!store.activeChatId">
            {{ store.activeChatId ? "..." : "Send" }}
          </button>
          <button v-if="store.activeChatId" type="button" class="stop-button" @click="store.stopChat()">Stop</button>
        </div>

        <div class="composer-ribbon">
          <select v-model="store.chatVariantId">
            <option v-for="variant in store.library" :key="variant.id" :value="variant.id">{{ variant.name }}</option>
          </select>
          <span v-for="tool in store.mcpTools.slice(0, 8)" :key="tool.name" class="tool-pill">{{ tool.name }}</span>
          <span v-if="store.mcpTools.length > 8" class="tool-pill muted">+{{ store.mcpTools.length - 8 }}</span>
        </div>
      </form>
    </main>
  </div>
</template>

<style scoped>
.chat-shell {
  display: flex;
  min-height: 0;
  flex: 1;
  background: #0f172a;
  color: #e5edf8;
}

.history-sidebar {
  display: flex;
  width: 260px;
  flex-shrink: 0;
  flex-direction: column;
  border-right: 1px solid rgba(148, 163, 184, 0.14);
  background: #070b14;
  transition: width 160ms ease;
}

.history-sidebar.collapsed {
  width: 48px;
}

.sidebar-top {
  display: flex;
  gap: 8px;
  padding: 12px;
}

.collapse-button,
.new-chat-button,
.session-item,
.icon-button,
.send-button,
.stop-button,
.quick-card,
.tool-console-head {
  border: 0;
  color: inherit;
  cursor: pointer;
}

.collapse-button {
  width: 28px;
  border-radius: 7px;
  background: #111827;
  color: #94a3b8;
}

.new-chat-button {
  flex: 1;
  border-radius: 8px;
  background: #f8fafc;
  padding: 9px 12px;
  color: #0f172a;
  font-weight: 700;
}

.session-list {
  min-height: 0;
  flex: 1;
  overflow-y: auto;
  padding: 0 8px 12px;
}

.session-item {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 8px;
  border-radius: 8px;
  background: transparent;
  padding: 8px;
  text-align: left;
  color: #cbd5e1;
}

.session-item.active,
.session-item:hover {
  background: #111827;
}

.session-title {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}

.session-glyph,
.session-delete {
  color: #64748b;
}

.session-delete {
  border: 0;
  background: transparent;
  opacity: 0;
}

.session-item:hover .session-delete {
  opacity: 1;
}

.session-rename {
  min-width: 0;
  flex: 1;
  border: 1px solid #334155;
  border-radius: 6px;
  background: #020617;
  padding: 3px 6px;
  color: #f8fafc;
}

.empty-history {
  padding: 20px 10px;
  color: #64748b;
  font-size: 12px;
}

.conversation-main {
  position: relative;
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  overflow: hidden;
  background:
    radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.12), transparent 34rem),
    #0b1120;
}

.conversation-header {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  background: rgba(11, 17, 32, 0.9);
  padding: 14px 24px;
  backdrop-filter: blur(14px);
}

.header-eyebrow {
  margin: 0 0 4px;
  color: #64748b;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.conversation-header h2,
.empty-copy h1 {
  margin: 0;
  font-size: 18px;
}

.header-meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  color: #94a3b8;
  font-size: 12px;
}

.header-meta span,
.mcp-toggle,
.tool-pill,
.file-pill,
.attachment-chip {
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.86);
  padding: 4px 8px;
}

.mcp-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
}

.message-scroll {
  min-height: 0;
  flex: 1;
  overflow-y: auto;
  padding: 24px 24px 180px;
}

.message-scroll.empty {
  display: none;
}

.message-row {
  display: flex;
  margin: 0 auto 18px;
  max-width: 900px;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.assistant,
.message-row.system,
.message-row.tool {
  justify-content: flex-start;
}

.message-card {
  max-width: min(760px, 86%);
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 18px;
  background: rgba(15, 23, 42, 0.92);
  padding: 12px 14px;
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.18);
}

.user .message-card {
  border-color: rgba(96, 165, 250, 0.28);
  background: #1d4ed8;
}

.system .message-card,
.tool .message-card {
  background: rgba(3, 7, 18, 0.72);
}

.message-label {
  margin-bottom: 6px;
  color: #94a3b8;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.plain-text {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: #e2e8f0;
  font-size: 14px;
  line-height: 1.6;
}

.code-block,
.tool-console,
.tool-response {
  overflow: hidden;
  margin-top: 10px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 10px;
  background: #020617;
}

.code-title {
  border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  padding: 6px 10px;
  color: #94a3b8;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
}

pre {
  margin: 0;
  max-height: 340px;
  overflow: auto;
  padding: 12px;
  color: #c4b5fd;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  line-height: 1.55;
}

.tool-console-head {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 8px;
  background: #111827;
  padding: 8px 10px;
  text-align: left;
}

.tool-console-head small {
  margin-left: auto;
  color: #94a3b8;
}

.spinner {
  width: 10px;
  height: 10px;
  border: 2px solid #334155;
  border-top-color: #22c55e;
  border-radius: 999px;
}

.spinner.active {
  animation: spin 900ms linear infinite;
}

.stream-caret {
  display: inline-block;
  width: 8px;
  height: 18px;
  margin-top: 6px;
  background: #34d399;
  animation: blink 1s steps(2, start) infinite;
}

.empty-dashboard {
  display: grid;
  place-content: center;
  min-height: 100%;
  padding: 40px 24px 180px;
}

.empty-copy {
  margin: 0 auto 24px;
  max-width: 620px;
  text-align: center;
}

.empty-copy p:last-child {
  color: #94a3b8;
}

.quick-grid {
  display: grid;
  width: min(760px, calc(100vw - 320px));
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.quick-card {
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.82);
  padding: 14px;
  text-align: left;
}

.quick-card span {
  display: block;
  font-weight: 700;
}

.quick-card small {
  color: #94a3b8;
}

.composer-dock {
  position: absolute;
  right: 24px;
  bottom: 20px;
  left: 24px;
  z-index: 10;
  width: min(768px, calc(100% - 48px));
  margin: 0 auto;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 18px;
  background: rgba(2, 6, 23, 0.92);
  padding: 10px;
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
  backdrop-filter: blur(18px);
}

.composer-row {
  display: flex;
  align-items: flex-end;
  gap: 8px;
}

.composer-row textarea {
  min-height: 44px;
  max-height: 180px;
  flex: 1;
  resize: none;
  border: 0;
  background: transparent;
  padding: 12px 4px;
  color: #f8fafc;
  font-size: 14px;
  line-height: 1.5;
  outline: none;
}

.icon-button,
.send-button,
.stop-button {
  min-height: 38px;
  border-radius: 10px;
  padding: 0 12px;
  background: #1e293b;
}

.send-button {
  background: #f8fafc;
  color: #020617;
  font-weight: 800;
}

.send-button:disabled {
  cursor: default;
  opacity: 0.45;
}

.stop-button {
  background: #7f1d1d;
}

.composer-ribbon,
.pending-files,
.message-attachments {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.composer-ribbon {
  margin-top: 8px;
  align-items: center;
}

.composer-ribbon select {
  max-width: 220px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 999px;
  background: #0f172a;
  padding: 4px 8px;
  color: #cbd5e1;
  font-size: 11px;
}

.pending-files {
  margin-bottom: 6px;
}

.file-pill,
.attachment-chip,
.tool-pill {
  color: #cbd5e1;
  font-size: 11px;
}

.file-pill button {
  margin-left: 4px;
  border: 0;
  background: transparent;
  color: #94a3b8;
}

.tool-pill {
  color: #a7f3d0;
}

.tool-pill.muted {
  color: #94a3b8;
}

.hidden-input {
  display: none;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes blink {
  50% { opacity: 0; }
}

@media (max-width: 820px) {
  .history-sidebar {
    position: absolute;
    z-index: 20;
    height: 100%;
  }

  .history-sidebar.collapsed {
    position: relative;
  }

  .quick-grid {
    width: min(100%, 520px);
    grid-template-columns: 1fr;
  }

  .conversation-header {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>