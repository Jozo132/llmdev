<script setup lang="ts">
import { onMounted, ref } from "vue";
import { usePipelineStore } from "./stores/pipeline";
import NodeCanvas from "./components/NodeCanvas.vue";
import PropertyPanel from "./components/PropertyPanel.vue";
import WorkspacePanel from "./components/WorkspacePanel.vue";
import ChatSandbox from "./components/ChatSandbox.vue";
import MetricsBar from "./components/MetricsBar.vue";
import LogConsole from "./components/LogConsole.vue";
import TrainingAnalyticsPanel from "./components/TrainingAnalyticsPanel.vue";

const store = usePipelineStore();
const tab = ref<"canvas" | "chat" | "analytics">("canvas");
onMounted(() => store.connect());
</script>

<template>
  <div class="flex h-full flex-col">
    <!-- Top bar -->
    <header class="flex items-center gap-4 border-b border-slate-800 bg-panel px-4 py-2">
      <h1 class="text-sm font-bold tracking-widest text-slate-400">
        LLMDEV <span class="text-slate-600">/ node-graph playground</span>
      </h1>

      <!-- tabs -->
      <nav class="flex rounded-lg bg-canvas p-0.5 text-xs">
        <button
          class="rounded-md px-3 py-1 font-semibold"
          :class="tab === 'canvas' ? 'bg-slate-700 text-slate-100' : 'text-slate-500 hover:text-slate-300'"
          @click="tab = 'canvas'"
        >⬡ Canvas</button>
        <button
          class="rounded-md px-3 py-1 font-semibold"
          :class="tab === 'chat' ? 'bg-slate-700 text-slate-100' : 'text-slate-500 hover:text-slate-300'"
          @click="tab = 'chat'"
        >💬 Chat Sandbox</button>
        <button
          class="rounded-md px-3 py-1 font-semibold"
          :class="tab === 'analytics' ? 'bg-slate-700 text-slate-100' : 'text-slate-500 hover:text-slate-300'"
          @click="tab = 'analytics'"
        >📈 Analytics</button>
      </nav>

      <span
        class="rounded-full px-2 py-0.5 text-xs"
        :class="store.connected ? 'bg-emerald-900 text-emerald-300' : 'bg-red-900 text-red-300'"
      >
        {{ store.connected ? "● bridge connected" : "○ reconnecting…" }}
      </span>
      <span v-if="store.state" class="text-xs text-slate-500">{{ store.state.name }}</span>
      <div class="ml-auto flex gap-2">
        <button
          class="rounded bg-emerald-700 px-3 py-1 text-xs font-semibold hover:bg-emerald-600 disabled:opacity-40"
          :disabled="!store.connected || store.state?.running"
          @click="store.run()"
        >
          ▶ Run pipeline
        </button>
        <button
          v-if="store.state?.running && !store.state?.paused"
          class="rounded bg-amber-700 px-3 py-1 text-xs font-semibold hover:bg-amber-600"
          title="Suspend between steps — weights & optimizer moments stay hot in memory"
          @click="store.pauseTraining()"
        >
          ⏸ Pause
        </button>
        <button
          v-if="store.state?.paused"
          class="rounded bg-sky-700 px-3 py-1 text-xs font-semibold hover:bg-sky-600"
          @click="store.resumeTraining()"
        >
          ⏵ Resume
        </button>
        <button
          v-if="store.state?.running"
          class="rounded bg-emerald-600 px-3 py-1 text-xs font-bold text-emerald-50 hover:bg-emerald-500"
          title="Commit & Proceed — stop iterating now, freeze weights + Adam moments, finish the node as done and fire downstream steps"
          @click="store.commitTraining()"
        >
          ✔ Commit &amp; Proceed
        </button>
        <button
          class="rounded bg-red-800 px-3 py-1 text-xs font-semibold hover:bg-red-700 disabled:opacity-40"
          :disabled="!store.state?.running"
          title="Abort the run and release the GPU context"
          @click="store.cancelTraining()"
        >
          ■ Cancel
        </button>
        <span
          v-if="store.state?.paused"
          class="rounded-full bg-amber-950 px-2 py-1 text-xs text-amber-300"
        >paused — state preserved</span>
      </div>
    </header>

    <MetricsBar v-show="tab === 'canvas'" />

    <!-- Canvas tab: library | SVG playground | properties -->
    <main v-show="tab === 'canvas'" class="flex min-h-0 flex-1">
      <WorkspacePanel class="w-72 shrink-0" />
      <NodeCanvas class="min-w-0 flex-1" />
      <PropertyPanel class="w-80 shrink-0 border-l border-slate-800 bg-panel" />
    </main>

    <!-- Chat tab: immersive widescreen — canvas & config panels hidden -->
    <main v-show="tab === 'chat'" class="flex min-h-0 flex-1">
      <ChatSandbox />
    </main>

    <!-- Analytics tab: widescreen historic charts + runtime controls -->
    <main v-show="tab === 'analytics'" class="flex min-h-0 flex-1">
      <TrainingAnalyticsPanel />
    </main>

    <LogConsole v-show="tab === 'canvas'" class="h-40 shrink-0 border-t border-slate-800" />

    <div
      v-if="store.lastError"
      class="fixed bottom-4 left-4 z-30 rounded bg-red-900/90 px-4 py-2 text-sm text-red-100 shadow-lg"
      @click="store.lastError = ''"
    >
      {{ store.lastError }} <span class="ml-2 cursor-pointer text-xs opacity-60">✕ dismiss</span>
    </div>
  </div>
</template>
