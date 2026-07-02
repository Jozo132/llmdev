<script setup lang="ts">
import { onMounted } from "vue";
import { usePipelineStore } from "./stores/pipeline";
import NodeCanvas from "./components/NodeCanvas.vue";
import PropertyPanel from "./components/PropertyPanel.vue";
import MetricsBar from "./components/MetricsBar.vue";
import LogConsole from "./components/LogConsole.vue";

const store = usePipelineStore();
onMounted(() => store.connect());
</script>

<template>
  <div class="flex h-full flex-col">
    <!-- Top bar -->
    <header class="flex items-center gap-4 border-b border-slate-800 bg-panel px-4 py-2">
      <h1 class="text-sm font-bold tracking-widest text-slate-400">
        LLMDEV <span class="text-slate-600">/ node-graph playground</span>
      </h1>
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
          class="rounded bg-red-800 px-3 py-1 text-xs font-semibold hover:bg-red-700 disabled:opacity-40"
          :disabled="!store.state?.running"
          @click="store.stop()"
        >
          ■ Stop
        </button>
      </div>
    </header>

    <MetricsBar />

    <!-- Main: canvas + property panel -->
    <main class="flex min-h-0 flex-1">
      <NodeCanvas class="min-w-0 flex-1" />
      <PropertyPanel class="w-80 shrink-0 border-l border-slate-800 bg-panel" />
    </main>

    <LogConsole class="h-40 shrink-0 border-t border-slate-800" />

    <div
      v-if="store.lastError"
      class="fixed bottom-4 left-4 rounded bg-red-900/90 px-4 py-2 text-sm text-red-100 shadow-lg"
      @click="store.lastError = ''"
    >
      {{ store.lastError }} <span class="ml-2 cursor-pointer text-xs opacity-60">✕ dismiss</span>
    </div>
  </div>
</template>
