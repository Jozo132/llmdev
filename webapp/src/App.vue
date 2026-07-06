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

const fmtLoss = (loss: number | null): string => loss == null ? "—" : loss.toFixed(4);
const fmtDelta = (delta: number | null): string => {
  if (delta == null) return "—";
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(4)}`;
};
const deltaClass = (delta: number | null): string =>
  delta == null ? "text-slate-400" : delta <= 0 ? "text-emerald-300" : "text-orange-300";
const changed = (tech: string, other: string[]): boolean => !other.includes(tech);
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

    <div
      v-if="store.pendingCommitConfirmation"
      class="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-4"
    >
      <div class="w-full max-w-xl rounded-lg border border-amber-700 bg-panel p-4 shadow-2xl">
        <h3 class="text-sm font-bold text-amber-300">Commit current training state?</h3>
        <p class="mt-1 text-xs text-slate-400">
          This stops training early and lets downstream evaluation/export continue. Final library weights will still require acceptance after the pipeline finishes.
        </p>
        <div class="mt-3 grid grid-cols-3 gap-2 text-xs">
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">previous train</p>
            <p class="font-mono text-slate-200">{{ fmtLoss(store.pendingCommitConfirmation.oldTrainLoss) }}</p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">current train</p>
            <p class="font-mono text-slate-200">{{ fmtLoss(store.pendingCommitConfirmation.newTrainLoss) }}</p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">delta</p>
            <p class="font-mono" :class="deltaClass(store.pendingCommitConfirmation.trainLossDelta)">
              {{ fmtDelta(store.pendingCommitConfirmation.trainLossDelta) }}
            </p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">previous dataset</p>
            <p class="font-mono text-slate-200">{{ fmtLoss(store.pendingCommitConfirmation.oldDatasetLoss) }}</p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">current dataset</p>
            <p class="font-mono text-slate-200">{{ fmtLoss(store.pendingCommitConfirmation.newDatasetLoss) }}</p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">delta</p>
            <p class="font-mono" :class="deltaClass(store.pendingCommitConfirmation.datasetLossDelta)">
              {{ fmtDelta(store.pendingCommitConfirmation.datasetLossDelta) }}
            </p>
          </div>
        </div>
        <div class="mt-3 grid grid-cols-2 gap-3 text-xs">
          <div>
            <p class="mb-1 text-[10px] uppercase tracking-widest text-slate-500">old stack</p>
            <span v-for="tech in store.pendingCommitConfirmation.oldTechStack" :key="tech" class="mr-1 inline-block rounded border border-slate-700 px-1.5 py-0.5 font-mono text-slate-400">{{ tech }}</span>
          </div>
          <div>
            <p class="mb-1 text-[10px] uppercase tracking-widest text-slate-500">new stack</p>
            <span
              v-for="tech in store.pendingCommitConfirmation.newTechStack"
              :key="tech"
              class="mr-1 inline-block rounded border px-1.5 py-0.5 font-mono"
              :class="changed(tech, store.pendingCommitConfirmation.oldTechStack) ? 'border-amber-600 text-amber-300' : 'border-slate-700 text-slate-400'"
            >{{ tech }}</span>
          </div>
        </div>
        <div class="mt-4 flex justify-end gap-2">
          <button class="rounded px-3 py-1 text-xs text-slate-400 hover:bg-slate-800" @click="store.cancelCommitTraining()">Keep training</button>
          <button class="rounded bg-emerald-700 px-3 py-1 text-xs font-semibold text-emerald-50 hover:bg-emerald-600" @click="store.confirmCommitTraining()">Commit & proceed</button>
        </div>
      </div>
    </div>

    <div
      v-if="store.pendingModelAcceptance"
      class="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-4"
    >
      <div class="w-full max-w-xl rounded-lg border border-emerald-700 bg-panel p-4 shadow-2xl">
        <h3 class="text-sm font-bold text-emerald-300">Accept trained model for “{{ store.pendingModelAcceptance.modelName }}”?</h3>
        <p class="mt-1 text-xs text-slate-400">
          Accept writes the new weights to the selected model. Reject leaves the previous model state on disk.
        </p>
        <div class="mt-3 grid grid-cols-3 gap-2 text-xs">
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">previous train</p>
            <p class="font-mono text-slate-200">{{ fmtLoss(store.pendingModelAcceptance.oldTrainLoss) }}</p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">new train</p>
            <p class="font-mono text-slate-200">{{ fmtLoss(store.pendingModelAcceptance.newTrainLoss) }}</p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">delta</p>
            <p class="font-mono" :class="deltaClass(store.pendingModelAcceptance.trainLossDelta)">
              {{ fmtDelta(store.pendingModelAcceptance.trainLossDelta) }}
            </p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">previous dataset</p>
            <p class="font-mono text-slate-200">{{ fmtLoss(store.pendingModelAcceptance.oldDatasetLoss) }}</p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">new dataset</p>
            <p class="font-mono text-slate-200">{{ fmtLoss(store.pendingModelAcceptance.newDatasetLoss) }}</p>
          </div>
          <div class="rounded border border-slate-800 bg-canvas p-2">
            <p class="text-[10px] uppercase tracking-widest text-slate-500">delta</p>
            <p class="font-mono" :class="deltaClass(store.pendingModelAcceptance.datasetLossDelta)">
              {{ fmtDelta(store.pendingModelAcceptance.datasetLossDelta) }}
            </p>
          </div>
        </div>
        <div class="mt-3 grid grid-cols-2 gap-3 text-xs">
          <div>
            <p class="mb-1 text-[10px] uppercase tracking-widest text-slate-500">old stack</p>
            <span v-for="tech in store.pendingModelAcceptance.oldTechStack" :key="tech" class="mr-1 inline-block rounded border border-slate-700 px-1.5 py-0.5 font-mono text-slate-400">{{ tech }}</span>
          </div>
          <div>
            <p class="mb-1 text-[10px] uppercase tracking-widest text-slate-500">new stack</p>
            <span
              v-for="tech in store.pendingModelAcceptance.newTechStack"
              :key="tech"
              class="mr-1 inline-block rounded border px-1.5 py-0.5 font-mono"
              :class="changed(tech, store.pendingModelAcceptance.oldTechStack) ? 'border-amber-600 text-amber-300' : 'border-slate-700 text-slate-400'"
            >{{ tech }}</span>
          </div>
        </div>
        <div class="mt-4 flex justify-end gap-2">
          <button class="rounded bg-red-900 px-3 py-1 text-xs font-semibold text-red-100 hover:bg-red-800" @click="store.rejectModelResult(store.pendingModelAcceptance.id)">Reject</button>
          <button class="rounded bg-emerald-700 px-3 py-1 text-xs font-semibold text-emerald-50 hover:bg-emerald-600" @click="store.acceptModelResult(store.pendingModelAcceptance.id)">Accept model</button>
        </div>
      </div>
    </div>
  </div>
</template>
