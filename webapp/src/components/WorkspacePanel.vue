<script setup lang="ts">
/**
 * WorkspacePanel — the model version library.
 * Lists local checkpoints & cloned variants; "Clone & Modify" forks any entry
 * (symlinked weights, isolated hyperparameters), variants train concurrently
 * and cross-benchmark on live loss sparklines.
 */
import { computed, reactive, ref } from "vue";
import { usePipelineStore } from "../stores/pipeline";
import type { ModelVariant } from "../types";

const store = usePipelineStore();

const cloneSource = ref<ModelVariant | null>(null);
const cloneForm = reactive({ name: "", mixer: "causal-mean", loss: "cross-entropy", dModel: 0 });

function openClone(v: ModelVariant) {
  cloneSource.value = v;
  cloneForm.name = `${v.name}-fork`;
  cloneForm.mixer = v.config.mixer;
  cloneForm.loss = v.config.loss;
  cloneForm.dModel = v.config.dModel;
}

function submitClone() {
  if (!cloneSource.value) return;
  store.cloneVariant(cloneSource.value.id, cloneForm.name, {
    mixer: cloneForm.mixer,
    loss: cloneForm.loss,
    dModel: cloneForm.dModel,
  });
  cloneSource.value = null;
}

const SPARK_COLORS = ["#f59e0b", "#10b981", "#0ea5e9", "#ec4899", "#8b5cf6", "#ef4444"];
const colorFor = (id: string) =>
  SPARK_COLORS[store.library.findIndex((v) => v.id === id) % SPARK_COLORS.length];

function sparkPath(id: string): string {
  const live = store.variantMetrics[id]?.map((m) => m.loss) ?? [];
  const hist = store.library.find((v) => v.id === id)?.history.map((h) => h.loss) ?? [];
  const vals = live.length ? live : hist.slice(-60);
  if (vals.length < 2) return "";
  const w = 120, h = 24;
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  return vals
    .map((v, i) => `${i === 0 ? "M" : "L"} ${(i / (vals.length - 1)) * w} ${h - ((v - min) / span) * h}`)
    .join(" ");
}

const latestLoss = computed(() => (id: string) => {
  const arr = store.variantMetrics[id];
  return arr?.length ? arr[arr.length - 1].loss.toFixed(3) : null;
});
</script>

<template>
  <aside class="flex flex-col overflow-y-auto border-r border-slate-800 bg-panel p-3">
    <h2 class="mb-2 text-xs font-bold uppercase tracking-widest text-slate-500">
      Model library
    </h2>

    <div v-if="!store.library.length" class="text-xs text-slate-600">
      No checkpoints yet — run the pipeline once to export a base model.
    </div>

    <div
      v-for="v in store.library"
      :key="v.id"
      class="mb-2 rounded border border-slate-800 p-2"
      :style="{ borderLeftColor: colorFor(v.id), borderLeftWidth: '3px' }"
    >
      <div class="flex items-center justify-between">
        <p class="text-xs font-semibold text-slate-200">{{ v.name }}</p>
        <span
          class="rounded px-1.5 text-[9px] uppercase"
          :class="v.source === 'export' ? 'bg-slate-800 text-slate-400' : 'bg-violet-950 text-violet-300'"
        >{{ v.source }}</span>
      </div>
      <p class="font-mono text-[10px] text-slate-500">
        {{ (v.paramCount / 1e6).toFixed(2) }}M · {{ v.config.mixer }} · {{ v.config.loss }}
        <span v-if="v.finalLoss"> · loss {{ v.finalLoss.toFixed(3) }}</span>
      </p>

      <!-- live cross-benchmark sparkline -->
      <div class="mt-1 flex items-center gap-2">
        <svg width="120" height="24">
          <path :d="sparkPath(v.id)" fill="none" :stroke="colorFor(v.id)" stroke-width="1.5" />
        </svg>
        <span v-if="latestLoss(v.id)" class="font-mono text-[10px]" :style="{ color: colorFor(v.id) }">
          {{ latestLoss(v.id) }}
        </span>
        <span v-if="v.training" class="animate-pulse text-[10px] text-amber-400">training…</span>
      </div>

      <div class="mt-1.5 flex gap-1.5">
        <button
          class="rounded bg-slate-800 px-2 py-0.5 text-[10px] hover:bg-slate-700"
          @click="openClone(v)"
        >Clone & Modify</button>
        <button
          v-if="v.source === 'clone' && !v.training"
          class="rounded bg-emerald-900 px-2 py-0.5 text-[10px] text-emerald-200 hover:bg-emerald-800"
          @click="store.trainVariant(v.id)"
        >▶ Train</button>
        <button
          v-if="v.training"
          class="rounded bg-red-900 px-2 py-0.5 text-[10px] text-red-200 hover:bg-red-800"
          @click="store.stopVariantTraining(v.id)"
        >■ Stop</button>
      </div>
    </div>

    <!-- Clone & Modify dialog -->
    <div
      v-if="cloneSource"
      class="fixed inset-0 z-20 flex items-center justify-center bg-black/60"
      @click.self="cloneSource = null"
    >
      <div class="w-80 rounded-lg border border-slate-700 bg-panel p-4">
        <h3 class="mb-3 text-sm font-bold">Clone “{{ cloneSource.name }}”</h3>
        <label class="mb-1 block text-xs text-slate-400">Variant name</label>
        <input v-model="cloneForm.name" class="mb-2 w-full rounded border border-slate-700 bg-canvas px-2 py-1 text-sm" />
        <label class="mb-1 block text-xs text-slate-400">Mixer (attention slot)</label>
        <input v-model="cloneForm.mixer" class="mb-2 w-full rounded border border-slate-700 bg-canvas px-2 py-1 font-mono text-sm"
               placeholder="causal-mean | your registered mixer" />
        <label class="mb-1 block text-xs text-slate-400">Loss fn</label>
        <input v-model="cloneForm.loss" class="mb-2 w-full rounded border border-slate-700 bg-canvas px-2 py-1 font-mono text-sm" />
        <label class="mb-1 block text-xs text-slate-400">dModel (change ⇒ fresh init)</label>
        <input v-model.number="cloneForm.dModel" type="number" class="mb-3 w-full rounded border border-slate-700 bg-canvas px-2 py-1 text-sm" />
        <div class="flex justify-end gap-2">
          <button class="rounded px-3 py-1 text-xs text-slate-400 hover:bg-slate-800" @click="cloneSource = null">Cancel</button>
          <button class="rounded bg-violet-700 px-3 py-1 text-xs font-semibold hover:bg-violet-600" @click="submitClone">Create variant</button>
        </div>
      </div>
    </div>
  </aside>
</template>
