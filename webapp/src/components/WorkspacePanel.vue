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

// ── Create / Rename / Delete ─────────────────────────────────────────
const showCreate = ref(false);
const createName = ref("");
const renameTarget = ref<ModelVariant | null>(null);
const renameName = ref("");
const deleteTarget = ref<ModelVariant | null>(null);

function submitCreate() {
  if (!createName.value.trim()) return;
  store.createVariant(createName.value.trim());
  createName.value = "";
  showCreate.value = false;
}

function openRename(v: ModelVariant) {
  renameTarget.value = v;
  renameName.value = v.name;
}

function submitRename() {
  if (!renameTarget.value || !renameName.value.trim()) return;
  store.renameVariant(renameTarget.value.id, renameName.value.trim());
  renameTarget.value = null;
}

function submitDelete() {
  if (!deleteTarget.value) return;
  store.deleteVariant(deleteTarget.value.id);
  deleteTarget.value = null;
}

const pendingOpenName = computed(
  () => store.library.find((v) => v.id === store.pendingOpenId)?.name ?? store.pendingOpenId
);

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
  const live = arr?.length ? arr[arr.length - 1].loss : null;
  const finalLoss = store.library.find((v) => v.id === id)?.finalLoss ?? null;
  const loss = live ?? finalLoss;
  return loss != null ? loss.toFixed(3) : null;
});

function techStack(v: ModelVariant): string[] {
  const cfg = v.config as unknown as Record<string, unknown>;
  const loraOn = cfg.fineTuneMode === "lora";
  const esOn = Boolean(cfg.stochasticExplorationPool);
  const mlp = String(cfg.mlp ?? "standard");
  const stack = [
    store.runtime?.backend ?? "backend ?",
    String(cfg.mixer ?? "causal-mean"),
    mlp === "swiglu" ? "SwiGLU" : "MLP",
    String(cfg.loss ?? "cross-entropy"),
    loraOn ? "LoRA" : "Full Adam",
  ];
  if (esOn) stack.push(loraOn ? "ES population" : "ES temporal");
  return stack;
}
</script>

<template>
  <aside class="flex flex-col overflow-y-auto border-r border-slate-800 bg-panel p-3">
    <div class="mb-2 flex items-center justify-between">
      <h2 class="text-xs font-bold uppercase tracking-widest text-slate-500">
        Model library
      </h2>
      <button
        class="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300 hover:bg-slate-700"
        @click="showCreate = !showCreate"
      >＋ New</button>
    </div>

    <div v-if="showCreate" class="mb-2 rounded border border-slate-700 p-2">
      <input
        v-model="createName"
        placeholder="model name"
        class="mb-1.5 w-full rounded border border-slate-700 bg-canvas px-2 py-1 text-xs"
        @keydown.enter="submitCreate"
      />
      <button
        class="w-full rounded bg-violet-700 px-2 py-1 text-[10px] font-semibold hover:bg-violet-600"
        @click="submitCreate"
      >Create blank model</button>
    </div>

    <div v-if="!store.library.length" class="text-xs text-slate-600">
      No checkpoints yet — run the pipeline once to export a base model.
    </div>

    <div
      v-for="v in store.library"
      :key="v.id"
      class="mb-2 cursor-pointer rounded border p-2"
      :class="v.id === store.chatVariantId ? 'border-amber-600 bg-slate-900' : 'border-slate-800 hover:border-slate-600'"
      :style="{ borderLeftColor: colorFor(v.id), borderLeftWidth: '3px' }"
      @click="store.openVariant(v.id)"
    >
      <div class="flex items-center justify-between">
        <p class="text-xs font-semibold text-slate-200">
          {{ v.name }}
          <span v-if="v.id === store.chatVariantId" class="ml-1 text-[9px] text-amber-400">● active</span>
        </p>
        <span
          class="rounded px-1.5 text-[9px] uppercase"
          :class="v.source === 'export' ? 'bg-slate-800 text-slate-400' : 'bg-violet-950 text-violet-300'"
        >{{ v.source }}</span>
      </div>
      <div class="mt-1 grid grid-cols-2 gap-1">
        <div class="rounded border border-slate-800 bg-canvas px-1.5 py-1">
          <p class="text-[8px] uppercase tracking-widest text-slate-600">size</p>
          <p class="font-mono text-[11px] text-slate-300">{{ (v.paramCount / 1e6).toFixed(2) }}M</p>
        </div>
        <div class="rounded border border-slate-800 bg-canvas px-1.5 py-1">
          <p class="text-[8px] uppercase tracking-widest text-slate-600">loss</p>
          <p class="font-mono text-[11px] text-amber-300">{{ latestLoss(v.id) ?? "—" }}</p>
        </div>
      </div>
      <div class="mt-1 flex flex-wrap gap-1">
        <span
          v-for="tech in techStack(v)"
          :key="tech"
          class="rounded border border-slate-800 bg-slate-950 px-1.5 py-0.5 font-mono text-[9px] text-slate-400"
        >{{ tech }}</span>
      </div>

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
          @click.stop="openClone(v)"
        >Clone & Modify</button>
        <button
          v-if="!v.training"
          class="rounded bg-slate-800 px-2 py-0.5 text-[10px] hover:bg-slate-700"
          title="Rename"
          @click.stop="openRename(v)"
        >✎</button>
        <button
          v-if="!v.training"
          class="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-red-300 hover:bg-red-900"
          title="Delete (removes files from disk)"
          @click.stop="deleteTarget = v"
        >🗑</button>
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
    <!-- Rename dialog -->
    <div
      v-if="renameTarget"
      class="fixed inset-0 z-20 flex items-center justify-center bg-black/60"
      @click.self="renameTarget = null"
    >
      <div class="w-72 rounded-lg border border-slate-700 bg-panel p-4">
        <h3 class="mb-3 text-sm font-bold">Rename “{{ renameTarget.name }}”</h3>
        <input
          v-model="renameName"
          class="mb-3 w-full rounded border border-slate-700 bg-canvas px-2 py-1 text-sm"
          @keydown.enter="submitRename"
        />
        <div class="flex justify-end gap-2">
          <button class="rounded px-3 py-1 text-xs text-slate-400 hover:bg-slate-800" @click="renameTarget = null">Cancel</button>
          <button class="rounded bg-violet-700 px-3 py-1 text-xs font-semibold hover:bg-violet-600" @click="submitRename">Rename</button>
        </div>
      </div>
    </div>

    <!-- Delete confirmation -->
    <div
      v-if="deleteTarget"
      class="fixed inset-0 z-20 flex items-center justify-center bg-black/60"
      @click.self="deleteTarget = null"
    >
      <div class="w-80 rounded-lg border border-red-900 bg-panel p-4">
        <h3 class="mb-2 text-sm font-bold text-red-300">Delete “{{ deleteTarget.name }}”?</h3>
        <p class="mb-3 text-xs text-slate-400">
          Weights, tokenizer and metadata are permanently removed from disk.
          Symlinked clones are materialized first so they keep working.
        </p>
        <div class="flex justify-end gap-2">
          <button class="rounded px-3 py-1 text-xs text-slate-400 hover:bg-slate-800" @click="deleteTarget = null">Cancel</button>
          <button class="rounded bg-red-800 px-3 py-1 text-xs font-semibold text-red-100 hover:bg-red-700" @click="submitDelete">Delete from disk</button>
        </div>
      </div>
    </div>

    <!-- Active-process blocker: force-cancel confirmation before switching -->
    <div
      v-if="store.pendingOpenId"
      class="fixed inset-0 z-30 flex items-center justify-center bg-black/70"
    >
      <div class="w-96 rounded-lg border border-amber-700 bg-panel p-4">
        <h3 class="mb-2 text-sm font-bold text-amber-300">⚠ Active process running</h3>
        <p class="mb-3 text-xs text-slate-300">
          Cannot open “{{ pendingOpenName }}” — {{ store.busyReason }}.
          Force-cancelling terminates worker threads and destroys the active
          memory context (unsaved training progress is lost).
        </p>
        <div class="flex justify-end gap-2">
          <button
            class="rounded px-3 py-1 text-xs text-slate-400 hover:bg-slate-800"
            @click="store.cancelPendingOpen()"
          >Keep running</button>
          <button
            class="rounded bg-red-800 px-3 py-1 text-xs font-semibold text-red-100 hover:bg-red-700"
            @click="store.confirmForceOpen()"
          >Force cancel & open</button>
        </div>
      </div>
    </div>
  </aside>
</template>
