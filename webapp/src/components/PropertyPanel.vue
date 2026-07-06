<script setup lang="ts">
/**
 * PropertyPanel — schema-driven editor for the selected node with baked-in
 * educational theory: each node shows its architectural function and every
 * parameter carries a ⓘ popover explaining its mathematical effect and a
 * strict safe operating range.
 */
import { computed, ref } from "vue";
import { usePipelineStore } from "../stores/pipeline";
import type { ParamSchemaEntry } from "../types";

const store = usePipelineStore();
const openTheory = ref<string | null>(null); // param key with expanded theory

function coerce(entry: ParamSchemaEntry, raw: string | boolean): unknown {
  if (entry.type === "number") return Number(raw);
  if (entry.type === "boolean") return Boolean(raw);
  return raw;
}

function onInput(entry: ParamSchemaEntry, ev: Event) {
  const t = ev.target as HTMLInputElement | HTMLSelectElement;
  const value = entry.type === "boolean" ? (t as HTMLInputElement).checked : t.value;
  if (store.selectedNodeId) {
    store.updateParams(store.selectedNodeId, { [entry.key]: coerce(entry, value) });
  }
}

function onResetModel() {
  const node = store.selectedNode;
  if (!node) return;
  const ok = window.confirm(
    "Reset this trainer's saved model weights and optimizer state? The next Run pipeline will start from fresh initialization."
  );
  if (ok) store.resetModel(node.id);
}

function matchesCondition(actual: unknown, expected: unknown): boolean {
  if (typeof expected === "string" && expected.startsWith("/") && expected.endsWith("/")) {
    return new RegExp(expected.slice(1, -1), "i").test(String(actual ?? ""));
  }
  return actual === expected;
}

function isVisible(entry: ParamSchemaEntry): boolean {
  const rules = entry.visibleWhen;
  if (!rules || !store.selectedNode) return true;
  return Object.entries(rules).every(([key, expected]) => {
    const actual = store.selectedNode?.params[key] ??
      store.selectedDescriptor?.paramSchema.find((candidate) => candidate.key === key)?.default;
    return matchesCondition(actual, expected);
  });
}

const visibleParams = computed(() => store.selectedDescriptor?.paramSchema.filter(isVisible) ?? []);

const executionSummary = computed(() => {
  const node = store.selectedNode;
  if (!node || node.type !== "model.architecture") return null;
  const params = node.params;
  const fineTuneMode = String(params.fineTuneMode ?? "Full Parameter");
  const loraOn = /lora/i.test(fineTuneMode);
  const stochasticOn = Boolean(params.stochasticExplorationPool);
  const populationSize = loraOn && stochasticOn ? Math.max(1, Math.round(Number(params.populationSize ?? 4))) : 1;
  const survivalRate = loraOn && stochasticOn ? Math.min(100, Math.max(1, Number(params.survivalRate ?? 25))) : 100;
  const survivalCount = loraOn && stochasticOn
    ? Math.max(1, Math.min(populationSize, Math.ceil(populationSize * survivalRate / 100)))
    : 1;
  return {
    method: loraOn ? "LoRA fine-tuning" : "Full-parameter training",
    methodTone: loraOn ? "text-cyan-300" : "text-amber-300",
    trainable: loraOn
      ? `Adapters only · r=${Number(params.loraRank ?? 8)} · alpha=${Number(params.loraAlpha ?? 16)}`
      : "All base weights trainable",
    exploration: !stochasticOn
      ? "Stochastic exploration off"
      : loraOn
        ? `LoRA ES population · P=${populationSize} · survivors=${survivalRate}% (N=${survivalCount})`
        : "Full-parameter ES temporal candidate · P=1",
    sigma: stochasticOn ? Number(params.stochasticMutationSigma ?? 0.002) : null,
    lockedPopulation: stochasticOn && !loraOn,
    technologies: [
      store.runtime?.backend ?? "backend ?",
      String(params.mixer ?? "causal-mean"),
      String(params.mlp ?? "standard") === "swiglu" ? "SwiGLU" : "MLP",
      String(params.loss ?? "cross-entropy"),
      loraOn ? "LoRA" : "Full Adam",
      ...(stochasticOn ? [loraOn ? "ES population" : "ES temporal"] : []),
    ],
  };
});
</script>

<template>
  <aside class="flex flex-col overflow-y-auto p-4">
    <template v-if="store.selectedNode && store.selectedDescriptor">
      <div class="flex items-start justify-between">
        <div>
          <h2 class="text-sm font-bold text-slate-200">{{ store.selectedNode.label }}</h2>
          <p class="mb-2 font-mono text-[11px] text-slate-500">
            {{ store.selectedNode.id }} · {{ store.selectedNode.type }}
          </p>
        </div>
        <button
          class="rounded bg-red-950 px-2 py-0.5 text-[10px] text-red-300 hover:bg-red-900"
          :disabled="store.state?.running"
          title="Remove this node and its connections"
          @click="store.removeNode(store.selectedNode.id)"
        >✕ delete</button>
      </div>

      <!-- Architectural theory for the node itself -->
      <details v-if="store.selectedDescriptor.theory" class="mb-4 rounded border border-slate-800 bg-canvas/50 p-2" open>
        <summary class="cursor-pointer text-[10px] font-bold uppercase tracking-widest text-sky-400">
          ⓘ Architecture theory
        </summary>
        <p class="mt-1 text-[11px] leading-relaxed text-slate-400">
          {{ store.selectedDescriptor.theory }}
        </p>
      </details>

      <div v-if="executionSummary" class="mb-4 rounded border border-slate-800 bg-panel p-3">
        <p class="text-[10px] font-bold uppercase tracking-widest text-slate-500">Active execution</p>
        <div class="mt-2 flex flex-wrap gap-2">
          <span class="rounded border border-slate-700 bg-canvas px-2 py-1 text-[11px] font-semibold" :class="executionSummary.methodTone">
            {{ executionSummary.method }}
          </span>
          <span class="rounded border border-slate-700 bg-canvas px-2 py-1 text-[11px] text-slate-300">
            {{ executionSummary.exploration }}
          </span>
        </div>
        <p class="mt-2 font-mono text-[11px] text-slate-400">{{ executionSummary.trainable }}</p>
        <p v-if="executionSummary.sigma !== null" class="mt-1 font-mono text-[11px] text-orange-300">
          sigma={{ executionSummary.sigma }}
        </p>
        <div class="mt-2 flex flex-wrap gap-1">
          <span
            v-for="tech in executionSummary.technologies"
            :key="tech"
            class="rounded border border-slate-700 bg-canvas px-1.5 py-0.5 font-mono text-[10px] text-slate-300"
          >{{ tech }}</span>
        </div>
        <p v-if="executionSummary.lockedPopulation" class="mt-2 text-[11px] leading-relaxed text-slate-500">
          Full-parameter ES is intentionally locked to one temporal candidate to avoid duplicating the whole model in VRAM. Switch Fine-Tuning Mode to LoRA to select population size and survivor rate.
        </p>
      </div>

      <div v-if="store.selectedNode.type === 'train.poc'" class="mb-4 rounded border border-red-950 bg-red-950/20 p-3">
        <p class="text-[10px] font-bold uppercase tracking-widest text-red-300">Model state</p>
        <p class="mt-1 text-[11px] leading-relaxed text-slate-400">
          Run pipeline resumes this trainer's saved weights by default. Reset only when you want a fresh initialization.
        </p>
        <button
          class="mt-2 rounded bg-red-900 px-3 py-1.5 text-xs font-semibold text-red-100 hover:bg-red-800 disabled:opacity-40"
          :disabled="store.state?.running"
          @click="onResetModel"
        >Reset model</button>
      </div>

      <div v-for="entry in visibleParams" :key="entry.key" class="mb-3">
        <div class="mb-1 flex items-center gap-1">
          <label class="block text-xs font-medium text-slate-400">{{ entry.label }}</label>
          <button
            v-if="entry.theory || entry.range"
            class="flex h-4 w-4 items-center justify-center rounded-full text-[9px]"
            :class="openTheory === entry.key ? 'bg-sky-700 text-white' : 'bg-slate-800 text-sky-400 hover:bg-slate-700'"
            :title="entry.theory"
            @click="openTheory = openTheory === entry.key ? null : entry.key"
          >i</button>
        </div>

        <!-- expanded hyperparameter theory -->
        <div
          v-if="openTheory === entry.key"
          class="mb-2 rounded border border-sky-900 bg-sky-950/40 p-2 text-[11px] leading-relaxed text-slate-300"
        >
          <p v-if="entry.theory">{{ entry.theory }}</p>
          <p v-if="entry.range" class="mt-1 font-mono text-[10px] text-amber-300">
            safe range: {{ entry.range }}
          </p>
        </div>

        <select
          v-if="entry.type === 'select'"
          class="w-full rounded border border-slate-700 bg-canvas px-2 py-1.5 text-sm"
          :value="String(store.selectedNode.params[entry.key] ?? entry.default ?? '')"
          :disabled="store.state?.running"
          @change="onInput(entry, $event)"
        >
          <option v-for="opt in entry.options" :key="opt" :value="opt">{{ opt }}</option>
        </select>

        <input
          v-else-if="entry.type === 'boolean'"
          type="checkbox"
          class="h-4 w-4 accent-emerald-500"
          :checked="Boolean(store.selectedNode.params[entry.key])"
          :disabled="store.state?.running"
          @change="onInput(entry, $event)"
        />

        <input
          v-else
          :type="entry.type === 'number' ? 'number' : 'text'"
          step="any"
          class="w-full rounded border border-slate-700 bg-canvas px-2 py-1.5 text-sm"
          :value="String(store.selectedNode.params[entry.key] ?? entry.default ?? '')"
          :disabled="store.state?.running"
          @change="onInput(entry, $event)"
        />

        <p v-if="entry.description" class="mt-1 text-[10px] leading-tight text-slate-600">
          {{ entry.description }}
        </p>
      </div>

      <div v-if="store.selectedNode.error" class="mt-2 rounded bg-red-950 p-2 text-xs text-red-300">
        {{ store.selectedNode.error }}
      </div>
    </template>

    <template v-else>
      <h2 class="text-sm font-bold text-slate-400">Node palette</h2>
      <p class="mb-3 text-xs text-slate-600">Select a node on the canvas to edit its parameters — hover any node for its architecture theory.</p>
      <div
        v-for="d in store.catalog"
        :key="d.type"
        class="mb-2 rounded border border-slate-800 p-2"
        :title="d.theory"
      >
        <p class="text-xs font-semibold text-slate-300">{{ d.label }}</p>
        <p class="font-mono text-[10px] text-slate-600">{{ d.type }} · {{ d.category }}</p>
        <p v-if="d.theory" class="mt-1 text-[10px] leading-tight text-slate-500">{{ d.theory.slice(0, 140) }}…</p>
      </div>
    </template>
  </aside>
</template>
