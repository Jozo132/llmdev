<script setup lang="ts">
/**
 * PropertyPanel — schema-driven editor for the selected node with baked-in
 * educational theory: each node shows its architectural function and every
 * parameter carries a ⓘ popover explaining its mathematical effect and a
 * strict safe operating range.
 */
import { ref } from "vue";
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

      <div v-for="entry in store.selectedDescriptor.paramSchema" :key="entry.key" class="mb-3">
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
