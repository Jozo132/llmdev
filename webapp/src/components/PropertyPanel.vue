<script setup lang="ts">
/**
 * PropertyPanel — schema-driven editor for the selected node. Edits are
 * pushed over WebSockets (update_params) and applied to the live engine,
 * e.g. click the Architecture node and swap the attention mixer.
 */
import { usePipelineStore } from "../stores/pipeline";
import type { ParamSchemaEntry } from "../types";

const store = usePipelineStore();

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
      <h2 class="text-sm font-bold text-slate-200">{{ store.selectedNode.label }}</h2>
      <p class="mb-4 font-mono text-[11px] text-slate-500">
        {{ store.selectedNode.id }} · {{ store.selectedNode.type }}
      </p>

      <div v-for="entry in store.selectedDescriptor.paramSchema" :key="entry.key" class="mb-3">
        <label class="mb-1 block text-xs font-medium text-slate-400">{{ entry.label }}</label>

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
      <p class="mb-3 text-xs text-slate-600">Select a node on the canvas to edit its parameters.</p>
      <div
        v-for="d in store.catalog"
        :key="d.type"
        class="mb-2 rounded border border-slate-800 p-2"
      >
        <p class="text-xs font-semibold text-slate-300">{{ d.label }}</p>
        <p class="font-mono text-[10px] text-slate-600">{{ d.type }} · {{ d.category }}</p>
      </div>
    </template>
  </aside>
</template>
