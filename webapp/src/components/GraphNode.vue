<script lang="ts">
export const NODE_W = 190;
export const NODE_H = 74;
</script>

<script setup lang="ts">
import { computed } from "vue";
import { usePipelineStore } from "../stores/pipeline";
import type { NodeStateSnapshot } from "../types";

const props = defineProps<{ node: NodeStateSnapshot; selected: boolean }>();
const store = usePipelineStore();

const CATEGORY_COLORS: Record<string, string> = {
  data: "#0ea5e9",
  tokenizer: "#8b5cf6",
  model: "#f59e0b",
  train: "#ef4444",
  eval: "#10b981",
  export: "#64748b",
  custom: "#ec4899",
};

const descriptor = computed(() => store.catalog.find((d) => d.type === props.node.type));
const accent = computed(() => CATEGORY_COLORS[descriptor.value?.category ?? "custom"]);
const pos = computed(() => props.node.position ?? { x: 0, y: 0 });

const STATUS_BADGE: Record<string, { text: string; cls: string }> = {
  idle: { text: "idle", cls: "fill-slate-500" },
  running: { text: "running…", cls: "fill-amber-400" },
  done: { text: "✓ done", cls: "fill-emerald-400" },
  error: { text: "✕ error", cls: "fill-red-400" },
  skipped: { text: "skipped", cls: "fill-slate-600" },
};
</script>

<template>
  <g :transform="`translate(${pos.x}, ${pos.y})`" class="cursor-move">
    <!-- running glow -->
    <rect
      v-if="node.status === 'running'"
      :width="NODE_W" :height="NODE_H" rx="10"
      fill="none" :stroke="accent" stroke-width="6" opacity="0.35"
    >
      <animate attributeName="opacity" values="0.15;0.5;0.15" dur="1.2s" repeatCount="indefinite" />
    </rect>

    <rect
      :width="NODE_W" :height="NODE_H" rx="10"
      class="fill-panel"
      :stroke="selected ? '#f8fafc' : accent"
      :stroke-width="selected ? 2.5 : 1.5"
    />
    <!-- category stripe -->
    <rect width="6" :height="NODE_H" rx="3" :fill="accent" />

    <text x="16" y="24" class="fill-slate-100 text-[13px] font-semibold">{{ node.label }}</text>
    <text x="16" y="42" class="fill-slate-500 text-[10px] font-mono">{{ node.id }} · {{ node.type }}</text>
    <text x="16" y="60" class="text-[11px]" :class="STATUS_BADGE[node.status].cls">
      {{ STATUS_BADGE[node.status].text }}
      <tspan v-if="node.error" class="fill-red-400"> — {{ node.error.slice(0, 24) }}</tspan>
    </text>

    <!-- ports -->
    <circle
      v-for="(p, i) in descriptor?.inputs ?? []"
      :key="'in' + p.name"
      :cx="0" :cy="NODE_H / 2 + (i - ((descriptor?.inputs.length ?? 1) - 1) / 2) * 14"
      r="5" class="fill-slate-400 stroke-canvas" stroke-width="2"
    >
      <title>{{ p.name }}: {{ p.dataType }}</title>
    </circle>
    <circle
      v-for="(p, i) in descriptor?.outputs ?? []"
      :key="'out' + p.name"
      :cx="NODE_W" :cy="NODE_H / 2 + (i - ((descriptor?.outputs.length ?? 1) - 1) / 2) * 14"
      r="5" :fill="accent" class="stroke-canvas" stroke-width="2"
    >
      <title>{{ p.name }}: {{ p.dataType }}</title>
    </circle>
  </g>
</template>
