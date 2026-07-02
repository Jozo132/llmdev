<script lang="ts">
export const NODE_W = 190;
export const NODE_H = 74;
</script>

<script setup lang="ts">
import { computed } from "vue";
import { usePipelineStore } from "../stores/pipeline";
import { countParams, fmtParams } from "../lib/paramMath";
import type { NodeStateSnapshot, PortSpec } from "../types";

const props = defineProps<{
  node: NodeStateSnapshot;
  selected: boolean;
  pendingDataType: string | null; // dataType of an in-flight connection drag
}>();
const emit = defineEmits<{
  nodedown: [e: PointerEvent];
  portdown: [port: PortSpec, isOutput: boolean, e: PointerEvent];
  portup: [port: PortSpec, isOutput: boolean];
}>();
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

/** Labels derive from live parameter configuration — never hardcoded sizes. */
const displayLabel = computed(() => {
  if (descriptor.value?.category !== "train") return props.node.label;
  const arch = store.state?.nodes.find((n) => n.type === "model.architecture");
  if (!arch) return props.node.label;
  const tok = store.state?.nodes.find((n) => n.type === "tokenizer.byteBpe");
  const p = arch.params as Record<string, unknown>;
  const total = countParams({
    vocabSize: Number(tok?.params.vocabSize ?? 8192),
    dModel: Number(p.dModel ?? 120),
    contextLength: Number(p.contextLength ?? 64),
    hiddenDim: Number(p.hiddenDim ?? 256),
    nLayers: Number(p.nLayers ?? 1),
    nHeads: Number(p.nHeads ?? 1),
    kvHeads: Number(p.kvHeads ?? 1),
    mlp: (p.mlp as "standard" | "swiglu") ?? "standard",
    tieEmbeddings: true,
  }).total;
  return `${props.node.label} (${fmtParams(total)})`;
});

/** Live 0..1 progress streamed as node_progress metrics. */
const progress = computed(() => store.nodeProgress[props.node.id] ?? null);
/** Last high-resolution execution time (performance.now() on the engine). */
const execTime = computed(() => {
  const t = store.nodeTimes[props.node.id];
  if (t == null) return null;
  return t < 1000 ? `${t.toFixed(0)}ms` : `${(t / 1000).toFixed(1)}s`;
});

const fmtClock = (ms: number): string => {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const mm = `${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;
  return h > 0 ? `${h}:${mm}` : mm;
};

/** "[⏳ 01:23 / 04:15 Remaining]" — windowed velocity ETA from the engine. */
const etaLabel = computed(() => {
  if (props.node.status !== "running") return null;
  const elapsed = store.nodeElapsed[props.node.id];
  if (elapsed == null) return null;
  const eta = store.nodeEta[props.node.id];
  return eta != null
    ? `[⏳ ${fmtClock(elapsed)} / ${fmtClock(eta)} Remaining]`
    : `[⏳ ${fmtClock(elapsed)}]`;
});

const STATUS_BADGE: Record<string, { text: string; cls: string }> = {
  idle: { text: "idle", cls: "fill-slate-500" },
  running: { text: "running…", cls: "fill-amber-400" },
  done: { text: "✓ done", cls: "fill-emerald-400" },
  error: { text: "✕ error", cls: "fill-red-400" },
  skipped: { text: "skipped", cls: "fill-slate-600" },
};

const statusText = computed(() => {
  if (props.node.status === "running" && progress.value !== null) {
    return `running… ${(progress.value * 100).toFixed(0)}%`;
  }
  if (props.node.status === "done" && execTime.value) {
    return `✓ done · ${execTime.value}`;
  }
  return STATUS_BADGE[props.node.status].text;
});

/** Input ports glow when they can accept the in-flight connection. */
const acceptsPending = (p: PortSpec) =>
  props.pendingDataType !== null && p.dataType === props.pendingDataType;
</script>

<template>
  <g :transform="`translate(${pos.x}, ${pos.y})`" class="cursor-move" @pointerdown="emit('nodedown', $event)">
    <!-- verbose theory tooltip (native SVG title on the whole node) -->
    <title v-if="descriptor?.theory">{{ descriptor.label }} — {{ descriptor.theory }}</title>

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

    <text x="16" y="24" class="fill-slate-100 text-[13px] font-semibold">{{ displayLabel }}</text>
    <text x="16" y="42" class="fill-slate-500 text-[10px] font-mono">{{ node.id }} · {{ node.type }}</text>
    <text x="16" y="58" class="text-[11px]" :class="STATUS_BADGE[node.status].cls">
      {{ statusText }}
      <tspan v-if="node.error" class="fill-red-400"> — {{ node.error.slice(0, 24) }}</tspan>
    </text>

    <!-- live progress bar (node_progress 0..1 streamed over WebSockets) -->
    <g v-if="node.status === 'running' && progress !== null">
      <rect x="16" y="64" :width="NODE_W - 32" height="4" rx="2" class="fill-slate-700" />
      <rect x="16" y="64" :width="Math.max(2, (NODE_W - 32) * Math.min(1, progress))" height="4" rx="2" :fill="accent" />
    </g>

    <!-- elapsed / windowed-velocity ETA readout next to the status indicator -->
    <text
      v-if="etaLabel"
      :x="NODE_W / 2" :y="NODE_H + 14" text-anchor="middle"
      class="fill-amber-300 text-[10px] font-mono"
    >{{ etaLabel }}</text>

    <!-- input ports — pointerup completes a pending connection -->
    <circle
      v-for="(p, i) in descriptor?.inputs ?? []"
      :key="'in' + p.name"
      :cx="0" :cy="NODE_H / 2 + (i - ((descriptor?.inputs.length ?? 1) - 1) / 2) * 14"
      :r="acceptsPending(p) ? 7 : 5"
      class="stroke-canvas"
      :class="acceptsPending(p) ? 'fill-sky-400' : 'fill-slate-400'"
      stroke-width="2"
      @pointerdown.stop="emit('portdown', p, false, $event)"
      @pointerup.stop="emit('portup', p, false)"
    >
      <title>{{ p.name }}: {{ p.dataType }}{{ p.required ? " (required)" : "" }}</title>
    </circle>
    <!-- output ports — pointerdown starts a connection -->
    <circle
      v-for="(p, i) in descriptor?.outputs ?? []"
      :key="'out' + p.name"
      :cx="NODE_W" :cy="NODE_H / 2 + (i - ((descriptor?.outputs.length ?? 1) - 1) / 2) * 14"
      r="5" :fill="accent" class="cursor-crosshair stroke-canvas" stroke-width="2"
      @pointerdown.stop="emit('portdown', p, true, $event)"
      @pointerup.stop="emit('portup', p, true)"
    >
      <title>{{ p.name }}: {{ p.dataType }} — drag to a matching input</title>
    </circle>
  </g>
</template>
