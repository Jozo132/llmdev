<script lang="ts">
export const NODE_W = 190;
export const NODE_H = 104;
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
const modelSizeLabel = computed(() => {
  const arch = store.state?.nodes.find((n) => n.type === "model.architecture");
  if (!arch) return null;
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
  return fmtParams(total);
});

const displayLabel = computed(() => props.node.label);

const latestNodeMetric = (name: string, nodeId = props.node.id): number | null => {
  const points = store.history[name] ?? [];
  for (let i = points.length - 1; i >= 0; i--) {
    if (points[i].nodeId === nodeId) return points[i].value;
  }
  return null;
};

const lossKpi = computed(() => {
  if (props.node.type !== "train.poc" && props.node.type !== "model.architecture") return null;
  const train = props.node.type === "train.poc"
    ? props.node
    : store.state?.nodes.find((n) => n.type === "train.poc");
  if (!train) return null;
  const best = latestNodeMetric("best_loss", train.id);
  const live = latestNodeMetric("loss", train.id);
  const value = best ?? live;
  return value == null ? "loss --" : `${best != null ? "best" : "loss"} ${value.toFixed(3)}`;
});

const techStack = computed(() => {
  if (props.node.type !== "model.architecture" && props.node.type !== "train.poc") return [] as string[];
  const arch = props.node.type === "model.architecture"
    ? props.node
    : store.state?.nodes.find((n) => n.type === "model.architecture");
  if (!arch) return [] as string[];
  const p = arch.params as Record<string, unknown>;
  const loraOn = /lora/i.test(String(p.fineTuneMode ?? ""));
  const esOn = Boolean(p.stochasticExplorationPool);
  const mlp = String(p.mlp ?? "standard");
  const backend = store.runtime?.backend ?? "backend ?";
  const stack = [
    backend,
    String(p.mixer ?? "causal-mean"),
    mlp === "swiglu" ? "SwiGLU" : "MLP",
    String(p.loss ?? "cross-entropy"),
    loraOn ? "LoRA" : "Full Adam",
  ];
  if (esOn) stack.push(loraOn ? "ES population" : "ES temporal");
  return stack;
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

const modeBadge = computed(() => {
  if (props.node.type !== "model.architecture") return null;
  const p = props.node.params as Record<string, unknown>;
  const loraOn = /lora/i.test(String(p.fineTuneMode ?? ""));
  const esOn = Boolean(p.stochasticExplorationPool);
  if (!esOn) return loraOn ? "LoRA · ES off" : "Full · ES off";
  if (!loraOn) return "Full · ES P=1";
  const populationSize = Math.max(1, Math.round(Number(p.populationSize ?? 4)));
  const survivalRate = Math.min(100, Math.max(1, Number(p.survivalRate ?? 25)));
  const survivalCount = Math.max(1, Math.min(populationSize, Math.ceil(populationSize * survivalRate / 100)));
  return `LoRA · ES P=${populationSize}/N=${survivalCount}`;
});

const primaryKpis = computed(() => {
  if (props.node.type !== "model.architecture" && props.node.type !== "train.poc") return [] as string[];
  return [
    modelSizeLabel.value ? `size ${modelSizeLabel.value}` : null,
    lossKpi.value,
  ].filter(Boolean) as string[];
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

    <text x="16" y="22" class="fill-slate-100 text-[13px] font-semibold">{{ displayLabel }}</text>
    <text x="16" y="38" class="fill-slate-500 text-[10px] font-mono">{{ node.id }} · {{ node.type }}</text>
    <text v-if="primaryKpis.length" x="16" y="54" class="fill-amber-300 text-[10px] font-mono">
      {{ primaryKpis.join(' · ') }}
    </text>
    <text v-if="modeBadge" x="16" y="70" class="fill-cyan-300 text-[10px] font-mono">{{ modeBadge }}</text>
    <text v-else-if="techStack.length" x="16" y="70" class="fill-cyan-300 text-[10px] font-mono">
      {{ techStack.slice(0, 3).join(' · ') }}
    </text>
    <text v-if="techStack.length" x="16" y="86" class="fill-slate-400 text-[9px] font-mono">
      {{ techStack.slice(modeBadge ? 0 : 3, modeBadge ? 3 : 6).join(' · ') }}
    </text>
    <text x="16" y="100" class="text-[10px]" :class="STATUS_BADGE[node.status].cls">
      {{ statusText }}
      <tspan v-if="node.error" class="fill-red-400"> — {{ node.error.slice(0, 24) }}</tspan>
    </text>

    <!-- live progress bar (node_progress 0..1 streamed over WebSockets) -->
    <g v-if="node.status === 'running' && progress !== null">
      <rect x="16" y="96" :width="NODE_W - 32" height="4" rx="2" class="fill-slate-700" />
      <rect x="16" y="96" :width="Math.max(2, (NODE_W - 32) * Math.min(1, progress))" height="4" rx="2" :fill="accent" />
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
