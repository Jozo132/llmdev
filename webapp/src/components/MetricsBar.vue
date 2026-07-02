<script setup lang="ts">
/**
 * MetricsBar — live telemetry strip: loss sparkline, tokens/sec, VRAM, RSS.
 */
import { computed } from "vue";
import { usePipelineStore } from "../stores/pipeline";

const store = usePipelineStore();

const lossPath = computed(() => {
  const pts = store.metrics["loss"] ?? [];
  if (pts.length < 2) return "";
  const w = 160;
  const h = 28;
  const vals = pts.map((p) => p.value);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;
  return vals
    .map((v, i) => `${i === 0 ? "M" : "L"} ${(i / (vals.length - 1)) * w} ${h - ((v - min) / span) * h}`)
    .join(" ");
});

const fmt = (name: string, digits = 1): string => {
  const m = store.latestMetric(name);
  return m ? m.value.toFixed(digits) : "—";
};
</script>

<template>
  <div class="flex items-center gap-6 border-b border-slate-800 bg-panel/60 px-4 py-1.5 text-xs">
    <div class="flex items-center gap-2">
      <span class="text-slate-500">loss</span>
      <span class="font-mono text-amber-300">{{ fmt("loss", 4) }}</span>
      <svg width="160" height="28" class="overflow-visible">
        <path :d="lossPath" fill="none" class="stroke-amber-400" stroke-width="1.5" />
      </svg>
    </div>
    <div><span class="text-slate-500">tok/s</span> <span class="font-mono text-sky-300">{{ fmt("tokens_per_sec", 0) }}</span></div>
    <div><span class="text-slate-500">VRAM</span> <span class="font-mono text-violet-300">{{ fmt("vram_mb", 0) }} MB</span></div>
    <div><span class="text-slate-500">RSS</span> <span class="font-mono text-slate-300">{{ fmt("rss_mb", 0) }} MB</span></div>
    <div><span class="text-slate-500">params</span> <span class="font-mono text-emerald-300">{{ fmt("param_count", 0) }}</span></div>
    <div><span class="text-slate-500">tokens written</span> <span class="font-mono text-slate-300">{{ fmt("tokens_written", 0) }}</span></div>
  </div>
</template>
