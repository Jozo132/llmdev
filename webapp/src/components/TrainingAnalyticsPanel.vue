<script setup lang="ts">
/**
 * TrainingAnalyticsPanel — widescreen historic analytics dashboard.
 *
 * Renders the FULL (non-truncated) metric history as SVG line charts:
 *   · Loss curve            (train loss per step)
 *   · Token throughput      (tokens/sec velocity)
 *   · Memory tracking       (VRAM + RSS, dual series)
 *
 * Runtime controls sit beside the charts: Pause/Resume, Cancel, and the
 * prominent "Commit & Proceed" seam — freeze weights + Adam moments mid-epoch
 * and let the engine fire the downstream eval/export nodes immediately.
 */
import { computed, ref, watch } from "vue";
import { usePipelineStore } from "../stores/pipeline";
import type { MetricEvent } from "../types";

const store = usePipelineStore();

// ── Real-time lr hot-tuning ───────────────────────────────────────────
// Logarithmic slider (1e-7 → 1e-2): position p ∈ [0,1] maps to 10^(-7+5p).
// Dragging while a trainer is hot sends update_learning_rate; the loop
// consumes the new scalar on its next Adam step without pausing.
const LR_MIN_EXP = -7;
const LR_MAX_EXP = -2;
const lrSlider = ref(lrToPos(0.003));
const userTouchedLr = ref(false);

function lrToPos(lr: number): number {
  const e = Math.log10(Math.min(Math.max(lr, 10 ** LR_MIN_EXP), 10 ** LR_MAX_EXP));
  return (e - LR_MIN_EXP) / (LR_MAX_EXP - LR_MIN_EXP);
}
function posToLr(p: number): number {
  return 10 ** (LR_MIN_EXP + p * (LR_MAX_EXP - LR_MIN_EXP));
}
const activeLr = computed(() => posToLr(lrSlider.value));

function onLrInput() {
  userTouchedLr.value = true;
  store.updateLearningRate(activeLr.value);
}

// Track the trainer's reported lr until the user takes over the slider.
watch(
  () => store.history.learning_rate?.length,
  () => {
    if (userTouchedLr.value) return;
    const arr = store.history.learning_rate;
    if (arr?.length) lrSlider.value = lrToPos(arr[arr.length - 1].value);
  }
);
// New run ⇒ hand slider ownership back to the trainer's configured lr.
watch(() => store.state?.running, (running) => {
  if (running) userTouchedLr.value = false;
});

const fmtLr = (lr: number): string => lr.toExponential(2).replace("e-", "e−");

const W = 960;
const H = 230;
const PAD = { l: 56, r: 16, t: 14, b: 24 };

interface Series {
  label: string;
  color: string;
  points: MetricEvent[];
}

interface Chart {
  title: string;
  unit: string;
  series: Series[];
}

const charts = computed<Chart[]>(() => [
  {
    title: "Loss Curve",
    unit: "loss",
    series: [
      { label: "train loss", color: "#f59e0b", points: store.history.loss ?? [] },
      { label: "eval loss", color: "#10b981", points: store.history.eval_loss ?? [] },
    ],
  },
  {
    title: "Token Throughput",
    unit: "tok/s",
    series: [
      { label: "tokens/sec", color: "#0ea5e9", points: store.history.tokens_per_sec ?? [] },
    ],
  },
  {
    title: "Memory Tracking",
    unit: "MB",
    series: [
      { label: "VRAM", color: "#8b5cf6", points: store.history.vram_mb ?? [] },
      { label: "RSS", color: "#ec4899", points: store.history.rss_mb ?? [] },
    ],
  },
  {
    title: "Eval Perplexity",
    unit: "ppl",
    series: [
      { label: "held-out ppl", color: "#22d3ee", points: store.history.eval_perplexity ?? [] },
    ],
  },
  {
    title: "MCP Tool-Calling Accuracy",
    unit: "%",
    series: [
      { label: "functional", color: "#10b981", points: store.history.tool_accuracy ?? [] },
      { label: "syntax", color: "#f59e0b", points: store.history.tool_syntax_accuracy ?? [] },
      { label: "schema", color: "#8b5cf6", points: store.history.tool_schema_accuracy ?? [] },
    ],
  },
]);

/** Shared y-domain across a chart's series; x = sample index (full history). */
function chartScale(c: Chart) {
  const all = c.series.flatMap((s) => s.points.map((p) => p.value)).filter((v) => Number.isFinite(v));
  const maxLen = Math.max(1, ...c.series.map((s) => s.points.length));
  let lo = all.length ? Math.min(...all) : 0;
  let hi = all.length ? Math.max(...all) : 1;
  if (hi - lo < 1e-9) { hi = lo + 1; lo = lo - (lo === 0 ? 0 : 1e-9); }
  const span = hi - lo;
  lo -= span * 0.05;
  hi += span * 0.05;
  return {
    x: (i: number) => PAD.l + (maxLen <= 1 ? 0 : (i / (maxLen - 1)) * (W - PAD.l - PAD.r)),
    y: (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b),
    lo, hi, maxLen,
  };
}

function linePath(c: Chart, s: Series): string {
  if (s.points.length < 2) return "";
  const { x, y } = chartScale(c);
  return s.points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(p.value).toFixed(1)}`)
    .join(" ");
}

function yTicks(c: Chart): Array<{ y: number; label: string }> {
  const { y, lo, hi } = chartScale(c);
  const n = 4;
  return Array.from({ length: n + 1 }, (_, i) => {
    const v = lo + ((hi - lo) * i) / n;
    return { y: y(v), label: fmtVal(v) };
  });
}

const fmtVal = (v: number): string =>
  Math.abs(v) >= 10_000 ? `${(v / 1000).toFixed(1)}k`
  : Math.abs(v) >= 100 ? v.toFixed(0)
  : v.toFixed(2);

const latest = (name: string): number | null => {
  const arr = store.history[name];
  return arr?.length ? arr[arr.length - 1].value : null;
};

const stats = computed(() => {
  const loss = store.history.loss ?? [];
  const tps = store.history.tokens_per_sec ?? [];
  return {
    samples: loss.length,
    lastLoss: latest("loss"),
    minLoss: loss.length ? Math.min(...loss.map((p) => p.value)) : null,
    avgTps: tps.length ? tps.reduce((s, p) => s + p.value, 0) / tps.length : null,
    peakVram: (store.history.vram_mb ?? []).reduce((m, p) => Math.max(m, p.value), 0),
  };
});
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col overflow-y-auto bg-canvas p-4">
    <!-- ── Header: run stats + runtime controls ─────────────────────────── -->
    <div class="mb-4 flex flex-wrap items-center gap-3">
      <h2 class="text-sm font-bold uppercase tracking-widest text-slate-400">
        Training analytics
      </h2>
      <span class="font-mono text-[11px] text-slate-500">
        {{ stats.samples }} samples (full history, non-truncated)
      </span>

      <div class="ml-auto flex items-center gap-2">
        <!-- ── Live learning-rate hot-tune (log scale 1e-5 → 1e-2) ── -->
        <div
          class="flex items-center gap-2 rounded border border-slate-800 bg-panel px-2 py-1"
          title="Hot-tune Adam's learning rate while training runs — the loop applies it on the very next step without pausing or resetting metrics"
        >
          <span class="text-[10px] uppercase tracking-widest text-slate-500">lr</span>
          <input
            v-model.number="lrSlider"
            type="range"
            min="0"
            max="1"
            step="0.001"
            class="h-1 w-36 cursor-pointer accent-amber-400"
            :disabled="!store.state?.running"
            @input="onLrInput"
          />
          <span class="w-16 text-right font-mono text-[11px] text-amber-300">{{ fmtLr(activeLr) }}</span>
        </div>
        <button
          v-if="store.state?.running && !store.state?.paused"
          class="rounded bg-amber-700 px-3 py-1.5 text-xs font-semibold hover:bg-amber-600"
          @click="store.pauseTraining()"
        >⏸ Pause</button>
        <button
          v-if="store.state?.paused"
          class="rounded bg-sky-700 px-3 py-1.5 text-xs font-semibold hover:bg-sky-600"
          @click="store.resumeTraining()"
        >⏵ Resume</button>
        <button
          class="rounded bg-emerald-600 px-4 py-1.5 text-xs font-bold text-emerald-50 shadow-lg shadow-emerald-900/40 hover:bg-emerald-500 disabled:opacity-40"
          :disabled="!store.state?.running"
          title="Interrupt remaining iterations NOW: freeze weights + Adam moments as-is, mark the node done, and fire downstream eval/export automatically"
          @click="store.commitTraining()"
        >✔ Commit &amp; Proceed</button>
        <button
          class="rounded bg-red-800 px-3 py-1.5 text-xs font-semibold hover:bg-red-700 disabled:opacity-40"
          :disabled="!store.state?.running"
          @click="store.cancelTraining()"
        >■ Cancel</button>
      </div>
    </div>

    <!-- ── Summary strip ─────────────────────────────────────────────────── -->
    <div class="mb-4 grid grid-cols-2 gap-2 md:grid-cols-4">
      <div class="rounded border border-slate-800 bg-panel p-2">
        <p class="text-[10px] uppercase tracking-widest text-slate-500">Last loss</p>
        <p class="font-mono text-lg text-amber-300">{{ stats.lastLoss?.toFixed(4) ?? "—" }}</p>
      </div>
      <div class="rounded border border-slate-800 bg-panel p-2">
        <p class="text-[10px] uppercase tracking-widest text-slate-500">Best loss</p>
        <p class="font-mono text-lg text-emerald-300">{{ stats.minLoss?.toFixed(4) ?? "—" }}</p>
      </div>
      <div class="rounded border border-slate-800 bg-panel p-2">
        <p class="text-[10px] uppercase tracking-widest text-slate-500">Avg throughput</p>
        <p class="font-mono text-lg text-sky-300">{{ stats.avgTps ? `${stats.avgTps.toFixed(0)} tok/s` : "—" }}</p>
      </div>
      <div class="rounded border border-slate-800 bg-panel p-2">
        <p class="text-[10px] uppercase tracking-widest text-slate-500">Peak VRAM</p>
        <p class="font-mono text-lg text-violet-300">{{ stats.peakVram ? `${stats.peakVram.toFixed(0)} MB` : "—" }}</p>
      </div>
    </div>

    <!-- ── Full-history charts ───────────────────────────────────────────── -->
    <div v-for="c in charts" :key="c.title" class="mb-4 rounded-lg border border-slate-800 bg-panel p-3">
      <div class="mb-1 flex items-center gap-3">
        <h3 class="text-xs font-bold uppercase tracking-widest text-slate-400">{{ c.title }}</h3>
        <span class="font-mono text-[10px] text-slate-600">{{ c.unit }}</span>
        <span
          v-for="s in c.series"
          :key="s.label"
          class="flex items-center gap-1 font-mono text-[10px] text-slate-400"
        >
          <span class="inline-block h-2 w-2 rounded-full" :style="{ background: s.color }" />
          {{ s.label }}
          <span v-if="s.points.length" :style="{ color: s.color }">
            {{ fmtVal(s.points[s.points.length - 1].value) }}
          </span>
        </span>
      </div>

      <svg
        :viewBox="`0 0 ${W} ${H}`"
        preserveAspectRatio="none"
        class="h-56 w-full"
      >
        <!-- grid + y axis labels -->
        <g v-for="(t, i) in yTicks(c)" :key="i">
          <line :x1="PAD.l" :x2="W - PAD.r" :y1="t.y" :y2="t.y" class="stroke-slate-800" stroke-width="1" />
          <text :x="PAD.l - 6" :y="t.y + 3" text-anchor="end" class="fill-slate-600 text-[9px] font-mono">{{ t.label }}</text>
        </g>
        <!-- series -->
        <path
          v-for="s in c.series"
          :key="s.label"
          :d="linePath(c, s)"
          fill="none"
          :stroke="s.color"
          stroke-width="1.6"
          vector-effect="non-scaling-stroke"
        />
        <text
          v-if="!c.series.some((s) => s.points.length >= 2)"
          :x="W / 2" :y="H / 2" text-anchor="middle"
          class="fill-slate-600 text-xs"
        >no samples yet — run a pipeline or train a variant</text>
      </svg>
    </div>

    <!-- ── Remote worker fleet (horizontal scaling) ──────────────────────── -->
    <div class="rounded-lg border border-slate-800 bg-panel p-3">
      <h3 class="mb-1 text-xs font-bold uppercase tracking-widest text-slate-400">
        Compute workers
      </h3>
      <p v-if="!store.workers.length" class="text-[11px] text-slate-600">
        No remote workers connected — attach one via ws://host:8881/worker
      </p>
      <div v-for="w in store.workers" :key="w.id" class="font-mono text-[11px] text-slate-300">
        ● {{ w.id }} <span class="text-slate-500">since {{ w.connectedAt }}</span>
        <span class="ml-2 text-slate-500">{{ JSON.stringify(w.capabilities) }}</span>
      </div>
    </div>
  </div>
</template>
