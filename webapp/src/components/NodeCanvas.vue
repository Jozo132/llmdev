<script setup lang="ts">
/**
 * NodeCanvas — interactive 2D SVG playground.
 * Full graph editing: drag nodes from the palette, connect output→input ports
 * with strict dataType validation, click edges/nodes + Delete to remove them,
 * apply architectural templates, and watch the live parameter-count badge
 * recompute as hyperparameters change.
 */
import { computed, onMounted, onUnmounted, ref } from "vue";
import { usePipelineStore } from "../stores/pipeline";
import { countParams, fmtParams } from "../lib/paramMath";
import GraphNode, { NODE_W, NODE_H } from "./GraphNode.vue";
import type { EdgeSpec, NodeStateSnapshot, PortSpec } from "../types";

const store = usePipelineStore();

const svgEl = ref<SVGSVGElement | null>(null);
const viewBox = ref({ x: 0, y: 0, w: 1600, h: 900 });

const portY = (i: number, count: number) => NODE_H / 2 + (i - (count - 1) / 2) * 14;

// ── Live parameter calculator badge ──────────────────────────────────────────
const showBreakdown = ref(false);
const paramBadge = computed(() => {
  const arch = store.state?.nodes.find((n) => n.type === "model.architecture");
  if (!arch) return null;
  const tok = store.state?.nodes.find((n) => n.type === "tokenizer.byteBpe");
  const p = arch.params as Record<string, number | string>;
  return countParams({
    vocabSize: Number(tok?.params.vocabSize ?? 8192),
    dModel: Number(p.dModel ?? 120),
    contextLength: Number(p.contextLength ?? 64),
    hiddenDim: Number(p.hiddenDim ?? 256),
    nLayers: Number(p.nLayers ?? 1),
    nHeads: Number(p.nHeads ?? 1),
    kvHeads: Number(p.kvHeads ?? 1),
    mlp: (p.mlp as "standard" | "swiglu") ?? "standard",
    tieEmbeddings: true,
  });
});

// ── Edge geometry ────────────────────────────────────────────────────────────
const selectedEdge = ref<EdgeSpec | null>(null);

function portPos(nodeId: string, portName: string, isOutput: boolean) {
  const node = store.state?.nodes.find((n) => n.id === nodeId);
  const desc = store.catalog.find((d) => d.type === node?.type);
  const pos = node?.position ?? { x: 0, y: 0 };
  const ports = (isOutput ? desc?.outputs : desc?.inputs) ?? [];
  const idx = Math.max(0, ports.findIndex((p) => p.name === portName));
  return {
    x: pos.x + (isOutput ? NODE_W : 0),
    y: pos.y + portY(idx, Math.max(1, ports.length)),
  };
}

const bezier = (x1: number, y1: number, x2: number, y2: number) => {
  const dx = Math.max(40, Math.abs(x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
};

const edgePaths = computed(() => {
  const st = store.state;
  if (!st) return [];
  return st.edges.map((e, i) => {
    const a = portPos(e.from.node, e.from.port, true);
    const b = portPos(e.to.node, e.to.port, false);
    const f = st.nodes.find((n) => n.id === e.from.node);
    const t = st.nodes.find((n) => n.id === e.to.node);
    return {
      key: `${i}`,
      edge: e,
      d: bezier(a.x, a.y, b.x, b.y),
      active: st.running && f?.status === "done" && t?.status === "running",
      selected:
        selectedEdge.value !== null &&
        JSON.stringify(selectedEdge.value) === JSON.stringify(e),
    };
  });
});

// ── Pending connection (drag from an output port) ────────────────────────────
const pending = ref<null | {
  fromNode: string;
  fromPort: string;
  dataType: string;
  mouse: { x: number; y: number };
}>(null);

const pendingPath = computed(() => {
  if (!pending.value) return "";
  const a = portPos(pending.value.fromNode, pending.value.fromPort, true);
  return bezier(a.x, a.y, pending.value.mouse.x, pending.value.mouse.y);
});

function onPortDown(node: NodeStateSnapshot, port: PortSpec, isOutput: boolean, e: PointerEvent) {
  if (!isOutput) return; // connections start at outputs
  e.stopPropagation();
  pending.value = {
    fromNode: node.id,
    fromPort: port.name,
    dataType: port.dataType,
    mouse: svgPoint(e),
  };
}

function onPortUp(node: NodeStateSnapshot, port: PortSpec, isOutput: boolean) {
  if (!pending.value || isOutput) return;
  // Strict type validation before we even ask the engine.
  if (port.dataType !== pending.value.dataType) {
    store.lastError = `Type mismatch: ${pending.value.dataType} → ${port.dataType}`;
  } else {
    store.addEdge({
      from: { node: pending.value.fromNode, port: pending.value.fromPort },
      to: { node: node.id, port: port.name },
    });
  }
  pending.value = null;
}

// ── Palette drag-to-add ──────────────────────────────────────────────────────
const paletteDrag = ref<null | { type: string; mouse: { x: number; y: number } }>(null);

function onPaletteDown(type: string, e: PointerEvent) {
  paletteDrag.value = { type, mouse: svgPoint(e) };
  window.addEventListener("pointerup", onPaletteDrop, { once: true });
}

function onPaletteDrop(e: PointerEvent) {
  if (!paletteDrag.value) return;
  const rect = svgEl.value?.getBoundingClientRect();
  if (rect && e.clientX >= rect.left && e.clientX <= rect.right &&
      e.clientY >= rect.top && e.clientY <= rect.bottom) {
    const p = svgPoint(e);
    const type = paletteDrag.value.type;
    const id = `${type.split(".").pop()}-${Math.random().toString(36).slice(2, 6)}`;
    store.addNode({
      id, type, params: {},
      position: { x: Math.round(p.x - NODE_W / 2), y: Math.round(p.y - NODE_H / 2) },
    });
  }
  paletteDrag.value = null;
}

// ── Node drag + canvas pan/zoom ──────────────────────────────────────────────
const drag = ref<null | { nodeId: string; offX: number; offY: number }>(null);
const pan = ref<null | { startX: number; startY: number; vx: number; vy: number }>(null);

function svgPoint(e: PointerEvent): { x: number; y: number } {
  const rect = svgEl.value!.getBoundingClientRect();
  return {
    x: viewBox.value.x + ((e.clientX - rect.left) / rect.width) * viewBox.value.w,
    y: viewBox.value.y + ((e.clientY - rect.top) / rect.height) * viewBox.value.h,
  };
}

function onNodeDown(node: NodeStateSnapshot, e: PointerEvent) {
  store.selectedNodeId = node.id;
  selectedEdge.value = null;
  const p = svgPoint(e);
  const pos = node.position ?? { x: 0, y: 0 };
  drag.value = { nodeId: node.id, offX: p.x - pos.x, offY: p.y - pos.y };
}

function onCanvasDown(e: PointerEvent) {
  if (e.target === svgEl.value) {
    store.selectedNodeId = null;
    selectedEdge.value = null;
    pan.value = {
      startX: e.clientX, startY: e.clientY,
      vx: viewBox.value.x, vy: viewBox.value.y,
    };
  }
}

function onMove(e: PointerEvent) {
  if (pending.value) {
    pending.value.mouse = svgPoint(e);
  } else if (drag.value) {
    const p = svgPoint(e);
    store.moveNode(drag.value.nodeId, {
      x: Math.round(p.x - drag.value.offX),
      y: Math.round(p.y - drag.value.offY),
    });
  } else if (pan.value) {
    const rect = svgEl.value!.getBoundingClientRect();
    const sx = viewBox.value.w / rect.width;
    viewBox.value.x = pan.value.vx - (e.clientX - pan.value.startX) * sx;
    viewBox.value.y = pan.value.vy - (e.clientY - pan.value.startY) * sx;
  }
}

function onUp() {
  drag.value = null;
  pan.value = null;
  pending.value = null;
}

function onWheel(e: WheelEvent) {
  const factor = e.deltaY > 0 ? 1.1 : 0.9;
  const vb = viewBox.value;
  const rect = svgEl.value!.getBoundingClientRect();
  const mx = vb.x + ((e.clientX - rect.left) / rect.width) * vb.w;
  const my = vb.y + ((e.clientY - rect.top) / rect.height) * vb.h;
  vb.x = mx - (mx - vb.x) * factor;
  vb.y = my - (my - vb.y) * factor;
  vb.w *= factor;
  vb.h *= factor;
}

// ── Keyboard deletion ────────────────────────────────────────────────────────
function onKeydown(e: KeyboardEvent) {
  if (e.key !== "Delete" && e.key !== "Backspace") return;
  const tag = (e.target as HTMLElement).tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if (selectedEdge.value) {
    store.removeEdge(selectedEdge.value);
    selectedEdge.value = null;
  } else if (store.selectedNodeId) {
    store.removeNode(store.selectedNodeId);
  }
}
onMounted(() => window.addEventListener("keydown", onKeydown));
onUnmounted(() => window.removeEventListener("keydown", onKeydown));

const CATEGORY_COLORS: Record<string, string> = {
  data: "#0ea5e9", tokenizer: "#8b5cf6", model: "#f59e0b",
  train: "#ef4444", eval: "#10b981", export: "#64748b", custom: "#ec4899",
};
</script>

<template>
  <div class="relative">
    <!-- ── Palette + templates overlay ─────────────────────────────────── -->
    <div class="absolute left-3 top-3 z-10 w-56 rounded-lg border border-slate-800 bg-panel/95 p-2 shadow-xl">
      <p class="mb-1 text-[10px] font-bold uppercase tracking-widest text-slate-500">Templates</p>
      <div v-for="t in store.templates" :key="t.id" class="mb-1">
        <button
          class="w-full rounded border border-slate-800 px-2 py-1 text-left text-[11px] hover:border-amber-600 hover:bg-slate-800"
          :title="t.description"
          @click="store.applyTemplate(t.id)"
        >
          <span class="font-semibold text-slate-200">{{ t.name }}</span>
          <span class="ml-1 font-mono text-[9px] text-amber-400">{{ fmtParams(t.designParams) }}</span>
        </button>
      </div>

      <p class="mb-1 mt-3 text-[10px] font-bold uppercase tracking-widest text-slate-500">
        Node palette <span class="normal-case text-slate-600">(drag onto canvas)</span>
      </p>
      <div
        v-for="d in store.catalog"
        :key="d.type"
        class="mb-1 cursor-grab rounded border border-slate-800 px-2 py-1 text-[11px] select-none hover:bg-slate-800 active:cursor-grabbing"
        :style="{ borderLeftColor: CATEGORY_COLORS[d.category], borderLeftWidth: '3px' }"
        @pointerdown="(e: PointerEvent) => onPaletteDown(d.type, e)"
      >
        <span class="text-slate-300">{{ d.label }}</span>
        <span class="ml-1 font-mono text-[9px] text-slate-600">{{ d.category }}</span>
      </div>
      <p class="mt-2 text-[9px] leading-tight text-slate-600">
        Drag output ● → input ○ to connect (types must match).
        Select node/edge + Delete to remove.
      </p>
    </div>

    <!-- ── Live parameter calculator badge ─────────────────────────────── -->
    <div v-if="paramBadge" class="absolute right-3 top-3 z-10">
      <button
        class="rounded-lg border border-amber-800 bg-panel/95 px-3 py-1.5 font-mono text-sm text-amber-300 shadow-xl hover:border-amber-500"
        @click="showBreakdown = !showBreakdown"
      >
        Σ {{ fmtParams(paramBadge.total) }} params
      </button>
      <div
        v-if="showBreakdown"
        class="mt-1 w-96 rounded-lg border border-slate-700 bg-panel/95 p-3 shadow-xl"
      >
        <p class="mb-1 text-[10px] font-bold uppercase tracking-widest text-slate-500">
          Design formula (live)
        </p>
        <pre class="whitespace-pre-wrap font-mono text-[10px] leading-relaxed text-slate-300">{{ paramBadge.formula.join("\n") }}</pre>
      </div>
    </div>

    <!-- ── SVG graph ────────────────────────────────────────────────────── -->
    <svg
      ref="svgEl"
      class="h-full w-full cursor-grab touch-none select-none active:cursor-grabbing"
      :viewBox="`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`"
      @pointerdown="onCanvasDown"
      @pointermove="onMove"
      @pointerup="onUp"
      @pointerleave="onUp"
      @wheel.prevent="onWheel"
    >
      <defs>
        <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="1" class="fill-slate-800" />
        </pattern>
      </defs>
      <rect
        :x="viewBox.x" :y="viewBox.y" :width="viewBox.w" :height="viewBox.h"
        fill="url(#grid)" pointer-events="none"
      />

      <!-- edges (wide invisible hit area + visible path) -->
      <g v-for="e in edgePaths" :key="e.key">
        <path
          :d="e.d" fill="none" stroke="transparent" stroke-width="14"
          class="cursor-pointer"
          @pointerdown.stop="selectedEdge = e.edge; store.selectedNodeId = null"
        />
        <path
          :d="e.d" fill="none" stroke-width="2.5" pointer-events="none"
          :class="e.selected ? 'stroke-red-400' : e.active ? 'stroke-emerald-400' : 'stroke-slate-600'"
          :stroke-dasharray="e.selected ? '6 3' : undefined"
        />
      </g>

      <!-- pending connection preview -->
      <path
        v-if="pending"
        :d="pendingPath" fill="none" stroke-width="2"
        class="stroke-sky-400" stroke-dasharray="5 4" pointer-events="none"
      />

      <!-- nodes -->
      <GraphNode
        v-for="n in store.state?.nodes ?? []"
        :key="n.id"
        :node="n"
        :selected="n.id === store.selectedNodeId"
        :pending-data-type="pending?.dataType ?? null"
        @nodedown="(e: PointerEvent) => onNodeDown(n, e)"
        @portdown="(port: PortSpec, isOutput: boolean, e: PointerEvent) => onPortDown(n, port, isOutput, e)"
        @portup="(port: PortSpec, isOutput: boolean) => onPortUp(n, port, isOutput)"
      />
    </svg>
  </div>
</template>
