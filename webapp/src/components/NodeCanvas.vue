<script setup lang="ts">
/**
 * NodeCanvas — interactive 2D SVG playground.
 * Drag nodes, watch live status glow while the backend executes, click a node
 * to edit its parameters. Positions sync over WebSockets so every client and
 * the headless engine share one graph.
 */
import { computed, ref } from "vue";
import { usePipelineStore } from "../stores/pipeline";
import GraphNode, { NODE_W, NODE_H } from "./GraphNode.vue";
import type { NodeStateSnapshot } from "../types";

const store = usePipelineStore();

const svgEl = ref<SVGSVGElement | null>(null);
const viewBox = ref({ x: 0, y: 0, w: 1600, h: 900 });

// ── Edge geometry: cubic bezier from output port to input port ──────────────
const edgePaths = computed(() => {
  const st = store.state;
  if (!st) return [];
  const pos = new Map(st.nodes.map((n) => [n.id, n.position ?? { x: 0, y: 0 }]));
  return st.edges.map((e, i) => {
    const a = pos.get(e.from.node)!;
    const b = pos.get(e.to.node)!;
    const x1 = a.x + NODE_W;
    const y1 = a.y + NODE_H / 2;
    const x2 = b.x;
    const y2 = b.y + NODE_H / 2;
    const dx = Math.max(40, Math.abs(x2 - x1) / 2);
    return {
      key: `${i}`,
      d: `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`,
      active: st.running && isEdgeActive(e.from.node, e.to.node, st.nodes),
    };
  });
});

function isEdgeActive(from: string, to: string, nodes: NodeStateSnapshot[]): boolean {
  const f = nodes.find((n) => n.id === from);
  const t = nodes.find((n) => n.id === to);
  return f?.status === "done" && t?.status === "running";
}

// ── Drag handling (node drag + canvas pan) ──────────────────────────────────
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
  const p = svgPoint(e);
  const pos = node.position ?? { x: 0, y: 0 };
  drag.value = { nodeId: node.id, offX: p.x - pos.x, offY: p.y - pos.y };
  (e.target as Element).setPointerCapture?.(e.pointerId);
}

function onCanvasDown(e: PointerEvent) {
  if (e.target === svgEl.value) {
    store.selectedNodeId = null;
    pan.value = {
      startX: e.clientX,
      startY: e.clientY,
      vx: viewBox.value.x,
      vy: viewBox.value.y,
    };
  }
}

function onMove(e: PointerEvent) {
  if (drag.value) {
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
</script>

<template>
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
    <!-- dot grid -->
    <defs>
      <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
        <circle cx="1" cy="1" r="1" class="fill-slate-800" />
      </pattern>
    </defs>
    <rect
      :x="viewBox.x" :y="viewBox.y" :width="viewBox.w" :height="viewBox.h"
      fill="url(#grid)" pointer-events="none"
    />

    <!-- edges -->
    <path
      v-for="e in edgePaths"
      :key="e.key"
      :d="e.d"
      fill="none"
      stroke-width="2.5"
      :class="e.active ? 'stroke-emerald-400' : 'stroke-slate-600'"
    >
      <animate v-if="e.active" attributeName="stroke-dashoffset" from="24" to="0" dur="0.6s" repeatCount="indefinite" />
    </path>

    <!-- nodes -->
    <GraphNode
      v-for="n in store.state?.nodes ?? []"
      :key="n.id"
      :node="n"
      :selected="n.id === store.selectedNodeId"
      @pointerdown="(e: PointerEvent) => onNodeDown(n, e)"
    />
  </svg>
</template>
