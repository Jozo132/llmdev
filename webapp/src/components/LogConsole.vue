<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
import { usePipelineStore } from "../stores/pipeline";

const store = usePipelineStore();
const box = ref<HTMLElement | null>(null);

watch(
  () => store.logs.length,
  async () => {
    await nextTick();
    box.value?.scrollTo({ top: box.value.scrollHeight });
  }
);
</script>

<template>
  <div ref="box" class="overflow-y-auto bg-black/40 px-4 py-2 font-mono text-[11px] leading-relaxed">
    <div v-for="(l, i) in store.logs" :key="i">
      <span class="text-sky-500">[{{ l.nodeId }}]</span>
      <span class="text-slate-300"> {{ l.message }}</span>
    </div>
    <div v-if="!store.logs.length" class="text-slate-600">— engine log —</div>
  </div>
</template>
