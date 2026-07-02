#!/usr/bin/env tsx
/**
 * Headless CLI — executes a node-graph pipeline without the web UI.
 *
 *   npm run cli -- run pipelines/poc-js-1m.json
 *   npm run cli -- validate pipelines/poc-js-1m.json
 *   npm run cli -- catalog
 */
import { readFileSync } from "node:fs";
import "./nodes/index.js";
import { Engine } from "./core/Engine.js";
import { listDescriptors } from "./core/Registry.js";
import type { PipelineSpec } from "./core/types.js";

const [cmd, file] = process.argv.slice(2);

function loadSpec(path: string): PipelineSpec {
  return JSON.parse(readFileSync(path, "utf8")) as PipelineSpec;
}

async function main(): Promise<void> {
  switch (cmd) {
    case "catalog": {
      for (const d of listDescriptors()) {
        const ins = d.inputs.map((p) => `${p.name}:${p.dataType}`).join(", ") || "—";
        const outs = d.outputs.map((p) => `${p.name}:${p.dataType}`).join(", ") || "—";
        console.log(`${d.type.padEnd(24)} [${d.category}]  in(${ins})  out(${outs})`);
      }
      return;
    }
    case "validate": {
      if (!file) throw new Error("Usage: cli validate <pipeline.json>");
      const engine = new Engine();
      engine.load(loadSpec(file));
      console.log(`OK — "${loadSpec(file).name}" is a valid acyclic pipeline.`);
      return;
    }
    case "run": {
      if (!file) throw new Error("Usage: cli run <pipeline.json>");
      const engine = new Engine();
      engine.on("log", ({ nodeId, message }) => console.log(`[${nodeId}] ${message}`));
      engine.on("state", (s: ReturnType<Engine["snapshot"]>) => {
        const line = s.nodes.map((n) => `${n.id}:${n.status}`).join("  ");
        process.stdout.write(`\x1b[2m${line}\x1b[0m\n`);
      });
      process.on("SIGINT", () => {
        console.log("\nStopping…");
        engine.stop();
      });
      engine.load(loadSpec(file));
      await engine.run();
      const failed = engine.snapshot().nodes.filter((n) => n.status === "error");
      if (failed.length) {
        console.error(`FAILED: ${failed.map((n) => `${n.id} (${n.error})`).join("; ")}`);
        process.exitCode = 1;
      } else {
        console.log("Pipeline complete.");
      }
      return;
    }
    default:
      console.log("Usage: cli <run|validate|catalog> [pipeline.json]");
      process.exitCode = 2;
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exitCode = 1;
});
