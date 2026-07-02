#!/usr/bin/env node
/**
 * serve.mjs — unified entrypoint for `npm run serve`.
 * Provisions (deps + optional native CUDA addon build), then boots:
 *   [engine] WebSocket backend with the node-tree executor  (ws://:8081)
 *   [webapp] Vue 3 frontend — SVG canvas + Chat Sandbox     (http://:5173)
 * Both are supervised; Ctrl-C tears everything down.
 */
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PIPELINE = process.argv[2] ?? "pipelines/poc-js-1m.json";

const C = { engine: "\x1b[36m", webapp: "\x1b[35m", serve: "\x1b[33m", reset: "\x1b[0m" };
const log = (tag, line) => console.log(`${C[tag]}[${tag}]${C.reset} ${line}`);

function sh(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { cwd: root, stdio: "inherit", ...opts });
  return r.status === 0;
}

// ── Provision ────────────────────────────────────────────────────────────────
if (!existsSync(path.join(root, "node_modules", "ws"))) {
  log("serve", "installing backend deps…");
  sh("npm", ["install", "--no-fund", "--no-audit"]);
}
if (!existsSync(path.join(root, "webapp", "node_modules", "vue"))) {
  log("serve", "installing webapp deps…");
  sh("npm", ["--prefix", "webapp", "install", "--no-fund", "--no-audit"]);
}

// Native CUDA addon: build only when nvcc is present (inside the dev
// container). Failure is NON-fatal — the engine falls back to the CPU backend.
const hasNvcc = spawnSync("nvcc", ["--version"], { stdio: "ignore" }).status === 0;
const addonBuilt = existsSync(path.join(root, "src/native/build/Release/llmdev_native.node"));
if (hasNvcc && !addonBuilt) {
  log("serve", "nvcc detected — compiling CUDA addon (sm_120)…");
  const ok =
    sh("npm", ["--prefix", "src/native", "install", "--no-fund", "--no-audit"]) &&
    sh("npm", ["--prefix", "src/native", "run", "build"]);
  log("serve", ok ? "CUDA addon ready." : "CUDA addon build FAILED — continuing on CPU backend.");
} else if (!hasNvcc) {
  log("serve", "nvcc not found — running on CPU backend (build inside the dev container for GPU).");
}

// ── Boot ─────────────────────────────────────────────────────────────────────
const procs = [];
function boot(tag, cmd, args) {
  const p = spawn(cmd, args, { cwd: root, env: process.env });
  p.stdout.on("data", (d) => String(d).split("\n").filter(Boolean).forEach((l) => log(tag, l)));
  p.stderr.on("data", (d) => String(d).split("\n").filter(Boolean).forEach((l) => log(tag, l)));
  p.on("exit", (code) => {
    log("serve", `${tag} exited (${code}) — shutting down.`);
    shutdown(code ?? 1);
  });
  procs.push(p);
  return p;
}

let closing = false;
function shutdown(code) {
  if (closing) return;
  closing = true;
  for (const p of procs) p.kill("SIGTERM");
  setTimeout(() => process.exit(code), 500);
}
process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

log("serve", `pipeline: ${PIPELINE}`);
boot("engine", "npx", ["tsx", "src/server/index.ts", PIPELINE]);
boot("webapp", "npm", ["--prefix", "webapp", "run", "dev"]);
log("serve", "canvas + chat → http://localhost:5173   bridge → ws://localhost:8081");
