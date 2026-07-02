// Loader for the compiled CUDA addon. Resolves to `null` when the addon has
// not been built (e.g. outside the dev container) so callers can fall back to
// the pure-JS CPU backend.
"use strict";
let native = null;
try {
  native = require("./build/Release/llmdev_native.node");
} catch {
  try {
    native = require("./build/Debug/llmdev_native.node");
  } catch {
    native = null;
  }
}
module.exports = native;
