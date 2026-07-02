/**
 * NodeRegistry — plug-and-play catalog of node classes.
 * Register a factory once; instantiate anywhere (CLI, server, tests) by type key.
 */
import type { NodeDescriptor, NodeFactory, NodeParams, PipelineNode } from "./types.js";

const factories = new Map<string, NodeFactory>();
const descriptors = new Map<string, NodeDescriptor>();

export function registerNode(descriptor: NodeDescriptor, factory: NodeFactory): void {
  if (factories.has(descriptor.type)) {
    throw new Error(`Node type already registered: ${descriptor.type}`);
  }
  factories.set(descriptor.type, factory);
  descriptors.set(descriptor.type, descriptor);
}

export function createNode(type: string, params?: NodeParams): PipelineNode {
  const factory = factories.get(type);
  if (!factory) {
    throw new Error(
      `Unknown node type "${type}". Registered: ${[...factories.keys()].join(", ")}`
    );
  }
  return factory(params);
}

export function listDescriptors(): NodeDescriptor[] {
  return [...descriptors.values()];
}
