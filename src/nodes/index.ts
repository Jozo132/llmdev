/**
 * Node catalog — importing this module registers every built-in node type.
 * Add custom nodes (PyTorch bridges, C++ kernels, alternative trainers) by
 * calling registerNode() from your own module and importing it before use.
 */
import { registerNode } from "../core/Registry.js";
import { JsIngestionNode, jsIngestionDescriptor } from "./data/JsIngestionNode.js";
import { TokenizerNode, tokenizerDescriptor } from "./tokenizer/TokenizerNode.js";
import { ModelArchitectureNode, modelArchitectureDescriptor } from "./model/ModelArchitectureNode.js";
import { PoCTrainer, pocTrainerDescriptor } from "./train/PoCTrainer.js";
import { EvaluationNode, evaluationDescriptor } from "./eval/EvaluationNode.js";
import { ExportNode, exportDescriptor } from "./export/ExportNode.js";

registerNode(jsIngestionDescriptor, (p) => new JsIngestionNode(p));
registerNode(tokenizerDescriptor, (p) => new TokenizerNode(p));
registerNode(modelArchitectureDescriptor, (p) => new ModelArchitectureNode(p));
registerNode(pocTrainerDescriptor, (p) => new PoCTrainer(p));
registerNode(evaluationDescriptor, (p) => new EvaluationNode(p));
registerNode(exportDescriptor, (p) => new ExportNode(p));
