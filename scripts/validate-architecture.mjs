import { readFileSync } from "node:fs";
import vm from "node:vm";

const root = new URL("../", import.meta.url);
const context = vm.createContext({ window: {} });

for (const filename of ["architecture-specs.js", "architecture-diagrams.js"]) {
  const source = readFileSync(new URL(filename, root), "utf8");
  vm.runInContext(source, context, { filename });
}

const specs = context.window.ARCHITECTURE_SPECS;
const diagrams = context.window.ARCHITECTURE_DIAGRAMS;
const errors = [];
const validModes = new Set(["both", "inference", "training"]);
const validTypes = new Set(["vision", "language", "state", "action", "world", "compute"]);
const validKinds = new Set(["data", "residual", "cross", "condition", "skip", "loop"]);
const validViews = new Set(["all", "modules", "layers"]);

const assert = (condition, message) => {
  if (!condition) errors.push(message);
};

assert(specs && typeof specs === "object", "Architecture specifications did not load.");
assert(diagrams && typeof diagrams === "object", "Architecture diagrams did not load.");

for (const [modelId, model] of Object.entries(specs || {})) {
  const prefix = `[${modelId}]`;
  assert(Array.isArray(model.components) && model.components.length > 0, `${prefix} has no components.`);
  assert(Array.isArray(model.links), `${prefix} has no links array.`);
  assert(Array.isArray(model.sources) && model.sources.length > 0, `${prefix} has no primary sources.`);

  const componentIds = new Set();
  for (const component of model.components || []) {
    assert(component.id && !componentIds.has(component.id), `${prefix} duplicate or empty component id: ${component.id || "<empty>"}.`);
    componentIds.add(component.id);
    assert(validTypes.has(component.type), `${prefix}/${component.id} has invalid type ${component.type}.`);
    assert(Number.isInteger(component.chapter) && component.chapter >= 1 && component.chapter <= 5, `${prefix}/${component.id} has invalid chapter ${component.chapter}.`);
    assert(component.output && component.output !== "—", `${prefix}/${component.id} has no output tensor.`);
    assert(validModes.has(component.mode || "both"), `${prefix}/${component.id} has invalid mode ${component.mode}.`);

    for (const [layerIndex, layer] of (component.layers || []).entries()) {
      for (const field of ["label", "input", "output", "op"]) {
        assert(layer[field], `${prefix}/${component.id}/layer-${layerIndex} is missing ${field}.`);
      }
    }
  }

  for (const link of model.links || []) {
    const from = Array.isArray(link) ? link[0] : link.from;
    const to = Array.isArray(link) ? link[1] : link.to;
    assert(componentIds.has(from), `${prefix} link references missing source ${from}.`);
    assert(componentIds.has(to), `${prefix} link references missing target ${to}.`);
    if (!Array.isArray(link)) {
      assert(validModes.has(link.mode || "both"), `${prefix} ${from}→${to} has invalid mode ${link.mode}.`);
      assert(validKinds.has(link.kind || "data"), `${prefix} ${from}→${to} has invalid kind ${link.kind}.`);
      if ((model.schemaVersion || 1) >= 2) assert(link.tensor, `${prefix} schema v${model.schemaVersion} link ${from}→${to} must declare its tensor.`);
    }
  }

  for (const [mode, steps] of Object.entries(model.flowSteps || {})) {
    assert(validModes.has(mode), `${prefix} has invalid trace mode ${mode}.`);
    for (const [stepIndex, step] of steps.entries()) {
      assert(step.title && step.tensor && step.note, `${prefix}/${mode}/step-${stepIndex} is missing instructional copy.`);
      assert(Array.isArray(step.nodes) && step.nodes.length > 0, `${prefix}/${mode}/step-${stepIndex} has no target nodes.`);
      for (const nodeId of step.nodes || []) assert(componentIds.has(nodeId), `${prefix}/${mode}/step-${stepIndex} references missing component ${nodeId}.`);
    }
  }

  for (const contract of model.validation?.tensorContracts || []) {
    const source = model.components.find((component) => component.id === contract.from);
    const matchingLink = (model.links || []).some((link) => {
      const from = Array.isArray(link) ? link[0] : link.from;
      const to = Array.isArray(link) ? link[1] : link.to;
      const mode = Array.isArray(link) ? "both" : (link.mode || "both");
      return from === contract.from && to === contract.to && (!contract.mode || mode === contract.mode || mode === "both");
    });
    assert(matchingLink, `${prefix} tensor contract ${contract.from}→${contract.to} has no matching graph link.`);
    assert(source?.output === contract.tensor, `${prefix} tensor contract ${contract.from}→${contract.to} expects ${contract.tensor}, source declares ${source?.output || "<missing>"}.`);
  }

  const diagram = diagrams?.[modelId];
  assert(diagram, `${prefix} has no architecture diagram.`);
  if (!diagram) continue;
  const diagramNodeIds = new Set();
  for (const node of diagram.nodes || []) {
    assert(node.id && !diagramNodeIds.has(node.id), `${prefix} diagram duplicate or empty node id: ${node.id || "<empty>"}.`);
    diagramNodeIds.add(node.id);
    if (node.componentId) {
      const component = model.components.find((item) => item.id === node.componentId);
      assert(component, `${prefix} diagram node ${node.id} references missing component ${node.componentId}.`);
      if (Number.isInteger(node.layerIndex)) assert(component?.layers?.[node.layerIndex], `${prefix} diagram node ${node.id} references missing ${node.componentId} layer ${node.layerIndex}.`);
    }
    if (node.view) assert(validViews.has(node.view), `${prefix} diagram node ${node.id} has invalid view ${node.view}.`);
  }
  for (const group of diagram.groups || []) if (group.view) assert(validViews.has(group.view), `${prefix} diagram group ${group.id} has invalid view ${group.view}.`);
  for (const edge of diagram.edges || []) {
    assert(diagramNodeIds.has(edge.from), `${prefix} diagram edge references missing source ${edge.from}.`);
    assert(diagramNodeIds.has(edge.to), `${prefix} diagram edge references missing target ${edge.to}.`);
    assert(validKinds.has(edge.kind || "data"), `${prefix} diagram edge ${edge.from}→${edge.to} has invalid kind ${edge.kind}.`);
    assert(validModes.has(edge.mode || "both"), `${prefix} diagram edge ${edge.from}→${edge.to} has invalid mode ${edge.mode}.`);
    if (edge.view) assert(validViews.has(edge.view), `${prefix} diagram edge ${edge.from}→${edge.to} has invalid view ${edge.view}.`);
  }
}

if ((specs?.act?.schemaVersion || 0) < 2) errors.push("[act] must use architecture schema v2 or later.");

if (errors.length) {
  console.error(`Architecture validation failed with ${errors.length} error(s):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

const modelCount = Object.keys(specs).length;
const componentCount = Object.values(specs).reduce((sum, model) => sum + model.components.length, 0);
const edgeCount = Object.values(diagrams).reduce((sum, diagram) => sum + diagram.edges.length, 0);
console.log(`Architecture validation passed: ${modelCount} models, ${componentCount} components, ${edgeCount} schematic edges.`);
