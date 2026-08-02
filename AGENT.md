# VLA Atlas repository instructions

This file defines repository-wide guidance for coding agents working on VLA Atlas. Read it before changing architecture data, diagrams, rendering, documentation, tests, or deployment configuration.

## Project objective

VLA Atlas is a dependency-free educational web application for understanding and comparing robot policies, Vision-Language-Action models, World Action Models, and world foundation models.

A successful change should make architectural differences easier to understand without sacrificing factual accuracy. Prefer explicit tensors, submodules, residual paths, cross-attention, activation functions, normalization, training-only branches, and inference-time control flow over generic boxes such as “visual encoder” or “policy.”

Public surfaces:

- Repository: `https://github.com/caisarl76/VLA_vis`
- GitHub Pages: `https://caisarl76.github.io/VLA_vis/`
- Deployment branch: `main`

## Repository map

| Path | Responsibility |
|---|---|
| `index.html` | Static application structure and accessible controls |
| `styles.css` | Layout, visual grammar, responsive states, diagrams, inspector, comparison UI |
| `app.js` | Model registry, lesson copy, rendering, interaction, tracing, comparison |
| `architecture-specs.js` | Canonical components, layers, tensor shapes, typed links, traces, provenance |
| `architecture-diagrams.js` | Hierarchical SVG groups, nodes, edges, residual/cross/skip routing |
| `scripts/validate-architecture.mjs` | Schema, graph, trace, diagram, and tensor-contract validation |
| `docs/ADDING_A_VLA.md` | End-to-end authoring workflow |
| `docs/MODEL_SCHEMA.md` | Field-level schema reference |
| `.github/workflows/deploy-pages.yml` | Validation and GitHub Pages deployment |

The site has no package manager, bundler, framework, or runtime API. Browser files are loaded directly in this order:

1. `architecture-specs.js`
2. `architecture-diagrams.js`
3. `app.js`

Do not introduce a build system or third-party dependency unless the user explicitly requests that architectural change.

## Before editing

1. Run `git status --short` and preserve unrelated user changes.
2. Identify whether the request affects lesson copy, canonical architecture, SVG diagram, 3-D renderer, or several of them.
3. Read the relevant model entry in both architecture files before changing a forward path.
4. For new models, read `docs/ADDING_A_VLA.md` and `docs/MODEL_SCHEMA.md` completely.
5. Treat papers, checkpoint configurations, and official repositories as primary evidence. Do not rely on memory for unstable or model-specific facts.

Use `apply_patch` for repository edits. Keep temporary QA screenshots named `qa-*.png`; they are ignored and must be removed after verification.

## Architecture truth rules

### Pin one configuration

Every model entry must identify one public checkpoint or reference configuration. Do not mix dimensions from different versions, embodiments, or task configs.

When a dimension is not public or varies by robot, use a documented symbol such as `Ds`, `Da`, `Nv`, or `Ltxt`. Never invent a number to make the diagram look complete.

### Keep all views consistent

The following must describe the same computation:

- Lesson prose
- Canonical `components`
- Canonical `links`
- System, Modules, and Layers views
- Inference and training pathways
- Forward trace
- Comparison metadata
- Tensor contracts

If one representation changes, audit the others.

### Use schema version 2

New models should set `schemaVersion: 2`. Every canonical link must include:

- Valid `from` and `to` component IDs
- `mode`: `both`, `inference`, or `training`
- `kind`: `data`, `residual`, `cross`, `condition`, `skip`, or `loop`
- Exact or symbolic `tensor`

### Preserve pathway separation

Training-only targets, posterior encoders, padding heads, losses, and augmentation branches must use `mode: "training"` on the component, canonical link, diagram node, and diagram edge where applicable.

Inference-only priors, samplers, cached paths, or deployment operations must use `mode: "inference"`.

Do not rely on only the edge mode to hide a pathway-specific component.

### Represent real internal structure

For documented modules, include:

- Layer or operator name
- Input and output tensor
- Projection widths
- Attention head count and head dimension
- Normalization placement and type
- Activation function
- Repetition count
- Residual, cross-attention, condition, skip, or loop path

Use `layers` for the inspector and inventory. Use explicit diagram nodes and edges when topology matters.

## Chapter and depth conventions

Model cards have six lesson chapters:

| Index | Meaning |
|---:|---|
| `0` | Overview |
| `1` | Inputs |
| `2` | Representation/tokenization |
| `3` | Core model/generator |
| `4` | Action or world output |
| `5` | Execution, control loop, or downstream use |

Canonical components use chapters `1` through `5`. The System view is generated from these chapters.

Depth behavior:

- **System**: functional stages and module counts
- **Modules**: repeated stacks collapsed into named modules
- **Layers**: internal operations, tensor transformations, residuals, and attention paths

For complex 2-D schematics, add paired nodes with `view: "modules"` and `view: "layers"`, plus view-matched edges. Never leave a visible dangling edge after filtering.

## Diagram conventions

In `architecture-diagrams.js`:

- `N.id` is local to the SVG.
- `componentId` references a canonical component in `architecture-specs.js`.
- `layerIndex` is zero-based and must exist in that component's `layers` array.
- `E.from` and `E.to` reference SVG node IDs.
- Arrows terminate at the operation that consumes the tensor.
- Residual paths should route around the transformed branch and terminate at the add/norm node.
- Cross-attention must show the query source and memory/context K,V source.
- U-Net skips must show the correct resolution-level concatenation.
- Denoising, flow, control, and environment cycles must use loop edges.

Avoid line crossings where a routed port or rail can make ownership clearer.

## Interaction and layout constraints

Do not regress these established behaviors:

- The layer inspector is positioned beside the selected 3-D module and follows orbit, zoom, and resize.
- The inspector falls above or below only when a narrow viewport cannot fit it side-by-side.
- System/Modules/Layers and Inference/Training state are reflected in the URL.
- `step` URLs activate a shareable forward-trace state.
- Trace controls highlight corresponding 2-D and 3-D nodes.
- Popup, trace controls, title, tabs, and legends must not overlap at supported widths.
- Canvas blocks remain clickable after depth or pathway changes.
- Keyboard interaction works for diagram nodes, chapter navigation, dialogs, and playback.

Maintain the existing restrained technical-publication aesthetic. Reuse the semantic type colors from `TYPE_COLORS`; do not assign arbitrary per-model meanings to those colors.

## Comparison rules

Use normalized functional stages:

- Perception/backbone
- Fusion/conditioning
- Policy/generator
- Output unit
- Execution loop

Comparison copy should explain mechanisms, not imply an unsupported leaderboard. Keep policy-only models, VLAs, World Action Models, and world foundation platforms distinguishable by scope.

## Required validation

Run after architecture, renderer, or diagram changes:

```bash
node --check app.js
node --check architecture-specs.js
node --check architecture-diagrams.js
node scripts/validate-architecture.mjs
git diff --check
```

Run `node scripts/validate-architecture.mjs` after documentation-only changes too, because the GitHub Pages workflow always executes it.

For a new model, test at least:

```text
?model=<id>&chapter=0&depth=system
?model=<id>&chapter=3&depth=modules
?model=<id>&chapter=3&depth=layers
?model=<id>&chapter=3&depth=layers&step=2
?model=<id>&chapter=2&pathway=training&depth=layers
```

Visual QA must cover:

- Desktop and narrow layout
- System, Modules, and Layers
- Inference and training
- At least one residual/cross/skip structure
- Forward trace
- Layer inspector placement
- Comparison dialog when comparison data changed

## Deployment

Pushing `main` triggers `.github/workflows/deploy-pages.yml`, which validates architecture data before deployment.

After an authorized push:

1. Confirm the workflow completed successfully.
2. Confirm the public site returns HTTP 200.
3. Confirm the deployed static asset contains a distinctive part of the change.
4. Report the commit hash and public URL.

Do not change repository visibility, Pages settings, permissions, or secrets unless explicitly requested.

## Common failure modes

- Component is marked training-only only on its link, so it still appears during inference.
- Diagram edge points to a component ID instead of a local diagram-node ID.
- `layerIndex` no longer matches after reordering a component's layer list.
- Producer tensor uses batch-first order while consumer documentation silently assumes sequence-first.
- Modules view hides operator nodes but not their edges.
- Repeated slabs are mistaken for repeated execution when they actually represent shared weights.
- A model family name is used without pinning a checkpoint/version.
- A world model is described as if it directly emits robot actions.
- A generic fallback component remains because the spec key does not exactly match the model-card ID.

## Completion checklist

Before handing off a change, confirm:

- The requested behavior is implemented, not merely documented.
- Architecture facts are source-pinned or explicitly symbolic.
- Inference and training paths are correct.
- Dimensions, norms, activations, repetitions, and topology agree across views.
- Validation and syntax checks pass.
- Relevant visual states were inspected.
- Temporary servers, browser processes, profiles, and QA screenshots were removed.
- Only intended files are committed.
- GitHub Pages deployment is successful when publication was requested or already part of the workflow.
