# VLA Atlas model schema reference

This reference documents the browser-loaded data structures used by VLA Atlas. For the end-to-end workflow, see [ADDING_A_VLA.md](ADDING_A_VLA.md).

## File ownership

| File | Owns |
|---|---|
| `app.js` | Model registry, lesson copy, renderer, interactions, comparison UI |
| `architecture-specs.js` | Canonical model graph, components, tensors, layers, traces, provenance, comparison facts |
| `architecture-diagrams.js` | Hierarchical SVG groups, nodes, and exact edge routing |
| `scripts/validate-architecture.mjs` | Structural and tensor-contract checks |

The model key must be identical in the `models` array, `ARCHITECTURE_SPECS`, and `ARCHITECTURE_DIAGRAMS`.

## Model specification

| Field | Type | Required | Description |
|---|---|---:|---|
| `schemaVersion` | number | For new models | Use `2` for typed links and tensor validation |
| `reference` | string | Yes | Exact checkpoint or public configuration |
| `configNote` | string | Yes | Compact list of dimensions and hyperparameters |
| `sources` | `[label, url][]` | Yes | Primary implementation, checkpoint, and paper links |
| `provenance` | object | Recommended | Evidence status, review date, and per-component evidence |
| `symbols` | string[] | Yes | Definitions for every symbolic tensor dimension |
| `components` | component[] | Yes | Canonical 3-D modules and internal layers |
| `links` | link[] | Yes | Canonical model-level graph |
| `flowRows` | object | Recommended | Compact branched inference/training summaries |
| `flowSteps` | object | Recommended | Animated inference/training explanation |
| `validation` | object | Recommended | Important tensor contracts |
| `compare` | object | Yes | Normalized comparison facts |

## Component helper

```js
node(id, label, type, chapter, x, y, shape, options)
```

### Positional fields

| Field | Type | Default | Description |
|---|---|---:|---|
| `x` | number | — | Horizontal branch position |
| `y` | number | — | Forward-stage position |
| `z` | number | `0` | Depth offset |
| `w` | number | `3.0` | Block width |
| `h` | number | `0.38` | Block height |
| `d` | number | `1.25` | Block depth |
| `grid` | `[number, number]` | `[8, 2]` | Decorative tensor grid |
| `repeat` | number | `1` | Number of repeated slabs |
| `gap` | number | `0.28` | Slab spacing |
| `twist` | number | `0` | Repetition stagger |

### Semantic fields

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | `—` | Input tensor or named multi-input signature |
| `output` | string | `shape` | Canonical output tensor |
| `op` | string | `—` | Operation or module behavior |
| `norm` | string | `—` | Normalization |
| `activation` | string | `—` | Activation or probability transform |
| `repeatLabel` | string | empty | Human-readable repetition explanation |
| `mode` | enum | `both` | `both`, `inference`, or `training` |
| `description` | string | empty | Educational role description |
| `role` | string | `module` | Optional semantic role for future tooling |
| `confidence` | string | empty | Optional component-level evidence status |
| `sourceRefs` | string[] | `[]` | Optional provenance identifiers |
| `accepts` | string[] | `[]` | Accepted tensors for multi-input/convergence modules |
| `layers` | layer[] | `[]` | Expandable internal operations |

Allowed component types and their visual meaning:

| Type | Meaning |
|---|---|
| `vision` | Images, visual tokenizer, CNN, ViT |
| `language` | Text, language encoder, semantic context, cross-attention |
| `state` | Proprioception, robot state, execution/environment state |
| `action` | Action queries, noise, decoder, trajectory, command |
| `world` | Video/world latent, future-state prediction |
| `compute` | Fusion, Transformer, diffusion/flow solver, generic computation |

## Layer helper

```js
layer(label, input, output, op, norm = "—", activation = "—", description = "")
```

`label`, `input`, `output`, and `op` are validated as required. A diagram node may reference an internal operation using `componentId` plus zero-based `layerIndex`.

Layer arrays describe module internals. They do not replace graph edges when a residual, cross-attention, skip, or conditioning relationship needs to be visually explicit.

## Canonical link helper

```js
link(from, to, mode = "both", label = "", options = {})
```

| Field | Type | Required in schema v2 | Description |
|---|---|---:|---|
| `from` | component ID | Yes | Producer component |
| `to` | component ID | Yes | Consumer component |
| `mode` | enum | Yes | `both`, `inference`, or `training` |
| `label` | string | No | Human-readable edge role |
| `options.tensor` | string | Yes | Tensor carried by the edge |
| `options.kind` | enum | Yes | Defaults to `data` |

Allowed kinds: `data`, `residual`, `cross`, `condition`, `skip`, `loop`.

## Forward summaries

`flowRows` is a compact branch representation:

```js
flowRows: {
  inference: [
    ["vision", "visionEncoder", "fusion", "policy"],
    ["language", "languageEncoder", "policy"],
    ["state", "stateEncoder", "policy"],
  ],
  training: [
    ["targetActions", "posterior", "policy", "training loss"],
  ],
},
```

Known component IDs become interactive nodes. Unknown strings are treated as explanatory literals.

`flowSteps` drives the animated trace:

| Field | Type | Required | Description |
|---|---|---:|---|
| `title` | string | Yes | Transformation name |
| `tensor` | string | Yes | Shape or tensor summary at this step |
| `nodes` | component ID[] | Yes | Components highlighted together |
| `note` | string | Yes | Explanation shown as trace context |

Every trace node must reference a canonical component.

## Tensor contracts

```js
validation: {
  tensorContracts: [
    { from: "encoder", to: "decoder", tensor: "B×N×d", mode: "inference", role: "cross-attention memory" },
  ],
},
```

| Field | Required | Description |
|---|---:|---|
| `from` | Yes | Producer component ID |
| `to` | Yes | Consumer component ID |
| `tensor` | Yes | Must exactly equal the producer's declared `output` |
| `mode` | No | Restricts the matching link |
| `role` | No | Human-readable contract purpose |

## Diagram specification

Each diagram declares `width`, `height`, `groups`, `nodes`, `edges`, and `legend`.

### Diagram node

```js
N(id, label, shape, type, chapter, x, y, width = 150, height = 48, options = {})
```

| Option | Description |
|---|---|
| `componentId` | Canonical component opened by click/keyboard |
| `layerIndex` | Zero-based internal layer opened by the inspector |
| `mode` | `both`, `inference`, or `training` |
| `view` | `all`, `modules`, or `layers` |

Diagram-node IDs are local to the SVG and may differ from canonical component IDs.

### Diagram group

```js
G(id, label, note, chapter, x, y, width, height, options)
```

| Option | Description |
|---|---|
| `nested` | Uses the solid nested-container style |
| `mode` | Pathway visibility |
| `view` | Depth visibility |

### Diagram edge

```js
E(from, to, kind = "data", options = {})
```

`from` and `to` reference diagram-node IDs, not canonical components.

| Option | Values | Description |
|---|---|---|
| `mode` | `both`, `inference`, `training` | Pathway visibility |
| `view` | `all`, `modules`, `layers` | Depth visibility |
| `fromPort` | `top`, `bottom`, `left`, `right` | Source anchor |
| `toPort` | `top`, `bottom`, `left`, `right` | Target anchor |
| `route` | `left`, `right`, `above` | Orthogonal routing rail |
| `label` | string | Edge annotation |

## Tensor notation

Use `×` between axes and define symbols in `symbols`.

Recommended symbols:

| Symbol | Meaning |
|---|---|
| `B` | Batch |
| `T`, `Ta`, `To` | Generic, action, or observation time |
| `Nv`, `Nc` | View or camera count |
| `N`, `Nimg`, `Nctx` | Token counts |
| `D`, `d` | Feature/model width |
| `Ds`, `Da` | State/action dimension |
| `Ltxt` | Language-token count |
| `H`, `W` | Spatial height and width |
| `F` | Video-frame count |

Conventions:

- Keep the axis order used by the selected implementation.
- Make required transposes explicit in `op` or a layer row.
- Use `·` for folded batch/view axes only when the implementation actually folds them.
- Use parentheses for concatenated or computed dimensions.
- Keep unknown embodiment-specific dimensions symbolic.
- Do not report a numerical dimension solely because another checkpoint uses it.

## Provenance

Model-level provenance appears in the layer inspector. Use `provenance.evidence` keys that match canonical component IDs.

```js
provenance: {
  status: "implementation-pinned",
  confidence: "Verified against the official checkpoint and repository",
  reviewed: "YYYY-MM-DD",
  evidence: {
    policy: "Repository policy configuration",
    actionHead: "Checkpoint action-head implementation",
  },
},
```

Primary sources are preferred over secondary summaries. When documentation and implementation disagree, pin the entry to one named artifact and explain the discrepancy in `configNote`, `description`, or evidence text.

## Validator guarantees

`node scripts/validate-architecture.mjs` currently checks:

- Specification and diagram files load
- Every model has components, links, and primary sources
- Component IDs are unique
- Component types, chapters, outputs, and modes are valid
- Internal layers contain required fields
- Canonical link endpoints exist
- Schema-v2 links declare tensors
- Trace modes, copy, and target component IDs are valid
- Tensor contracts have matching links and producer outputs
- Every model has a diagram
- Diagram-node IDs are unique
- Diagram `componentId` and `layerIndex` references exist
- Diagram edge endpoints, kinds, modes, and views are valid

The validator does not prove that a research claim is correct. Source review and visual QA remain required.
