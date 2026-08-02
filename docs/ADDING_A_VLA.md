# Adding a new VLA structure

This guide explains how to add a robot policy, VLA, World Action Model, or world foundation model to VLA Atlas. A complete model entry supplies four coordinated views of the same architecture:

1. Guided lesson copy in `app.js`
2. Canonical components, layers, tensors, and connections in `architecture-specs.js`
3. Hierarchical 2-D schematic in `architecture-diagrams.js`
4. Forward-trace and comparison metadata in `architecture-specs.js`

The implementation is dependency-free and loaded directly by the browser. Do not add a build step for model data.

## 1. Choose one pinned reference

Before writing code, select one public configuration that will define every displayed dimension. Record:

- Model/checkpoint name and version
- Official implementation or checkpoint configuration
- Paper or technical report
- Image size and camera/view count
- State and action dimensions
- Context and action horizons
- Model widths, attention heads, layer counts, and MLP widths
- Training-only modules
- Inference-time solver, sampler, or control loop

Do not combine dimensions from different checkpoints. If a dimension depends on embodiment or deployment, keep it symbolic, such as `Ds`, `Da`, `Nv`, or `Ltxt`.

Use the following evidence labels consistently:

- `implementation-pinned`: directly supported by the selected public implementation/configuration
- `paper-pinned`: explicitly stated in the paper but not exposed in public code
- `inferred`: derived from compatible published shapes or operations; explain the inference
- `unspecified`: not public; display a symbol or `—` rather than guessing

## 2. Add the model card and lesson

Add an object to the `models` array in `app.js`. Its `id` must exactly match the keys later added to `ARCHITECTURE_SPECS` and `ARCHITECTURE_DIAGRAMS`.

Every model currently uses six chapters. Chapter index `0` is the overview; architecture components use chapters `1` through `5`.

| Chapter | Purpose |
|---|---|
| `0` | Overview and defining architectural idea |
| `1` | Inputs |
| `2` | Representation and tokenization |
| `3` | Core policy, generator, or world model |
| `4` | Action or world output |
| `5` | Execution, replanning, or downstream use |

Minimal model-card structure:

```js
{
  id: "my-vla",
  tab: "My VLA",
  name: "My Vision-Language-Action Model",
  sceneName: "My VLA",
  category: "Generalist VLA",
  year: "2026",
  accent: "#4f7fbc",
  source: "https://example.org/project",
  chapters: [
    makeChapter("Overview", "My VLA", "One-sentence positioning.", [
      "Explain the model's defining idea.",
      "Explain what makes its structure different from adjacent models.",
    ], "One concise takeaway."),
    makeChapter("Inputs", "What enters the policy", "Input summary.", ["...", "..."], "..."),
    makeChapter("Representation", "How inputs become features", "Representation summary.", ["...", "..."], "..."),
    makeChapter("Core model", "How the model computes actions", "Core summary.", ["...", "..."], "..."),
    makeChapter("Action output", "What the model predicts", "Output summary.", ["...", "..."], "..."),
    makeChapter("Control loop", "How it runs on the robot", "Loop summary.", ["...", "..."], "..."),
  ],
  components: [],
  links: [],
  compare: {},
},
```

The empty architecture fields are intentional. After the `models` array is declared, `architecture-specs.js` is merged into the matching model card and replaces them.

## 3. Define the canonical architecture

Add the same model key to `window.ARCHITECTURE_SPECS` in `architecture-specs.js`. New structures should use schema version `2`.

```js
"my-vla": {
  schemaVersion: 2,
  reference: "My VLA · checkpoint-name",
  configNote: "B symbolic · Nv=2 · d=1024 · horizon=16",
  sources: [
    ["Official implementation", "https://github.com/org/repo"],
    ["Paper", "https://arxiv.org/abs/0000.00000"],
  ],
  provenance: {
    status: "implementation-pinned",
    confidence: "Verified against the selected public checkpoint and paper",
    reviewed: "YYYY-MM-DD",
    evidence: {
      visionEncoder: "Official checkpoint vision configuration",
      policy: "Official policy implementation",
    },
  },
  symbols: [
    "B batch",
    "Nv camera views",
    "Ltxt language tokens",
    "Da embodiment action dimension",
  ],
  components: [],
  links: [],
  flowRows: { inference: [], training: [] },
  flowSteps: { inference: [], training: [] },
  validation: { tensorContracts: [] },
  compare: {},
},
```

### Add components

Use the local `node()` helper. Coordinates describe the exploded 3-D canvas.

```js
node("visionEncoder", "Vision encoder", "vision", 2, -2.5, -4.5, "B×Nv×Nimg×1024", {
  w: 3.6,
  d: 1.7,
  input: "B×Nv×3×H×W",
  op: "Patch embedding + Transformer",
  norm: "LayerNorm",
  activation: "GELU",
  repeat: 24,
  repeatLabel: "24 Transformer blocks",
  description: "Encodes every camera view into spatial visual tokens.",
  layers: [
    layer("Patch projection", "B·Nv×3×H×W", "B·Nv×Nimg×1024", "Conv2d p×p, stride p"),
    layer("Pre-norm self-attention", "B·Nv×Nimg×1024", "B·Nv×Nimg×1024", "16 heads ×64", "LayerNorm", "softmax"),
    layer("MLP", "B·Nv×Nimg×1024", "B·Nv×Nimg×1024", "Linear 1024→4096→1024", "LayerNorm", "GELU"),
  ],
}),
```

Coordinate conventions:

- `x`: horizontal branch placement
- `y`: forward stage; current models use negative values for inputs and positive values for outputs
- `z`: depth offset
- `w`, `h`, `d`: rendered block dimensions
- `repeat`: number of repeated slabs in Layers view
- `gap`: distance between repeated slabs
- `twist`: optional horizontal/depth stagger
- `grid`: decorative tensor grid `[columns, rows]`

Each component must have:

- A unique stable `id`
- One semantic `type`: `vision`, `language`, `state`, `action`, `world`, or `compute`
- A chapter from `1` to `5`
- An output tensor shape
- A description that explains its role, not merely its name

Use `mode: "training"` or `mode: "inference"` for pathway-specific components. The default is `both`.

### Add internal layers

Use `layers` whenever a module contains documented operations worth expanding. Every layer requires:

- `label`
- `input`
- `output`
- `op`
- `norm`, when applicable
- `activation`, when applicable

Represent residual additions and cross-attention explicitly in the 2-D diagram, even when the component's `layers` array describes them in prose.

### Add typed connections

Schema-v2 links must declare the tensor moving across the connection.

```js
link("visionEncoder", "fusion", "both", "visual tokens", {
  tensor: "B×(Nv·Nimg)×1024",
  kind: "data",
}),
link("languageEncoder", "policy", "both", "language K,V", {
  tensor: "B×Ltxt×1024",
  kind: "cross",
}),
link("stateEncoder", "policy", "both", "state condition", {
  tensor: "B×1×1024",
  kind: "condition",
}),
```

Allowed connection kinds:

| Kind | Meaning |
|---|---|
| `data` | Ordinary forward tensor |
| `residual` | Residual/add path |
| `cross` | Cross-attention memory or context |
| `condition` | Conditioning signal, FiLM, AdaLN, state, or time |
| `skip` | U-Net or other long skip connection |
| `loop` | Denoising, flow integration, replanning, or environment loop |

The endpoint IDs must refer to canonical component IDs, not 2-D diagram-node IDs.

## 4. Add the hierarchical 2-D diagram

Add the model key to `window.ARCHITECTURE_DIAGRAMS` in `architecture-diagrams.js`.

The diagram uses three helpers:

```js
N(id, label, shape, type, chapter, x, y, width, height, options)
G(id, label, note, chapter, x, y, width, height, options)
E(from, to, kind, options)
```

Basic structure:

```js
"my-vla": {
  width: 900,
  height: 1050,
  groups: [
    G("g-vision", "VISION ENCODER", "24 blocks · width 1024", 2, 25, 110, 300, 250),
    G("g-policy", "ACTION POLICY", "cross-attends to language and vision", 3, 150, 430, 600, 350),
  ],
  nodes: [
    N("rgb", "Camera views", "B×Nv×3×H×W", "vision", 1, 55, 30, 210, 50, { componentId: "cameras" }),
    N("vision", "Vision encoder", "B×Nv×Nimg×1024", "vision", 2, 65, 160, 230, 55, { componentId: "visionEncoder" }),
    N("policy", "Policy Transformer", "B×Ta×1024", "compute", 3, 300, 520, 300, 60, { componentId: "policy" }),
  ],
  edges: [
    E("rgb", "vision"),
    E("vision", "policy", "cross", { label: "visual K,V" }),
  ],
  legend: [["data", "forward"], ["cross", "cross-attention"]],
},
```

Important distinctions:

- `N.id` is local to the SVG diagram.
- `componentId` must reference an ID from `architecture-specs.js`.
- `layerIndex` selects an entry from that component's `layers` array for the inspector.
- `E.from` and `E.to` reference SVG `N.id` values.
- `mode` controls inference/training visibility.
- `view` controls Modules/Layers visibility.

### Modules and Layers views

For complex modules, define one summary node for Modules view and exact operation nodes for Layers view:

```js
N("policy-stack", "Policy Transformer ×12", "B×Ta×1024", "compute", 3, 300, 520, 300, 110, {
  componentId: "policy",
  view: "modules",
}),
N("policy-attn", "Cross-attention", "Q:Ta · KV:Nctx", "language", 3, 300, 520, 300, 45, {
  componentId: "policy",
  layerIndex: 0,
  view: "layers",
}),
```

Give alternate edges the same `view` so hidden nodes never leave dangling connections. The System view is generated automatically from component chapters and does not require separate SVG data.

### Routing edges

Use ports and rails to keep branches readable:

```js
E("memory", "cross-attn", "cross", {
  fromPort: "right",
  toPort: "left",
  route: "right",
  label: "memory K,V",
}),
```

Ports: `top`, `bottom`, `left`, `right`

Routes: `left`, `right`, `above`, or the default midpoint route

## 5. Define forward traces

Add explicit inference and training traces to `flowSteps`. A trace step highlights canonical component IDs in both the 2-D and 3-D views.

```js
flowSteps: {
  inference: [
    {
      title: "Encode observations and instruction",
      tensor: "visual B×Nimg×1024 · text B×Ltxt×1024",
      nodes: ["visionEncoder", "languageEncoder"],
      note: "The two encoders run in parallel before policy conditioning.",
    },
    {
      title: "Generate an action chunk",
      tensor: "B×Ta×Da",
      nodes: ["policy", "actionHead", "actions"],
      note: "The action head maps policy features into continuous robot commands.",
    },
  ],
  training: [
    {
      title: "Encode the demonstrated target",
      tensor: "B×Ta×Da",
      nodes: ["targetActions", "trainingEncoder"],
      note: "This branch is unavailable during inference.",
    },
  ],
},
```

Keep steps short enough to explain one transformation. Prefer 8–15 steps for a full inference path. `flowRows` remains useful as a compact branch summary and provides a fallback trace for models without explicit `flowSteps`.

## 6. Add tensor contracts

Tensor contracts turn selected important edges into executable assertions:

```js
validation: {
  tensorContracts: [
    { from: "visionEncoder", to: "fusion", tensor: "B×(Nv·Nimg)×1024" },
    { from: "policy", to: "actionHead", tensor: "B×Ta×1024" },
    { from: "actionHead", to: "actions", tensor: "B×Ta×Da" },
  ],
},
```

The contract tensor must exactly match the declared output of `from`, and a compatible canonical link must exist. Add contracts at branch convergence, cross-attention memory, width-changing projections, and outputs.

## 7. Add normalized comparison metadata

Use the shared keys below so the comparison dialog aligns equivalent decisions:

```js
compare: {
  Scope: "Generalist VLA",
  Inputs: "images + language + proprioception",
  Backbone: "ViT + language Transformer",
  Fusion: "visual and state tokens cross-attend in the policy",
  Generator: "12-layer autoregressive action Transformer",
  Output: "16×Da continuous action chunk",
  "Temporal loop": "execute prefix + replan",
  "World prediction": "No",
},
```

Describe architectural mechanisms, not performance claims. Models of different scope should not be presented as one leaderboard.

## 8. Validate locally

Run syntax and graph checks:

```bash
node --check app.js
node --check architecture-specs.js
node --check architecture-diagrams.js
node scripts/validate-architecture.mjs
```

Start the static site:

```bash
python3 -m http.server 8000
```

Useful test URLs:

```text
http://localhost:8000/?model=my-vla&chapter=0&depth=system
http://localhost:8000/?model=my-vla&chapter=3&depth=modules
http://localhost:8000/?model=my-vla&chapter=3&depth=layers
http://localhost:8000/?model=my-vla&chapter=3&depth=layers&step=2
http://localhost:8000/?model=my-vla&chapter=2&pathway=training&depth=layers
```

## 9. Visual QA checklist

- Model tab, title, reference, and sources are correct.
- System stages appear in forward order and contain sensible module counts.
- Modules view collapses repeated stacks.
- Layers view exposes operations, dimensions, norms, and activations.
- Inference contains no posterior, target, padding, or loss-only branch.
- Training shows every training-only branch.
- Residual, cross-attention, condition, skip, and loop edges use the correct style.
- Every diagram node opens the intended component or internal layer.
- Inspector popup stays beside the selected 3-D block.
- Forward trace highlights the expected nodes at every step.
- Comparison stages use equivalent concepts across both models.
- Layout remains usable at desktop and narrow widths.

## 10. Definition of done

A new structure is ready when:

- It is tied to a named public reference configuration.
- All public numerical dimensions are shown and uncertain values remain symbolic.
- Canonical graph, nested schematic, trace, and comparison agree.
- Inference and training paths are separately correct.
- Primary sources and evidence status are visible.
- `node scripts/validate-architecture.mjs` passes.
- The five depth/pathway URLs above have been visually reviewed.

See [MODEL_SCHEMA.md](MODEL_SCHEMA.md) for the complete field reference.
