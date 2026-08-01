/*
 * Architecture data is deliberately separated from the renderer. Every tab is
 * pinned to one named public reference configuration so tensor widths are not
 * silently mixed across checkpoints or robot embodiments.
 */
(function () {
  const layer = (label, input, output, op, norm = "—", activation = "—", description = "") => ({
    label, input, output, op, norm, activation, description,
  });

  const node = (id, label, type, chapter, x, y, shape, options = {}) => ({
    id, label, type, chapter, x, y, z: options.z || 0,
    w: options.w || 3.0, h: options.h || 0.38, d: options.d || 1.25,
    grid: options.grid || [8, 2], shape,
    input: options.input || "—", output: options.output || shape,
    op: options.op || "—", norm: options.norm || "—", activation: options.activation || "—",
    repeat: options.repeat || 1, gap: options.gap || 0.28, twist: options.twist || 0,
    mode: options.mode || "both", repeatLabel: options.repeatLabel || "",
    description: options.description || "",
    layers: options.layers || [],
  });

  const link = (from, to, mode = "both", label = "") => ({ from, to, mode, label });

  window.ARCHITECTURE_SPECS = {
    act: {
      reference: "ACT · ALOHA, 4-camera example",
      configNote: "B symbolic · Nc=4 · 480×640 RGB · state/action=14 · chunk=100 · d=512",
      sources: [
        ["ACT implementation", "https://github.com/tonyzhaozh/act"],
        ["ACT paper", "https://arxiv.org/abs/2304.13705"],
      ],
      symbols: ["B batch", "Nc cameras (4 here)", "T chunk (100)", "Da action dim (14)", "d model width (512)"],
      components: [
        node("cameras", "RGB cameras × Nc", "vision", 1, -2.5, -7.1, "B×4×3×480×640", {
          w: 3.5, d: 1.9, grid: [8, 4], op: "ImageNet channel normalization",
          description: "Four synchronized ALOHA views in this concrete example; the implementation accepts a configurable camera list.",
        }),
        node("qpos", "Current joint state", "state", 1, 2.0, -7.0, "B×14", {
          w: 2.5, d: 0.8, grid: [14, 1], op: "Normalize with dataset statistics",
          description: "Current 14-dimensional bimanual joint state.",
        }),
        node("demoActions", "Demonstrated actions", "action", 1, 4.6, -6.8, "B×100×14", {
          w: 2.8, mode: "training", grid: [12, 3], op: "Training target only",
          description: "The demonstrated action chunk is used only by the CVAE posterior and the reconstruction loss.",
        }),
        node("resnet", "Shared ResNet-18", "vision", 2, -2.9, -5.1, "B·4×512×15×20", {
          w: 3.35, d: 1.65, repeat: 4, gap: 0.18, repeatLabel: "shared across 4 cameras",
          op: "CNN feature extraction", norm: "FrozenBatchNorm2d", activation: "ReLU",
          description: "One shared ResNet-18 maps every camera image to a stride-32 spatial feature map.",
          layers: [
            layer("Conv 7×7, s2", "B·4×3×480×640", "B·4×64×240×320", "Conv2d 3→64", "Frozen BN", "ReLU"),
            layer("MaxPool 3×3, s2", "B·4×64×240×320", "B·4×64×120×160", "MaxPool2d", "—", "—"),
            layer("Residual stage 1 ×2", "B·4×64×120×160", "B·4×64×120×160", "2 BasicBlocks: Conv3×3 → Conv3×3 + skip", "Frozen BN", "ReLU"),
            layer("Residual stage 2 ×2", "B·4×64×120×160", "B·4×128×60×80", "First block stride 2", "Frozen BN", "ReLU"),
            layer("Residual stage 3 ×2", "B·4×128×60×80", "B·4×256×30×40", "First block stride 2", "Frozen BN", "ReLU"),
            layer("Residual stage 4 ×2", "B·4×256×30×40", "B·4×512×15×20", "First block stride 2", "Frozen BN", "ReLU"),
          ],
        }),
        node("inputProj", "Image 1×1 projection", "vision", 2, -2.9, -3.55, "B×4×512×15×20", {
          w: 3.3, op: "Conv2d 512→512, kernel 1", description: "Projects each final ResNet feature map into ACT's 512-wide Transformer space.",
          layers: [layer("1×1 channel projection", "B·4×512×15×20", "B·4×512×15×20", "Conv2d 512→512, k1", "—", "Linear")],
        }),
        node("imageTokens", "Spatial image tokens", "vision", 2, -2.4, -2.25, "1200×B×512", {
          w: 3.6, op: "Concatenate cameras along width; flatten H×W", description: "Four 15×20 maps become 1,200 ordered spatial tokens with sine positional encodings.",
          layers: [
            layer("Camera-width concat", "4 × [B×512×15×20]", "B×512×15×80", "torch.cat(width)", "—", "—"),
            layer("Spatial flatten", "B×512×15×80", "1200×B×512", "flatten + permute", "—", "—"),
            layer("2D sine position", "15×80 grid", "1200×B×512", "PositionEmbeddingSine", "L2 normalize", "sin / cos"),
          ],
        }),
        node("qposProj", "Proprio projection", "state", 2, 1.15, -4.55, "1×B×512", {
          w: 2.45, input: "B×14", op: "Linear 14→512", description: "Creates one proprioceptive memory token.",
          layers: [layer("Joint-state linear", "B×14", "1×B×512", "Linear 14→512")],
        }),
        node("prior", "Inference prior z=0", "compute", 2, 3.75, -4.9, "B×32", {
          w: 2.3, mode: "inference", op: "Deterministic zero latent", description: "The released ACT inference path uses a zero 32-D latent, not demonstrated actions.",
        }),
        node("cvae", "CVAE posterior encoder", "compute", 2, 4.2, -3.45, "B×32", {
          w: 3.2, d: 1.45, mode: "training", repeat: 4, repeatLabel: "4 encoder layers", op: "Transformer encoder", norm: "LayerNorm", activation: "ReLU",
          description: "Training-only posterior: [CLS], current joints, and 100 demonstrated actions are encoded into μ and logσ² for a 32-D latent.",
          layers: [
            layer("Action embedding", "B×100×14", "100×B×512", "Linear 14→512"),
            layer("Joint embedding", "B×14", "1×B×512", "Linear 14→512"),
            layer("[CLS] + qpos + actions", "1+1+100 tokens", "102×B×512", "Concatenate + learned positions"),
            layer("Posterior encoder ×4", "102×B×512", "102×B×512", "MHA 8 heads + FFN 512→3200→512", "LayerNorm", "ReLU"),
            layer("Latent statistics", "B×512 [CLS]", "B×64", "Linear 512→64; split μ/logσ²", "—", "—"),
            layer("Reparameterize", "μ, logσ²: B×32", "z: B×32", "μ + σ⊙ε", "—", "Gaussian ε"),
          ],
        }),
        node("latentProj", "Latent projection", "compute", 2, 3.3, -2.1, "1×B×512", {
          w: 2.45, input: "B×32", op: "Linear 32→512", description: "Maps the training posterior sample—or the inference zero prior—into one memory token.",
          layers: [layer("Latent linear", "B×32", "1×B×512", "Linear 32→512")],
        }),
        node("memoryInput", "Encoder source tokens", "compute", 3, 0, -1.25, "1202×B×512", {
          w: 4.35, d: 1.55, op: "Concat [latent, qpos, image tokens]", description: "Correct ACT encoder order: latent and proprio tokens are prepended to 1,200 image tokens.",
        }),
        node("encoder", "Transformer encoder ×4", "compute", 3, -0.5, 0.05, "1202×B×512", {
          w: 4.5, d: 2.0, repeat: 4, gap: 0.31, repeatLabel: "4 layers · shared shape", norm: "post-LayerNorm", activation: "ReLU", op: "Self-attention + FFN",
          description: "Four post-norm encoder layers contextualize every source token.",
          layers: [
            layer("Multi-head self-attention", "1202×B×512", "1202×B×512", "8 heads × 64", "—", "softmax"),
            layer("Residual + norm", "1202×B×512", "1202×B×512", "Dropout 0.1 + residual", "LayerNorm 512", "—"),
            layer("Feed-forward up", "1202×B×512", "1202×B×3200", "Linear 512→3200", "—", "ReLU"),
            layer("Feed-forward down", "1202×B×3200", "1202×B×512", "Dropout 0.1 + Linear 3200→512", "—", "—"),
            layer("Residual + norm", "1202×B×512", "1202×B×512", "Dropout 0.1 + residual", "LayerNorm 512", "—"),
          ],
        }),
        node("queries", "Learned action queries", "action", 3, 3.55, 0.1, "100×B×512", {
          w: 2.8, op: "100 learned positional embeddings", description: "One learned query position for every future action in the chunk; decoder content starts at zero.",
        }),
        node("decoder", "Transformer decoder ×7", "compute", 3, 0.35, 2.1, "100×B×512", {
          w: 4.55, d: 2.1, repeat: 7, gap: 0.29, repeatLabel: "7 decoder layers", norm: "post-LayerNorm + final LayerNorm", activation: "ReLU", op: "Self-attention + cross-attention + FFN",
          description: "Seven decoder layers let 100 action queries communicate, then cross-attend to the encoded visual/proprio/latent memory.",
          layers: [
            layer("Query self-attention", "100×B×512", "100×B×512", "8 heads × 64", "LayerNorm after residual", "softmax"),
            layer("Cross-attention to memory", "Q:100; KV:1202 tokens", "100×B×512", "8 heads × 64", "LayerNorm after residual", "softmax"),
            layer("Feed-forward up", "100×B×512", "100×B×3200", "Linear 512→3200", "—", "ReLU"),
            layer("Feed-forward down", "100×B×3200", "100×B×512", "Linear 3200→512 + residual", "LayerNorm 512", "—"),
          ],
        }),
        node("actionHead", "Action head", "action", 4, -1.7, 4.8, "B×100×14", {
          w: 3.05, op: "Linear 512→14", description: "Projects each decoded query into a 14-D joint target.",
          layers: [layer("Continuous action linear", "B×100×512", "B×100×14", "Linear 512→14", "—", "linear")],
        }),
        node("padHead", "Padding head", "action", 4, 2.0, 4.75, "B×100×1", {
          w: 2.65, op: "Linear 512→1", description: "Predicts whether a sequence position is padding during variable-length training.",
          layers: [layer("Padding logit", "B×100×512", "B×100×1", "Linear 512→1", "—", "logit")],
        }),
        node("chunk", "Predicted action chunk", "action", 4, 0, 5.9, "B×100×14", {
          w: 4.6, d: 1.5, grid: [14, 4], op: "Denormalize to joint targets", description: "All 100 future 14-D targets are produced in one forward pass.",
        }),
        node("ensemble", "Temporal ensemble", "action", 5, 0, 7.15, "B×14 now", {
          w: 3.5, repeat: 3, gap: 0.18, repeatLabel: "overlapping chunks", op: "Exponentially weighted average", description: "Overlapping predictions for the current timestep are blended; newer chunks receive higher weight.",
          layers: [
            layer("Collect overlaps", "K predictions × B×14", "K×B×14", "Align by execution timestamp"),
            layer("Exponential weights", "ages 0…K−1", "K scalars", "exp(−0.01·age)", "sum-to-one", "exp"),
            layer("Blend current action", "K×B×14", "B×14", "Weighted sum"),
          ],
        }),
      ],
      links: [
        link("cameras", "resnet"), link("resnet", "inputProj"), link("inputProj", "imageTokens"), link("imageTokens", "memoryInput"),
        link("qpos", "qposProj"), link("qposProj", "memoryInput"), link("prior", "latentProj", "inference"),
        link("demoActions", "cvae", "training"), link("qpos", "cvae", "training"), link("cvae", "latentProj", "training"),
        link("latentProj", "memoryInput"), link("memoryInput", "encoder"), link("encoder", "decoder"), link("queries", "decoder"),
        link("decoder", "actionHead"), link("decoder", "padHead", "training"), link("actionHead", "chunk"), link("chunk", "ensemble"),
      ],
      flowRows: {
        inference: [
          ["cameras", "resnet", "inputProj", "imageTokens", "memoryInput", "encoder"],
          ["qpos", "qposProj", "memoryInput"],
          ["prior", "latentProj", "memoryInput"],
          ["encoder", "decoder", "actionHead", "chunk", "ensemble"],
          ["queries", "decoder"],
        ],
        training: [
          ["cameras", "resnet", "inputProj", "imageTokens", "memoryInput", "encoder"],
          ["qpos", "qposProj", "memoryInput"],
          ["demoActions", "cvae", "latentProj", "memoryInput"],
          ["encoder", "decoder", "actionHead", "chunk"],
          ["queries", "decoder"],
          ["decoder", "padHead"],
        ],
      },
      compare: { Scope: "ACT ALOHA specialist policy", Inputs: "4× RGB 480×640 + qpos 14", Backbone: "Shared ResNet-18 → 512", Generator: "4-layer encoder + 7-layer decoder", Output: "100×14 action chunk", "Temporal loop": "Exponential temporal ensemble", "World prediction": "No" },
    },

    "diffusion-policy": {
      reference: "Diffusion Policy · Push-T image CNN",
      configNote: "To=2 · Ta=16 · execute=8 · Da=2 · U-Net=[512,1024,2048] · 100 denoise steps",
      sources: [["Diffusion Policy implementation", "https://github.com/real-stanford/diffusion_policy"], ["Diffusion Policy paper", "https://arxiv.org/abs/2303.04137"]],
      symbols: ["B batch", "To obs horizon (2)", "Ta action horizon (16)", "Da action dim (2)", "H temporal length"],
      components: [
        node("rgb", "RGB observation history", "vision", 1, -2.4, -7.0, "B×2×3×96×96", { w: 3.4, d: 1.8, grid: [8, 4], op: "Two most recent frames", description: "The published Push-T image workspace uses two 96×96 RGB observations." }),
        node("robotState", "Agent position history", "state", 1, 2.1, -6.9, "B×2×2", { w: 2.7, op: "Low-dimensional observation", description: "Two recent 2-D agent positions accompany the two images." }),
        node("crop", "Crop + normalize", "vision", 2, -3.1, -5.4, "B·2×3×84×84", { w: 3.0, op: "84×84 crop; ImageNet normalize", description: "Random crop is used for training and a fixed center crop for evaluation.", layers: [layer("84×84 crop", "B·2×3×96×96", "B·2×3×84×84", "RandomCrop(train) / center crop(eval)"), layer("Channel normalize", "B·2×3×84×84", "B·2×3×84×84", "ImageNet mean/std")] }),
        node("obsResnet", "ResNet-18 encoder", "vision", 2, -2.9, -3.85, "B·2×512×3×3", { w: 3.4, d: 1.6, repeat: 4, repeatLabel: "4 residual stages", norm: "BatchNorm", activation: "ReLU", description: "A ResNet-18 encodes each cropped image independently.", layers: [
          layer("Conv 7×7, s2", "B·2×3×84×84", "B·2×64×42×42", "Conv2d 3→64", "BatchNorm", "ReLU"),
          layer("MaxPool + stage 1 ×2", "B·2×64×42×42", "B·2×64×21×21", "MaxPool; 2 BasicBlocks", "BatchNorm", "ReLU"),
          layer("Residual stage 2 ×2", "B·2×64×21×21", "B·2×128×11×11", "First block stride 2", "BatchNorm", "ReLU"),
          layer("Residual stage 3 ×2", "B·2×128×11×11", "B·2×256×6×6", "First block stride 2", "BatchNorm", "ReLU"),
          layer("Residual stage 4 ×2", "B·2×256×6×6", "B·2×512×3×3", "First block stride 2", "BatchNorm", "ReLU"),
        ] }),
        node("spatialSoftmax", "SpatialSoftmax · 32 keypoints", "vision", 2, -2.4, -2.35, "B×2×64", { w: 3.2, op: "32 expected (x,y) coordinates", activation: "spatial softmax", description: "Thirty-two soft keypoints compress each spatial feature map to 64 numbers.", layers: [layer("Keypoint logits", "B·2×512×3×3", "B·2×32×3×3", "Learned 1×1 keypoint projection"), layer("Spatial probability", "B·2×32×3×3", "B·2×32×9", "softmax over 3×3", "—", "softmax"), layer("Expected coordinates", "B·2×32×9", "B×2×64", "Σp(x,y)")] }),
        node("obsCondition", "Global observation condition", "state", 2, 1.4, -2.3, "B×132", { w: 3.15, input: "B×2×(64+2)", op: "Concat vision64 + state2; flatten To", description: "Per timestep, 64 visual coordinates and 2 agent-state values form 66 features; two timesteps flatten to 132." }),
        node("noise", "Gaussian action sample", "action", 2, 4.15, -3.8, "B×16×2", { w: 2.8, op: "x₁ ~ N(0,I)", description: "Inference begins from random noise already shaped like the desired 16-step, 2-D trajectory." }),
        node("timeEmbed", "Diffusion-time embedding", "compute", 3, 3.65, -1.15, "B×128", { w: 2.7, op: "Sinusoidal 128 → Linear 512 → Mish → Linear 128", activation: "Mish", description: "The current scheduler timestep is embedded and concatenated with the 132-D global condition, giving 260 conditioning features.", layers: [layer("Sinusoidal position", "B×1", "B×128", "sin/cos frequencies"), layer("Time MLP up", "B×128", "B×512", "Linear 128→512", "—", "Mish"), layer("Time MLP down", "B×512", "B×128", "Linear 512→128")] }),
        node("unet", "Conditional 1-D U-Net", "compute", 3, 0, 0.15, "B×16×2", { w: 4.8, d: 2.1, repeat: 6, gap: 0.25, repeatLabel: "100 shared denoising passes", op: "ConditionalResidualBlock1D", norm: "GroupNorm(8)", activation: "Mish", description: "A weight-shared conditional U-Net predicts trajectory noise at every one of 100 scheduler steps. FiLM uses the 260-D time+observation condition.", layers: [
          layer("Input transpose", "B×16×2", "B×2×16", "rearrange B,T,C → B,C,T"),
          layer("Down level 0 · 2 ResBlocks", "B×2×16", "B×512×16", "Conv1d k5; FiLM 260→1024; Conv1d k5", "GroupNorm(8)", "Mish"),
          layer("Downsample 0", "B×512×16", "B×512×8", "Conv1d k3, stride 2"),
          layer("Down level 1 · 2 ResBlocks", "B×512×8", "B×1024×8", "Conditional residual blocks", "GroupNorm(8)", "Mish"),
          layer("Downsample 1", "B×1024×8", "B×1024×4", "Conv1d k3, stride 2"),
          layer("Down level 2 · 2 ResBlocks", "B×1024×4", "B×2048×4", "Conditional residual blocks", "GroupNorm(8)", "Mish"),
          layer("Middle · 2 ResBlocks", "B×2048×4", "B×2048×4", "Conditional residual blocks", "GroupNorm(8)", "Mish"),
          layer("Up + skip level 1", "B×4096×4", "B×1024×8", "2 ResBlocks + ConvTranspose1d k4,s2", "GroupNorm(8)", "Mish"),
          layer("Up + skip level 0", "B×2048×8", "B×512×16", "2 ResBlocks + ConvTranspose1d k4,s2", "GroupNorm(8)", "Mish"),
          layer("Final projection", "B×512×16", "B×2×16", "Conv1dBlock 512→512 + Conv1d 1×1 512→2", "GroupNorm(8)", "Mish → linear"),
        ] }),
        node("trajectory", "Denoised trajectory", "action", 4, 0, 4.3, "B×16×2", { w: 4.5, d: 1.5, grid: [16, 3], op: "Scheduler sample after 100 updates", description: "The final denoised sample is a 16-step continuous Push-T action trajectory." }),
        node("prefix", "Executed action prefix", "action", 5, -1.55, 5.8, "B×8×2", { w: 3.2, grid: [8, 2], op: "Slice steps [1:9]", description: "The published workspace executes eight actions, starting after the observation-aligned first index." }),
        node("replan", "Observe and replan", "state", 5, 1.7, 7.0, "next B×2 history", { w: 3.0, op: "Receding-horizon loop", description: "New observations seed a fresh 100-step denoising process before the 16-step horizon is exhausted." }),
      ],
      links: [link("rgb", "crop"), link("crop", "obsResnet"), link("obsResnet", "spatialSoftmax"), link("spatialSoftmax", "obsCondition"), link("robotState", "obsCondition"), link("obsCondition", "unet"), link("noise", "unet"), link("timeEmbed", "unet"), link("unet", "trajectory"), link("trajectory", "prefix"), link("prefix", "replan"), link("replan", "rgb")],
      flowRows: { inference: [["rgb", "crop", "obsResnet", "spatialSoftmax", "obsCondition", "unet"], ["robotState", "obsCondition"], ["noise", "unet", "trajectory", "prefix", "replan"], ["timeEmbed", "unet"]], training: [["rgb", "crop", "obsResnet", "spatialSoftmax", "obsCondition", "unet"], ["robotState", "obsCondition"], ["noised target xₜ", "unet", "noise-prediction loss"], ["timeEmbed", "unet"]] },
      compare: { Scope: "Push-T specialist policy", Inputs: "2× RGB 96² + 2× agent xy", Backbone: "ResNet-18 + SpatialSoftmax64", Generator: "Conditional 1-D U-Net; 100 steps", Output: "16×2 trajectory; execute 8", "Temporal loop": "Receding-horizon regeneration", "World prediction": "No" },
    },

    gr00t: {
      reference: "NVIDIA GR00T N1 · 2B checkpoint",
      configNote: "Eagle width=1536 · 64 image tokens/view · DiT width=1024 · 16 layers · 32 heads · horizon=16",
      sources: [["GR00T N1 checkpoint config", "https://huggingface.co/nvidia/GR00T-N1-2B/blob/main/config.json"], ["GR00T N1 paper", "https://arxiv.org/abs/2503.14734"]],
      symbols: ["B batch", "Nv camera views", "L language/context tokens", "Ds≤64 padded state", "Da≤32 padded action"],
      components: [
        node("vision", "Camera views", "vision", 1, -3.2, -7.0, "B×Nv×3×H×W", { w: 3.0, d: 1.7, grid: [8, 4], op: "Eagle processor; ≤1 tile per view", description: "One or more embodiment-configured camera streams." }),
        node("language", "Language tokens", "language", 1, 0, -7.0, "B×Ltxt", { w: 2.8, op: "Qwen2-chat token template", description: "Natural-language task instruction." }),
        node("proprio", "Embodiment state", "state", 1, 3.25, -7.0, "B×1×Ds; Ds≤64", { w: 3.0, op: "Normalize + pad to max_state_dim 64", description: "Robot-specific state is normalized and padded into the shared maximum dimension." }),
        node("eagleVision", "Eagle visual tokenizer", "vision", 2, -2.8, -5.1, "B×(64Nv)×1536", { w: 3.5, d: 1.7, repeat: 4, repeatLabel: "vision tower + projector", op: "64 tokens per image", description: "The public N1 config fixes 64 image tokens and exposes 1,536-wide backbone features.", layers: [layer("Image tiling", "B×Nv×3×H×W", "B·Nv×tiles×3×Hr×Wr", "Resize/tile; max_input_tiles=1"), layer("Eagle vision tower", "image patches", "B×(64Nv)×Dvis", "vision attention blocks", "LayerNorm", "GELU"), layer("Multimodal projector", "B×(64Nv)×Dvis", "B×(64Nv)×2048", "projector_dim=2048", "—", "GELU"), layer("Backbone representation", "visual + text sequence", "B×Lctx×1536", "select hidden layer 12", "RMS/Layer norm", "—")] }),
        node("eagleText", "Eagle language backbone", "language", 2, 0.2, -3.65, "B×Lctx×1536", { w: 3.6, d: 1.5, repeat: 5, repeatLabel: "selected layer 12 features", norm: "RMSNorm", activation: "SiLU-gated MLP", op: "Causal multimodal Transformer", description: "Image tokens and instruction tokens are jointly contextualized; the N1 action head consumes 1,536-wide features.", layers: [layer("Token embedding", "B×Ltxt", "B×Ltxt×1536", "Embedding table"), layer("Visual/text interleave", "64Nv visual + Ltxt text", "B×Lctx×1536", "Sequence assembly"), layer("Causal self-attention", "B×Lctx×1536", "B×Lctx×1536", "Eagle/Qwen attention", "RMSNorm", "softmax"), layer("Gated MLP", "B×Lctx×1536", "B×Lctx×1536", "up/gate/down projections", "RMSNorm", "SiLU") ] }),
        node("stateTokens", "Embodiment state encoder", "state", 2, 3.0, -4.5, "B×1×1536", { w: 3.15, input: "B×1×64 + embodiment id", op: "CategorySpecificMLP 64→1024→1536", activation: "ReLU", description: "The embodiment id selects robot-specific weights for a two-layer state MLP; its 1,536-D output aligns with the action and VLM token width.", layers: [layer("Category-specific linear 1", "B×1×64", "B×1×1024", "one 64→1024 weight matrix per embodiment", "—", "ReLU"), layer("Category-specific linear 2", "B×1×1024", "B×1×1536", "one 1024→1536 matrix per embodiment", "—", "linear")] }),
        node("actionNoise", "Noisy action chunk", "action", 3, 3.45, -2.0, "B×16×32", { w: 3.0, op: "Beta-distributed flow noise", description: "The shared action space is padded to 32 dimensions across a 16-step horizon." }),
        node("actionProjector", "Embodiment action encoder", "action", 3, 2.1, -0.65, "B×16×1536", { w: 3.7, input: "actions B×16×32 + time B", op: "MultiEmbodimentActionEncoder", activation: "Swish", description: "Robot-specific matrices embed noisy actions and a 1,536-D sinusoidal flow-time code into one 1,536-D token per future step.", layers: [layer("Action projection W1", "B×16×32", "B×16×1536", "category-specific linear 32→1536"), layer("Flow-time sinusoid", "B timestep bucket", "B×16×1536", "SinusoidalPositionalEncoding, repeated over horizon", "—", "sin / cos"), layer("Concatenate action + time", "1536 + 1536", "B×16×3072", "feature concat"), layer("Fusion W2", "B×16×3072", "B×16×1536", "category-specific linear 3072→1536", "—", "Swish"), layer("Output W3", "B×16×1536", "B×16×1536", "category-specific linear 1536→1536") ] }),
        node("actionExpert", "Diffusion Transformer ×16", "compute", 3, 0, 1.0, "B×17×1536 → B×16×1024", { w: 4.8, d: 2.2, repeat: 8, gap: 0.25, repeatLabel: "16 layers · shown compressed", norm: "AdaNorm", activation: "GELU feed-forward", op: "Interleaved self/cross attention", description: "The state token and 16 action tokens form the motor sequence. Sixteen DiT layers use 32 heads × 48 = 1,536 model channels, cross-condition on Eagle context, then return 1,024-D action features.", layers: [layer("Motor-token concat", "state B×1×1536 + action B×16×1536", "B×17×1536", "concatenate on sequence axis"), layer("Adaptive normalization", "B×17×1536", "B×17×1536", "timestep-conditioned scale/shift", "AdaNorm", "—"), layer("Interleaved attention", "Q:17×1536; context:Lctx×1536", "B×17×1536", "32 heads ×48; self/cross attention", "—", "softmax"), layer("Feed-forward", "B×17×1536", "B×17×1536", "1536→6144→1536", "AdaNorm", "GELU"), layer("Action feature output", "last 16 tokens ×1536", "B×16×1024", "final output projection 1536→1024 + dropout 0.2") ] }),
        node("flowLoop", "Flow integration ×16", "compute", 3, -3.5, 1.3, "B×16×32", { w: 2.9, repeat: 4, repeatLabel: "16 inference timesteps", op: "Euler-style flow update", description: "The public N1 config uses 16 inference timesteps; the same DiT parameters are reused at each step." }),
        node("decoder", "Embodiment action decoder", "action", 4, 0, 4.55, "B×16×32", { w: 3.8, op: "CategorySpecificMLP 1024→1024→32", activation: "ReLU", description: "The embodiment id selects the robot-specific velocity decoder at the physical interface.", layers: [layer("Category-specific decoder 1", "B×16×1024", "B×16×1024", "one 1024→1024 matrix per embodiment", "—", "ReLU"), layer("Category-specific decoder 2", "B×16×1024", "B×16×32", "one 1024→32 matrix per embodiment", "—", "linear")] }),
        node("actions", "Embodiment action chunk", "action", 4, 0, 5.75, "B×16×Da; Da≤32", { w: 4.4, grid: [16, 3], op: "Unpad + denormalize by embodiment", description: "Only dimensions defined by the selected robot embodiment are returned in physical action units." }),
        node("closedLoop", "Execute prefix + refresh", "state", 5, 0, 7.0, "B×Te×Da → new obs", { w: 3.8, op: "Receding-horizon control", description: "A deployment may execute fewer than all 16 actions, then re-run semantic grounding and motor generation." }),
      ],
      links: [link("vision", "eagleVision"), link("language", "eagleText"), link("eagleVision", "eagleText"), link("proprio", "stateTokens"), link("eagleText", "actionExpert"), link("stateTokens", "actionExpert"), link("actionNoise", "actionProjector"), link("actionProjector", "actionExpert"), link("actionExpert", "flowLoop"), link("flowLoop", "actionExpert"), link("actionExpert", "decoder"), link("decoder", "actions"), link("actions", "closedLoop")],
      flowRows: { inference: [["vision", "eagleVision", "eagleText", "actionExpert", "decoder", "actions"], ["language", "eagleText"], ["proprio", "stateTokens", "actionExpert"], ["actionNoise", "actionProjector", "actionExpert"], ["actionExpert", "flowLoop", "actionExpert"], ["actions", "closedLoop"]], training: [["vision", "eagleVision", "eagleText", "actionExpert"], ["language", "eagleText"], ["proprio", "stateTokens", "actionExpert"], ["noised action target", "actionProjector", "actionExpert", "flow-matching loss"]] },
      compare: { Scope: "Generalist VLA · N1-2B", Inputs: "images + text + state≤64", Backbone: "Eagle VLM width 1536", Generator: "16-layer DiT; 16 flow steps", Output: "16×Da, Da≤32", "Temporal loop": "Embodiment-specific receding horizon", "World prediction": "No" },
    },

    dreamzero: {
      reference: "DreamZero-DROID · Wan2.1 I2V 14B",
      configNote: "3 views · 176×320 · 33 video frames · 24 actions/block · Wan width=5120 · 32 blocks · 16 heads",
      sources: [["DreamZero implementation", "https://github.com/dreamzero0/dreamzero"], ["DreamZero paper", "https://arxiv.org/abs/2602.15922"]],
      symbols: ["B batch", "F video frames (33)", "Ta action horizon (24)", "Ds state dimension", "Da embodiment action dimension"],
      components: [
        node("scene", "Three DROID camera views", "vision", 1, -3.1, -7.0, "B×3×3×176×320", { w: 3.2, d: 1.8, grid: [8, 4], op: "Exterior-left ×2 + wrist-left", description: "The released DROID recipe aligns three camera views and uses 176×320 training resolution." }),
        node("task", "Language instruction", "language", 1, 0.2, -7.0, "B×Ltxt", { w: 2.7, op: "umT5 tokenizer", description: "A task description conditions the joint world-and-action rollout." }),
        node("robot", "Robot state", "state", 1, 3.2, -7.0, "B×Ds", { w: 2.8, op: "Embodiment normalization", description: "Current DROID proprioception; Ds is defined by the embodiment data schema." }),
        node("textEncoder", "umT5-XXL encoder", "language", 2, 2.55, -5.0, "B×Ltxt×4096", { w: 3.45, repeat: 6, repeatLabel: "24 encoder blocks", norm: "RMSNorm", activation: "Gated GELU", op: "Text Transformer", description: "DreamZero uses the umT5-XXL text encoder, producing 4,096-wide language features.", layers: [layer("Token embedding", "B×Ltxt", "B×Ltxt×4096", "shared embedding"), layer("Self-attention ×24", "B×Ltxt×4096", "B×Ltxt×4096", "64 heads × 64", "RMSNorm", "softmax"), layer("Gated feed-forward ×24", "B×Ltxt×4096", "B×Ltxt×4096", "4096→10240→4096", "RMSNorm", "GELU-gated") ] }),
        node("videoVae", "Wan causal video VAE", "world", 2, -2.85, -4.8, "B×16×9×22×40", { w: 3.6, d: 1.8, repeat: 5, repeatLabel: "temporal×4 · spatial×8", op: "3-D causal VAE encode", norm: "RMS/Group norm", activation: "SiLU", description: "The 33-frame, 176×320 video tensor is compressed to 16 latent channels, 9 latent frames, and a 22×40 spatial grid.", layers: [layer("Causal 3-D stem", "B×3×33×176×320", "B×C×33×176×320", "CausalConv3d", "GroupNorm", "SiLU"), layer("Spatial downsamples ×3", "176×320", "22×40", "stride-2 2-D/3-D residual blocks", "RMS/Group norm", "SiLU"), layer("Temporal downsamples ×2", "33 frames", "9 latent frames", "causal temporal compression ×4", "—", "SiLU"), layer("Latent moments", "B×C×9×22×40", "B×32×9×22×40", "mean/logvar projection"), layer("Sample video latent", "moments", "B×16×9×22×40", "reparameterization") ] }),
        node("actionRegisters", "State + action registers", "action", 2, 2.2, -3.2, "B×Nreg×5120; 25/block", { w: 3.7, input: "state B×Ds; noisy action B×24×Da per block", op: "Embodiment encoders into Wan width", description: "Each video block receives one state token and 24 noisy-action tokens. Full-sequence Nreg = Nblocks×25; cached inference processes one 25-token register at a time." }),
        node("patchEmbed", "Video latent patching", "world", 2, -2.3, -2.45, "B×(9·11·20)×5120", { w: 3.8, input: "B×16×9×22×40", op: "Conv3d patch (1,2,2), 16→5120", description: "A (1,2,2) patch embedding yields 1,980 video tokens for the concrete 33-frame input." }),
        node("wam", "Causal Wan Transformer ×32", "compute", 3, 0, 0.25, "B×(1980+Nreg)×5120", { w: 5.1, d: 2.4, repeat: 8, gap: 0.27, twist: 0.025, repeatLabel: "32 multimodal blocks · compressed", norm: "RMSNorm", activation: "GELU", op: "Video + action-register DiT", description: "DreamZero's documented Wan2.1 configuration uses 32 5,120-wide blocks and 16 attention heads. Video patches and action/state registers share the causal Transformer.", layers: [layer("Sequence assembly", "1980 video tokens + Nreg robot tokens", "B×(1980+Nreg)×5120", "[video tokens | action/state register]"), layer("RMSNorm + self-attention", "B×(1980+Nreg)×5120", "same", "16 heads ×320; blockwise causal mask", "RMSNorm", "softmax"), layer("Language cross-attention", "Q:(1980+Nreg)×5120; KV:Ltxt×4096", "same", "16-head cross-attention", "RMSNorm", "softmax"), layer("Feed-forward up", "B×(1980+Nreg)×5120", "B×(1980+Nreg)×13824", "Linear 5120→13824", "RMSNorm", "GELU"), layer("Feed-forward down", "…×13824", "…×5120", "Linear 13824→5120 + residual"), layer("Blockwise cache / register routing", "video + robot sequence", "same", "3-D video RoPE + 1-D register RoPE") ] }),
        node("flow", "Joint flow denoising", "compute", 3, -3.65, 1.0, "video latent + B×24×Da", { w: 3.0, repeat: 4, repeatLabel: "iterative flow solver", op: "Flow-matching update", description: "The same WAM is reused while noisy video and action variables move toward a clean joint sample." }),
        node("futureVideo", "Predicted future video", "world", 4, -2.0, 4.45, "B×3×33×176×320", { w: 3.9, d: 1.9, repeat: 4, grid: [8, 4], op: "Wan VAE decode", description: "Decoded video makes the model's predicted physical future directly inspectable." }),
        node("actionTrajectory", "Predicted action chunk", "action", 4, 2.15, 4.55, "B×24×Da", { w: 3.6, grid: [12, 3], op: "Register projection + denormalize", description: "Twenty-four continuous robot actions are predicted jointly with the future video; Da remains embodiment-specific." }),
        node("execute", "Execute + observe again", "state", 5, 0, 7.0, "prefix → new 3-view obs", { w: 4.0, op: "Closed-loop WAM policy", description: "The robot executes an action prefix and calls DreamZero again from updated cameras and state." }),
      ],
      links: [link("scene", "videoVae"), link("videoVae", "patchEmbed"), link("task", "textEncoder"), link("robot", "actionRegisters"), link("textEncoder", "wam"), link("patchEmbed", "wam"), link("actionRegisters", "wam"), link("wam", "flow"), link("flow", "wam"), link("wam", "futureVideo"), link("wam", "actionTrajectory"), link("futureVideo", "execute"), link("actionTrajectory", "execute")],
      flowRows: { inference: [["scene", "videoVae", "patchEmbed", "wam", "futureVideo"], ["task", "textEncoder", "wam"], ["robot", "actionRegisters", "wam", "actionTrajectory", "execute"], ["wam", "flow", "wam"]], training: [["3-view target video", "videoVae", "patchEmbed", "wam", "joint flow loss"], ["task", "textEncoder", "wam"], ["state + target actions", "actionRegisters", "wam", "joint flow loss"]] },
      compare: { Scope: "14B World Action Model", Inputs: "3× RGB + text + state", Backbone: "Wan2.1 I2V 14B + umT5-XXL", Generator: "32-layer joint video/action flow model", Output: "33 video frames + 24×Da actions/block", "Temporal loop": "Execute chunk + cached next block", "World prediction": "Yes, jointly with actions" },
    },

    cosmos: {
      tab: "Cosmos P2.5",
      name: "Cosmos Predict2.5-2B",
      sceneName: "Cosmos P2.5",
      category: "World foundation model",
      year: "2025",
      source: "https://github.com/nvidia-cosmos/cosmos-predict2.5",
      reference: "Cosmos Predict2.5 · 2B base",
      configNote: "latent C=16 · patch=(1,2,2) · DiT width=2048 · 28 layers · 16 heads×128 · MLP=8192",
      sources: [["Cosmos Predict2.5 implementation", "https://github.com/nvidia-cosmos/cosmos-predict2.5"], ["Cosmos model collection", "https://huggingface.co/collections/nvidia/cosmos-predict25"]],
      symbols: ["B batch", "F/H/W input video dimensions", "F′/H′/W′ VAE latent dimensions", "L≤512 text tokens", "N latent patch tokens"],
      components: [
        node("text", "Text condition", "language", 1, -3.1, -7.0, "B×L; L≤512", { w: 2.8, op: "Qwen2.5-VL tokenizer", description: "A physical-world prompt is padded to the model's text conditioning length." }),
        node("image", "Image / video context", "vision", 1, 0.5, -7.0, "B×3×F×H×W", { w: 3.5, d: 1.9, grid: [8, 4], op: "Conditioning frame(s)", description: "Predict2.5 accepts text with an image or video depending on the selected world-generation task." }),
        node("video", "Optional action / view condition", "state", 1, 3.55, -6.8, "B×F×Dc", { w: 3.0, op: "Post-trained variant condition", description: "Action-conditioned, robot, and multiview checkpoints add structured conditioning. It is not a universal base-model input." }),
        node("reasoner", "Reason1 text encoder", "language", 2, -2.65, -4.85, "B×512×3584", { w: 3.7, d: 1.6, repeat: 7, repeatLabel: "28 language layers", op: "Qwen2.5-VL 7B text tower", norm: "RMSNorm ε=1e−6", activation: "SiLU", description: "The Predict2.5 conditioning stack uses a 3,584-wide, 28-layer Reason1/Qwen text representation.", layers: [layer("Token embedding", "B×L", "B×512×3584", "pad/embedding to 512 tokens"), layer("Grouped-query attention ×28", "B×512×3584", "B×512×3584", "28 query heads; 4 KV heads", "RMSNorm", "softmax"), layer("Gated MLP ×28", "B×512×3584", "B×512×3584", "3584→18944→3584", "RMSNorm", "SiLU") ] }),
        node("tokenizer", "Cosmos causal video VAE", "world", 2, 1.8, -4.45, "B×16×F′×H′×W′", { w: 3.8, d: 1.9, repeat: 5, repeatLabel: "causal video compression", op: "Encode visual context to 16 channels", norm: "Group/RMS norm", activation: "SiLU", description: "The video tokenizer compresses RGB space to a 16-channel causal latent. F′, H′, and W′ depend on input length/resolution and the checkpoint's VAE compression.", layers: [layer("Causal 3-D input conv", "B×3×F×H×W", "B×C×F×H×W", "causal Conv3d", "Group norm", "SiLU"), layer("Residual down blocks", "RGB-scale features", "compressed F′×H′×W′", "causal spatial/temporal downsampling", "Group/RMS norm", "SiLU"), layer("Latent projection", "B×C×F′×H′×W′", "B×16×F′×H′×W′", "mean/logvar → 16-D sample") ] }),
        node("patch", "3-D latent patches", "world", 2, 1.2, -2.75, "B×N×2048", { w: 3.5, input: "B×16×F′×H′×W′", op: "Conv3d patch (1,2,2), 16→2048", description: "Each temporal 1 × spatial 2×2 latent patch becomes one 2,048-D DiT token." }),
        node("condition", "AdaLN-LoRA condition", "compute", 2, -2.5, -2.8, "B×N×256 controls", { w: 3.2, input: "text 3584 + flow time", op: "Condition projection to rank 256", activation: "SiLU", description: "Text and rectified-flow time generate compact 256-D adaptive normalization controls." }),
        node("worldModel", "Cosmos DiT ×28", "compute", 3, 0, 0.45, "B×N×2048", { w: 5.0, d: 2.45, repeat: 9, gap: 0.25, twist: 0.025, repeatLabel: "28 layers · 2B parameters", norm: "AdaLN-LoRA(256)", activation: "GELU", op: "3-D RoPE self-attention + MLP", description: "Twenty-eight 2,048-wide diffusion-Transformer blocks process the latent world tokens using 16 heads of dimension 128.", layers: [layer("Adaptive normalization", "B×N×2048", "B×N×2048", "time/text scale-shift rank 256", "AdaLN-LoRA", "—"), layer("3-D RoPE attention", "B×N×2048", "B×N×2048", "16 heads ×128; temporal/spatial RoPE", "—", "softmax"), layer("Attention residual", "B×N×2048", "B×N×2048", "output projection + residual"), layer("MLP up", "B×N×2048", "B×N×8192", "Linear 2048→8192", "AdaLN-LoRA", "GELU"), layer("MLP down", "B×N×8192", "B×N×2048", "Linear 8192→2048 + residual") ] }),
        node("solver", "Rectified-flow solver", "compute", 3, -3.5, 1.1, "B×16×F′×H′×W′", { w: 3.0, repeat: 4, repeatLabel: "UniPC integration steps", op: "Iterative velocity update", description: "UniPC integrates the DiT's velocity field from noise toward a clean world latent; parameters are reused at every solver step." }),
        node("latentOut", "Clean world latent", "world", 4, 0, 4.25, "B×16×F′×H′×W′", { w: 4.0, op: "Unpatchify 2048→16·1·2·2", description: "The final projection returns 64 values per patch and reconstructs the 16-channel latent grid.", layers: [layer("Final adaptive norm", "B×N×2048", "B×N×2048", "flow-time conditioning", "AdaLN", "—"), layer("Patch output linear", "B×N×2048", "B×N×64", "Linear 2048→(16×1×2×2)"), layer("Unpatchify", "B×N×64", "B×16×F′×H′×W′", "reshape latent grid") ] }),
        node("future", "Future world video", "world", 4, 0, 5.55, "B×3×Fout×H×W", { w: 4.4, d: 2.0, repeat: 4, grid: [8, 4], op: "Causal video VAE decode", description: "Cosmos produces future-state video rather than robot motor commands." }),
        node("policyData", "Policy training / evaluation", "action", 5, 0, 7.0, "dataset or imagined rollout", { w: 4.0, op: "Downstream consumer", description: "A separate policy can learn from or evaluate against generated worlds; the base 2B WFM is not itself an action policy." }),
      ],
      links: [link("text", "reasoner"), link("image", "tokenizer"), link("tokenizer", "patch"), link("reasoner", "condition"), link("video", "condition"), link("patch", "worldModel"), link("condition", "worldModel"), link("worldModel", "solver"), link("solver", "worldModel"), link("worldModel", "latentOut"), link("latentOut", "future"), link("future", "policyData")],
      flowRows: { inference: [["text", "reasoner", "condition", "worldModel", "latentOut", "future"], ["image", "tokenizer", "patch", "worldModel"], ["optional action/view", "condition"], ["worldModel", "solver", "worldModel"], ["future", "policyData"]], training: [["text", "reasoner", "condition", "worldModel"], ["target video", "tokenizer", "add flow noise", "worldModel", "velocity loss"], ["flow time", "condition"]] },
      compare: { Scope: "2B world foundation model", Inputs: "text + image/video (+variant condition)", Backbone: "Reason1 text + causal video VAE", Generator: "28-layer, width-2048 DiT", Output: "Future-state video", "Temporal loop": "UniPC rectified-flow sampling", "World prediction": "Yes; downstream policy is separate" },
    },
  };
})();
