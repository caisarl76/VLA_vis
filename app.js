const TYPE_COLORS = {
  vision: "#486fb2",
  language: "#8c5eaa",
  state: "#41835f",
  action: "#c65b5b",
  world: "#347f85",
  compute: "#80753e",
};

const TYPE_LABELS = {
  vision: "Vision",
  language: "Language",
  state: "Robot state",
  action: "Action",
  world: "World prediction",
  compute: "Computation",
};

const makeChapter = (name, title, lead, paragraphs, takeaway) => ({ name, title, lead, paragraphs, takeaway });

const models = [
  {
    id: "act",
    tab: "ACT",
    name: "Action Chunking Transformer",
    sceneName: "ACT",
    category: "Visuomotor policy",
    year: "2023",
    accent: "#367fc8",
    source: "https://tonyzhaozh.github.io/aloha/",
    chapters: [
      makeChapter("Overview", "Action Chunking Transformer", "A specialist imitation-learning policy for precise, long-horizon manipulation.", [
        "ACT changes the unit of prediction. Instead of asking the robot for only its <em>next command</em>, it predicts a whole sequence of future joint targets in one pass.",
        "The architecture is a conditional VAE with a Transformer policy. Multi-view images and joint positions describe the current situation; the decoder turns that context into an action chunk.",
      ], "ACT’s defining idea is temporal compression: one prediction spans many control timesteps."),
      makeChapter("Inputs", "See the task from several angles", "The policy starts with synchronized camera views and the robot’s current joints.", [
        "RGB cameras provide the external scene while joint positions provide the robot’s internal configuration. Neither stream contains language—the intended behavior is learned from demonstrations of a specific task.",
        "Multiple viewpoints reduce ambiguity during contact-rich bimanual tasks, where one camera can easily lose sight of a hand or manipulated object.",
      ], "Vision says what surrounds the robot; proprioception says where the robot itself is."),
      makeChapter("Representation", "Turn observations into tokens", "Visual features, robot state, and a training-time style latent meet in a shared representation.", [
        "Each camera frame passes through a visual encoder. The resulting spatial features are combined with embedded joint state.",
        "During training, the conditional VAE also encodes the demonstrated trajectory into a latent variable <em>z</em>. This captures variation between valid demonstrations; at inference, the prior supplies that latent.",
      ], "ACT represents both the current scene and variation in how a demonstrated behavior can unfold."),
      makeChapter("Core model", "Decode an entire horizon in parallel", "A Transformer maps the observation context to many future actions simultaneously.", [
        "The Transformer decoder attends to visual features and robot state while producing queries for every position in the future chunk.",
        "Parallel decoding shortens the effective horizon of the imitation-learning problem: the model does not need to recursively call itself for every tiny movement.",
      ], "The Transformer learns coordination across every action inside the chunk."),
      makeChapter("Action output", "An action ribbon, not one command", "The output is a time-indexed block of continuous joint targets.", [
        "Each column in the action ribbon is one future control step; each row corresponds to part of the bimanual robot state.",
        "Because the complete sequence is generated together, the policy can maintain coherent motion across delicate phases such as approach, contact, and insertion.",
      ], "Chunk prediction gives the robot a locally coherent plan."),
      makeChapter("Control loop", "Overlap, blend, observe again", "New chunks are continuously predicted and temporally ensembled during execution.", [
        "Predictions from nearby observation times overlap. ACT blends them with greater weight on newer estimates, producing smoother control than abruptly replacing one chunk with another.",
        "The robot executes the blended action, receives new camera and joint observations, and repeats the cycle.",
      ], "Temporal ensembling converts discrete chunk predictions into smooth closed-loop behavior."),
    ],
    components: [
      { id: "cameras", label: "Multi-view RGB", type: "vision", chapter: 1, x: -2.2, y: -6.4, z: 0.2, w: 3.2, h: 0.42, d: 2.1, grid: [6, 3], description: "Synchronized scene images from multiple viewpoints." },
      { id: "joints", label: "Joint positions", type: "state", chapter: 1, x: 2.3, y: -6.2, z: -0.2, w: 2.3, h: 0.35, d: 0.8, grid: [10, 1], description: "The current bimanual robot configuration." },
      { id: "visionEncoder", label: "Visual encoder", type: "vision", chapter: 2, x: -2.1, y: -4.3, z: 0, w: 2.7, h: 0.42, d: 1.7, repeat: 3, gap: 0.28, grid: [5, 2], description: "Converts every camera frame into spatial feature maps." },
      { id: "stateEmbed", label: "State embedding", type: "state", chapter: 2, x: 2.1, y: -4.0, z: 0, w: 2.1, h: 0.36, d: 0.75, grid: [8, 1], description: "Projects joint positions into the Transformer token space." },
      { id: "latent", label: "CVAE latent z", type: "compute", chapter: 2, x: 4.2, y: -1.8, z: -0.6, w: 1.5, h: 0.35, d: 0.65, grid: [6, 1], description: "A latent style variable encoding variation in demonstrations." },
      { id: "transformer", label: "Transformer policy", type: "compute", chapter: 3, x: 0, y: -1.7, z: 0, w: 4.6, h: 0.34, d: 2.5, repeat: 8, gap: 0.42, grid: [9, 4], description: "Fuses context and decodes all positions of the future action chunk." },
      { id: "head", label: "Action projection", type: "action", chapter: 4, x: 0, y: 2.7, z: 0, w: 3.6, h: 0.35, d: 1.35, grid: [10, 2], description: "Projects decoder features into continuous robot joint targets." },
      { id: "chunk", label: "Action chunk", type: "action", chapter: 4, x: 0, y: 4.1, z: 0, w: 4.8, h: 0.38, d: 1.7, grid: [14, 3], description: "A fixed-horizon sequence of future bimanual actions." },
      { id: "ensemble", label: "Temporal ensemble", type: "action", chapter: 5, x: 0, y: 6.1, z: 0, w: 3.3, h: 0.38, d: 1.1, repeat: 3, gap: 0.24, grid: [9, 2], description: "Blends overlapping predictions into the command executed now." },
    ],
    links: [["cameras", "visionEncoder"], ["joints", "stateEmbed"], ["visionEncoder", "transformer"], ["stateEmbed", "transformer"], ["latent", "transformer"], ["transformer", "head"], ["head", "chunk"], ["chunk", "ensemble"]],
    compare: { Scope: "Specialist visuomotor policy", Inputs: "Images + joint state", Backbone: "Task-trained visual encoder", Generator: "CVAE Transformer", Output: "Continuous action chunk", "Temporal loop": "Overlapping chunks + temporal ensemble", "World prediction": "No" },
  },
  {
    id: "diffusion-policy",
    tab: "Diffusion Policy",
    name: "Diffusion Policy",
    sceneName: "DP",
    category: "Visuomotor policy",
    year: "2023",
    accent: "#c15378",
    source: "https://diffusion-policy.cs.columbia.edu/",
    chapters: [
      makeChapter("Overview", "Diffusion Policy", "A control policy that generates action trajectories by progressively removing noise.", [
        "Diffusion Policy treats robot control as a conditional generative problem. A candidate action trajectory begins as random noise and is refined until it becomes compatible with the observed scene.",
        "This is useful when a task has several valid solutions: the model can represent multiple distinct trajectory modes instead of averaging them into an invalid compromise.",
      ], "The policy searches action space through iterative refinement rather than direct regression."),
      makeChapter("Inputs", "Condition generation on the present", "Recent images and robot state constrain which trajectories make sense.", [
        "Visual observations encode objects, geometry, and task progress. Low-dimensional robot state supplies pose and proprioceptive context.",
        "An observation history can also reveal motion that a single frame cannot—for example whether an object is slipping or already moving.",
      ], "Observations do not directly become actions; they guide a generative trajectory process."),
      makeChapter("Representation", "Place the scene beside noisy actions", "Observation features become the condition; a random action sequence becomes the object being transformed.", [
        "The observation encoder creates a compact context vector or token sequence. Separately, inference samples a full horizon of Gaussian action noise.",
        "The action tensor already has the dimensions of the desired trajectory, but none of the behavior. Structure appears over several denoising steps.",
      ], "The model starts with the right-shaped answer and progressively gives it meaning."),
      makeChapter("Core model", "Refine a trajectory repeatedly", "A conditional denoising network predicts how to remove noise at each iteration.", [
        "Every denoising pass sees the observation condition, the current noisy trajectory, and a representation of diffusion time.",
        "Early passes establish coarse motion; later passes resolve precise control. The same network is reused across the sequence of refinement steps.",
      ], "Iteration lets the policy revise the entire trajectory before committing to execution."),
      makeChapter("Action output", "Resolve one coherent behavior", "The final sample is a smooth continuous action horizon.", [
        "Different random initializations can resolve into different valid behaviors, preserving multimodality in demonstrations.",
        "The output describes coordinated actions across time rather than isolated commands, which helps maintain smoothness in high-dimensional control spaces.",
      ], "A denoised trajectory is one concrete solution sampled from several possible modes."),
      makeChapter("Control loop", "Execute the prefix, then regenerate", "Receding-horizon control closes the loop between planning and new observations.", [
        "The controller executes only the near-term prefix of the generated trajectory. Before the full horizon expires, it observes the world again.",
        "A fresh denoising process then adapts the remaining plan to disturbances, modeling error, and task progress.",
      ], "Receding-horizon execution keeps an expensive generative plan responsive to the real world."),
    ],
    components: [
      { id: "rgb", label: "RGB history", type: "vision", chapter: 1, x: -2.2, y: -6.4, z: 0.2, w: 3.1, h: 0.42, d: 2.0, grid: [6, 3], description: "Current or recent camera observations." },
      { id: "robotState", label: "Robot state", type: "state", chapter: 1, x: 2.3, y: -6.2, z: -0.2, w: 2.2, h: 0.34, d: 0.75, grid: [9, 1], description: "Pose, joints, or other low-dimensional proprioception." },
      { id: "obsEncoder", label: "Observation encoder", type: "vision", chapter: 2, x: -1.8, y: -4.2, z: 0, w: 3.2, h: 0.38, d: 1.6, repeat: 3, gap: 0.25, grid: [7, 3], description: "Compresses visual and state history into conditioning features." },
      { id: "noise", label: "Gaussian action noise", type: "action", chapter: 2, x: 2.6, y: -3.7, z: -0.4, w: 2.7, h: 0.37, d: 1.2, grid: [13, 3], description: "A random tensor with the shape of the future action horizon." },
      { id: "denoiser", label: "Denoising network", type: "compute", chapter: 3, x: 0, y: -1.9, z: 0, w: 4.5, h: 0.3, d: 2.2, repeat: 10, gap: 0.38, twist: 0.09, grid: [10, 4], description: "Repeatedly predicts the noise or score needed to refine the action trajectory." },
      { id: "trajectory", label: "Denoised trajectory", type: "action", chapter: 4, x: 0, y: 2.9, z: 0, w: 4.8, h: 0.4, d: 1.65, grid: [15, 3], description: "A coherent continuous action sequence sampled from the learned distribution." },
      { id: "prefix", label: "Executed prefix", type: "action", chapter: 5, x: -1.5, y: 4.7, z: 0.15, w: 2.0, h: 0.38, d: 1.0, grid: [5, 2], description: "Only the first portion of the trajectory is sent to the robot." },
      { id: "replan", label: "Observe + replan", type: "state", chapter: 5, x: 1.5, y: 6.1, z: 0, w: 2.2, h: 0.38, d: 1.0, repeat: 2, gap: 0.25, grid: [6, 2], description: "New observations trigger another full trajectory generation cycle." },
    ],
    links: [["rgb", "obsEncoder"], ["robotState", "obsEncoder"], ["obsEncoder", "denoiser"], ["noise", "denoiser"], ["denoiser", "trajectory"], ["trajectory", "prefix"], ["prefix", "replan"], ["replan", "obsEncoder"]],
    compare: { Scope: "Specialist visuomotor policy", Inputs: "Images + robot state", Backbone: "Task-trained observation encoder", Generator: "Iterative conditional diffusion", Output: "Continuous action trajectory", "Temporal loop": "Execute prefix + regenerate", "World prediction": "No" },
  },
  {
    id: "gr00t",
    tab: "GR00T N1",
    name: "GR00T N1",
    sceneName: "GR00T N1",
    category: "Generalist VLA",
    year: "2025",
    accent: "#5f9844",
    source: "https://research.nvidia.com/labs/lpr/publication/gr00tn1_2025/",
    chapters: [
      makeChapter("Overview", "GR00T N1", "A humanoid foundation model coupling semantic reasoning to continuous action generation.", [
        "GR00T N1 uses a dual-system design. A pretrained vision-language model handles semantic understanding, while a diffusion Transformer specializes in fast continuous motor control.",
        "The split lets web-scale visual and linguistic knowledge inform a policy without forcing robot actions into the same representation as language tokens.",
      ], "A semantic ‘brain’ conditions a dedicated continuous action expert."),
      makeChapter("Inputs", "Combine instruction, sight, and embodiment", "The model receives language, visual observations, and robot-specific proprioception.", [
        "Language states the goal, cameras ground that goal in the current environment, and proprioception identifies what the particular embodiment can do from its present pose.",
        "This combination is what makes GR00T a VLA rather than a vision-only behavior-cloning policy.",
      ], "Task meaning and physical state enter through distinct but coordinated channels."),
      makeChapter("Representation", "Build semantic and embodied context", "An Eagle VLM represents the scene while a state encoder represents the robot.", [
        "The VLM contributes object recognition, instruction grounding, and broad semantic knowledge. Robot state is projected into embodiment-aware tokens.",
        "Keeping these pathways distinct preserves the strengths of the pretrained VLM while exposing precise numerical state to the controller.",
      ], "GR00T keeps world semantics and embodiment state legible until they meet at the action expert."),
      makeChapter("Core model", "Condition a diffusion action expert", "A dedicated diffusion Transformer converts context into temporally coherent motion.", [
        "The action expert receives semantic features from the VLM together with encoded robot state. It iteratively refines a continuous action chunk.",
        "This resembles a fast motor system guided by a slower semantic system: understanding and control cooperate but do not share one decoder.",
      ], "The dual-system boundary is the architectural signature to look for."),
      makeChapter("Action output", "Decode for the target embodiment", "The generated chunk is mapped into the action space of a particular robot.", [
        "Humanoid platforms may differ in joints, degrees of freedom, sensors, and command conventions. Embodiment-specific encoders and decoders translate between that interface and shared model features.",
        "The output remains continuous, supporting smooth whole-body or manipulation control.",
      ], "Shared intelligence reaches the robot through an embodiment-specific control interface."),
      makeChapter("Control loop", "Refresh semantics and motion together", "New observations repeatedly update both high-level context and low-level action generation.", [
        "The VLM re-grounds the instruction as the scene changes. The action expert generates a new chunk conditioned on that updated context and current proprioception.",
        "This closed loop allows semantic goals to remain stable while motor details adapt continuously.",
      ], "GR00T closes the loop at two timescales: task understanding and physical control."),
    ],
    components: [
      { id: "vision", label: "Camera observations", type: "vision", chapter: 1, x: -3.2, y: -6.5, z: 0.3, w: 2.6, h: 0.42, d: 1.8, grid: [6, 3], description: "One or more visual views of the environment." },
      { id: "language", label: "Language instruction", type: "language", chapter: 1, x: 0, y: -6.2, z: -0.3, w: 2.7, h: 0.35, d: 0.75, grid: [9, 1], description: "A natural-language description of the robot’s task." },
      { id: "proprio", label: "Embodiment state", type: "state", chapter: 1, x: 3.2, y: -6.4, z: 0.1, w: 2.5, h: 0.35, d: 0.8, grid: [10, 1], description: "Robot-specific proprioception and configuration." },
      { id: "vlm", label: "Eagle VLM · System 2", type: "language", chapter: 2, x: -1.8, y: -4.1, z: 0, w: 3.4, h: 0.3, d: 2.0, repeat: 7, gap: 0.34, grid: [8, 4], description: "A pretrained vision-language backbone providing semantic context." },
      { id: "stateTokens", label: "State encoder", type: "state", chapter: 2, x: 2.5, y: -3.4, z: -0.2, w: 2.1, h: 0.36, d: 0.85, repeat: 2, gap: 0.25, grid: [8, 2], description: "Projects embodiment state into features for the action expert." },
      { id: "fusion", label: "Cross-system context", type: "compute", chapter: 3, x: 0, y: -0.7, z: 0, w: 4.3, h: 0.36, d: 1.7, grid: [10, 3], description: "Semantic VLM features and precise robot-state features condition motor generation." },
      { id: "actionExpert", label: "Diffusion Transformer · System 1", type: "compute", chapter: 3, x: 0, y: 0.6, z: 0, w: 4.5, h: 0.3, d: 2.1, repeat: 8, gap: 0.37, grid: [10, 4], description: "A continuous-action expert that iteratively refines robot motion." },
      { id: "decoder", label: "Embodiment decoder", type: "action", chapter: 4, x: 0, y: 4.1, z: 0, w: 3.3, h: 0.36, d: 1.1, grid: [10, 2], description: "Maps shared action features into the target robot’s command space." },
      { id: "actions", label: "Continuous action chunk", type: "action", chapter: 4, x: 0, y: 5.25, z: 0, w: 4.7, h: 0.4, d: 1.5, grid: [14, 3], description: "Temporally coherent commands for the selected embodiment." },
      { id: "closedLoop", label: "Closed-loop execution", type: "state", chapter: 5, x: 0, y: 6.7, z: 0, w: 2.8, h: 0.38, d: 1.0, grid: [8, 2], description: "Execution changes the scene; new observations refresh both systems." },
    ],
    links: [["vision", "vlm"], ["language", "vlm"], ["proprio", "stateTokens"], ["vlm", "fusion"], ["stateTokens", "fusion"], ["fusion", "actionExpert"], ["actionExpert", "decoder"], ["decoder", "actions"], ["actions", "closedLoop"]],
    compare: { Scope: "Generalist humanoid VLA", Inputs: "Images + language + proprioception", Backbone: "Pretrained Eagle VLM", Generator: "Diffusion Transformer action expert", Output: "Embodiment-specific continuous chunk", "Temporal loop": "Chunked closed-loop control", "World prediction": "No" },
  },
  {
    id: "dreamzero",
    tab: "DreamZero",
    name: "DreamZero",
    sceneName: "DreamZero",
    category: "World Action Model",
    year: "2026",
    accent: "#7564ce",
    source: "https://dreamzero0.github.io/",
    chapters: [
      makeChapter("Overview", "DreamZero", "A World Action Model that predicts what happens and how the robot makes it happen.", [
        "DreamZero builds a robot policy on a pretrained video diffusion backbone. Instead of predicting only actions, it generates future visual states and robot actions together.",
        "This joint target lets motion priors learned from video participate directly in control, aiming to improve generalization to unfamiliar physical motions and environments.",
      ], "DreamZero makes world prediction and action prediction one coupled generative problem."),
      makeChapter("Inputs", "Start from a scene, task, and body", "Visual context, task conditioning, and robot state anchor the generated future.", [
        "The current observation says where the world begins. Task conditioning specifies the intended outcome, and robot state constrains which motion can realize it.",
        "These inputs seed both branches of the prediction: imagined future frames and the aligned action trajectory.",
      ], "One shared context must support both seeing the future and acting toward it."),
      makeChapter("Representation", "Align video time with action time", "World latents and action latents describe the same future at different levels.", [
        "A pretrained video representation compresses frames while preserving temporal dynamics. Robot actions are embedded along the same time axis.",
        "This alignment is crucial: every motor segment should correspond to a visible change in the imagined rollout.",
      ], "The model learns a synchronized latent language for motion and control."),
      makeChapter("Core model", "Denoise world and action together", "A World Action Model refines paired visual and motor futures.", [
        "The video diffusion backbone supplies broad motion priors. Robot learning adapts that generator so actions and future frames remain mutually consistent.",
        "Unlike a VLM with a separate policy head, the imagined world is part of the same temporal generative process as the control signal.",
      ], "Joint generation is the structural difference between a WAM and a conventional VLA head."),
      makeChapter("Joint output", "Expose both intention and control", "The model emits a future video rollout beside a robot action trajectory.", [
        "The video branch shows the outcome the model expects, making the policy’s physical intention partially inspectable.",
        "The action branch supplies the continuous commands intended to realize that future. Their shared timing reveals how imagined change maps to motor behavior.",
      ], "DreamZero pairs ‘what should happen’ with ‘how to cause it.’"),
      makeChapter("Control loop", "Act toward the generated future", "Execution tests the imagined rollout against the real world.", [
        "The robot executes the near-term action segment, observes the resulting scene, and conditions a new paired future on reality.",
        "Mismatch between imagined and observed motion can be corrected in the next generation cycle.",
      ], "Closed-loop regeneration continually reconciles the dream with the physical scene."),
    ],
    components: [
      { id: "scene", label: "Current scene", type: "vision", chapter: 1, x: -3.0, y: -6.4, z: 0.2, w: 2.8, h: 0.42, d: 1.8, grid: [6, 3], description: "The current visual state of the physical environment." },
      { id: "task", label: "Task condition", type: "language", chapter: 1, x: 0, y: -6.15, z: -0.25, w: 2.3, h: 0.34, d: 0.7, grid: [8, 1], description: "Semantic conditioning describing the intended task." },
      { id: "robot", label: "Robot state", type: "state", chapter: 1, x: 3.0, y: -6.35, z: 0.1, w: 2.3, h: 0.34, d: 0.78, grid: [9, 1], description: "The robot’s present physical configuration." },
      { id: "videoLatent", label: "Video latent", type: "world", chapter: 2, x: -2.1, y: -4.25, z: 0, w: 3.3, h: 0.38, d: 1.8, repeat: 3, gap: 0.26, grid: [7, 3], description: "A compressed representation carrying temporal visual dynamics." },
      { id: "actionLatent", label: "Action latent", type: "action", chapter: 2, x: 2.2, y: -4.0, z: -0.2, w: 2.7, h: 0.36, d: 1.1, grid: [12, 2], description: "A time-aligned representation of future robot control." },
      { id: "wam", label: "World Action Model", type: "compute", chapter: 3, x: 0, y: -1.9, z: 0, w: 4.8, h: 0.3, d: 2.45, repeat: 11, gap: 0.38, twist: -0.055, grid: [10, 5], description: "A video-diffusion-based generator jointly modeling future frames and actions." },
      { id: "futureVideo", label: "Predicted future video", type: "world", chapter: 4, x: -2.1, y: 3.3, z: 0.2, w: 3.4, h: 0.4, d: 1.9, repeat: 4, gap: 0.24, grid: [6, 3], description: "A visual rollout of the physical outcome the model expects." },
      { id: "actionTrajectory", label: "Action trajectory", type: "action", chapter: 4, x: 2.4, y: 3.65, z: -0.2, w: 2.8, h: 0.4, d: 1.15, grid: [14, 3], description: "Continuous robot actions aligned with the predicted frames." },
      { id: "execute", label: "Execute + observe", type: "state", chapter: 5, x: 0, y: 6.05, z: 0, w: 3.1, h: 0.38, d: 1.1, repeat: 2, gap: 0.25, grid: [8, 2], description: "The real result conditions the next jointly generated future." },
    ],
    links: [["scene", "videoLatent"], ["task", "videoLatent"], ["task", "actionLatent"], ["robot", "actionLatent"], ["videoLatent", "wam"], ["actionLatent", "wam"], ["wam", "futureVideo"], ["wam", "actionTrajectory"], ["futureVideo", "execute"], ["actionTrajectory", "execute"]],
    compare: { Scope: "World Action Model", Inputs: "Scene + task + robot state", Backbone: "Pretrained video diffusion model", Generator: "Joint video-action diffusion", Output: "Future video + continuous actions", "Temporal loop": "Regenerate paired future", "World prediction": "Yes" },
  },
  {
    id: "cosmos",
    tab: "Cosmos",
    name: "NVIDIA Cosmos",
    sceneName: "Cosmos",
    category: "World model platform",
    year: "2025–26",
    accent: "#2a8b8e",
    source: "https://docs.nvidia.com/cosmos/index.html",
    chapters: [
      makeChapter("Overview", "NVIDIA Cosmos", "A family of world foundation models and tools for building physical-AI systems.", [
        "Cosmos is broader than a single robot policy. Its models reason about, generate, and transform physical-world video for robotics, autonomous driving, simulation, and synthetic data.",
        "The family is versioned, so the exact architecture and capability depend on the selected Cosmos release. In this atlas it occupies the world-model layer surrounding downstream policies.",
      ], "Cosmos models possible worlds; a downstream controller may learn or plan with those worlds."),
      makeChapter("Inputs", "Condition a possible world", "Text, images, video, and physical-AI data can define the scenario.", [
        "Different family members support text-to-world, image-to-world, video-to-world, transformation, or reasoning workflows.",
        "The inputs describe environmental context rather than only the numerical state of one robot embodiment.",
      ], "Cosmos starts from a world-level prompt, observation, or simulation state."),
      makeChapter("Representation", "Compress pixels and add physical context", "World tokenizers reduce video cost while reasoning features enrich the condition.", [
        "Video tokenization maps large pixel sequences into compact spatiotemporal latents. This allows generative models to work across longer, higher-resolution physical scenes.",
        "Reasoning-oriented components can interpret objects, motion, and constraints before generation.",
      ], "Efficient world latents are the substrate; physical reasoning shapes what they should become."),
      makeChapter("Core model", "Generate a temporally coherent world", "A version-specific world foundation model predicts or transforms physical scenes.", [
        "Generative Transformer and diffusion-style components model change over time. Conditioning controls the environment, camera, objects, or intended scenario.",
        "Unlike a policy-only generator, the primary product is a world rollout that can be inspected, varied, or fed into another system.",
      ], "The core generator operates in world space, not just robot action space."),
      makeChapter("World output", "Branch into many possible futures", "Generated videos and scenario variations expand the physical-AI data surface.", [
        "One starting scene can yield multiple controlled futures, including uncommon or safety-critical situations that are difficult to collect in reality.",
        "Transformation models can also modify existing simulation or recorded data while retaining useful physical structure.",
      ], "World generation turns one scenario into a controllable distribution of physical possibilities."),
      makeChapter("Downstream use", "Train, evaluate, or plan with generated worlds", "Policies consume Cosmos outputs; Cosmos is not always the policy itself.", [
        "Synthetic rollouts can augment robot-policy training, probe failure cases, or provide imagined futures for planning and evaluation.",
        "Keeping this boundary visible prevents an apples-to-oranges comparison with ACT, Diffusion Policy, or GR00T’s action head.",
      ], "Cosmos contributes environments and foresight to the control stack, while downstream policies produce motor commands."),
    ],
    components: [
      { id: "text", label: "Text condition", type: "language", chapter: 1, x: -3.4, y: -6.35, z: 0.1, w: 2.2, h: 0.34, d: 0.7, grid: [8, 1], description: "A prompt describing the desired physical scenario." },
      { id: "image", label: "Image context", type: "vision", chapter: 1, x: 0, y: -6.55, z: 0.25, w: 2.7, h: 0.42, d: 1.75, grid: [6, 3], description: "A starting visual state for image-to-world generation." },
      { id: "video", label: "Video / simulation", type: "world", chapter: 1, x: 3.4, y: -6.35, z: -0.1, w: 2.8, h: 0.42, d: 1.7, repeat: 2, gap: 0.22, grid: [6, 3], description: "Recorded or simulated physical-AI context." },
      { id: "tokenizer", label: "World tokenizer", type: "world", chapter: 2, x: -1.9, y: -4.2, z: 0, w: 3.4, h: 0.36, d: 1.8, repeat: 4, gap: 0.27, grid: [8, 4], description: "Compresses video into efficient spatiotemporal latents." },
      { id: "reasoner", label: "Physical reasoning context", type: "language", chapter: 2, x: 2.5, y: -3.8, z: -0.25, w: 2.8, h: 0.34, d: 1.1, repeat: 3, gap: 0.25, grid: [8, 2], description: "Semantic and physical context used by reasoning-capable family members." },
      { id: "worldModel", label: "World foundation model", type: "compute", chapter: 3, x: 0, y: -1.7, z: 0, w: 5.0, h: 0.3, d: 2.6, repeat: 12, gap: 0.38, twist: 0.045, grid: [11, 5], description: "A version-specific generator that predicts or transforms physical-world video." },
      { id: "future", label: "Future world rollout", type: "world", chapter: 4, x: -2.0, y: 3.6, z: 0.2, w: 3.5, h: 0.4, d: 2.0, repeat: 5, gap: 0.24, grid: [7, 3], description: "A temporally coherent generated physical scenario." },
      { id: "variations", label: "Scenario variations", type: "world", chapter: 4, x: 2.5, y: 4.0, z: -0.2, w: 2.8, h: 0.38, d: 1.5, repeat: 3, gap: 0.24, grid: [6, 3], description: "Controlled alternatives used to broaden coverage and probe rare cases." },
      { id: "policyData", label: "Policy data + evaluation", type: "action", chapter: 5, x: 0, y: 6.5, z: 0, w: 4.0, h: 0.4, d: 1.25, grid: [12, 2], description: "Generated worlds support downstream policy training, planning, and evaluation." },
    ],
    links: [["text", "reasoner"], ["image", "tokenizer"], ["video", "tokenizer"], ["tokenizer", "worldModel"], ["reasoner", "worldModel"], ["worldModel", "future"], ["worldModel", "variations"], ["future", "policyData"], ["variations", "policyData"]],
    compare: { Scope: "World foundation model platform", Inputs: "Text + image + video + simulation", Backbone: "World tokenizer + reasoning context", Generator: "Version-specific world generator", Output: "Video worlds + scenario variations", "Temporal loop": "Downstream training / planning", "World prediction": "Yes" },
  },
];

// Replace the introductory sketches above with checkpoint-pinned architecture
// specifications. Keeping the prose model cards here makes the lesson copy easy
// to edit; architecture-specs.js owns all tensor and layer-level facts.
models.forEach((model) => {
  const spec = window.ARCHITECTURE_SPECS?.[model.id];
  if (spec) Object.assign(model, spec);
});

const query = new URLSearchParams(window.location.search);
const validModel = (id) => models.some((model) => model.id === id);
let selectedModelId = validModel(query.get("model")) ? query.get("model") : "act";
let chapterIndex = Math.max(0, Math.min(5, Number(query.get("chapter")) || 0));
let pathwayMode = query.get("pathway") === "training" ? "training" : "inference";
let isPlaying = false;
let playTimer = null;
let selectedComponentId = null;
let sceneBlocks = [];
let hitTargets = [];
let dragState = null;
let didDrag = false;

const camera = {
  yaw: -0.58,
  pitch: -0.12,
  zoom: 1,
  targetY: 0,
  desiredYaw: -0.58,
  desiredPitch: -0.12,
  desiredZoom: 1,
  desiredTargetY: 0,
};

const el = {
  previous: document.querySelector("#previousChapter"),
  next: document.querySelector("#nextChapter"),
  chapterNumber: document.querySelector("#chapterNumber"),
  chapterName: document.querySelector("#chapterName"),
  modelClass: document.querySelector("#modelClass"),
  chapterTitle: document.querySelector("#chapterTitle"),
  chapterLead: document.querySelector("#chapterLead"),
  miniDiagram: document.querySelector("#miniDiagram"),
  toc: document.querySelector("#tableOfContents"),
  chapterContent: document.querySelector("#chapterContent"),
  takeaway: document.querySelector("#chapterTakeaway"),
  lessonScroll: document.querySelector("#lessonScroll"),
  continueButton: document.querySelector("#continueButton"),
  skipButton: document.querySelector("#skipButton"),
  playButton: document.querySelector("#playButton"),
  timeline: document.querySelector("#timeline"),
  playbackHint: document.querySelector("#playbackHint"),
  stepReadout: document.querySelector("#stepReadout"),
  modelTabs: document.querySelector("#modelTabs"),
  visualPanel: document.querySelector("#visualPanel"),
  canvas: document.querySelector("#sceneCanvas"),
  sceneCategory: document.querySelector("#sceneCategory"),
  sceneModelName: document.querySelector("#sceneModelName"),
  sceneYear: document.querySelector("#sceneYear"),
  sceneReference: document.querySelector("#sceneReference"),
  pathwayToggle: document.querySelector("#pathwayToggle"),
  visualLegend: document.querySelector("#visualLegend"),
  inspector: document.querySelector("#blockInspector"),
  inspectorType: document.querySelector("#inspectorType"),
  inspectorTitle: document.querySelector("#inspectorTitle"),
  inspectorBody: document.querySelector("#inspectorBody"),
  inspectorSpecs: document.querySelector("#inspectorSpecs"),
  compareDialog: document.querySelector("#compareDialog"),
  compareA: document.querySelector("#compareA"),
  compareB: document.querySelector("#compareB"),
  compareVisual: document.querySelector("#compareVisual"),
  compareHead: document.querySelector("#compareHead"),
  compareBody: document.querySelector("#compareBody"),
  aboutDialog: document.querySelector("#aboutDialog"),
  aboutSourceLink: document.querySelector("#aboutSourceLink"),
};

const ctx = el.canvas.getContext("2d");

function getModel(id = selectedModelId) {
  return models.find((model) => model.id === id);
}

function setAccent(color) {
  document.documentElement.style.setProperty("--accent", color);
  document.documentElement.style.setProperty("--accent-soft", `${color}28`);
}

function updateUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set("model", selectedModelId);
  url.searchParams.set("chapter", chapterIndex);
  url.searchParams.set("pathway", pathwayMode);
  window.history.replaceState({}, "", url);
}

function renderModelTabs() {
  el.modelTabs.innerHTML = models.map((model) => `
    <button class="model-tab ${model.id === selectedModelId ? "active" : ""}" data-model="${model.id}" style="--model-color:${model.accent}">${model.tab}</button>
  `).join("");
  el.modelTabs.querySelectorAll(".model-tab").forEach((button) => {
    button.addEventListener("click", () => selectModel(button.dataset.model));
  });
}

function renderMiniDiagram(model) {
  el.miniDiagram.innerHTML = `
    <span class="reference-kicker">PINNED REFERENCE</span>
    <strong>${model.reference}</strong>
    <p>${model.configNote}</p>
    <div class="reference-links">${(model.sources || []).map(([label, href]) => `<a href="${href}" target="_blank" rel="noreferrer">${label} ↗</a>`).join("")}</div>
  `;
}

function isModeVisible(item) {
  return !item?.mode || item.mode === "both" || item.mode === pathwayMode;
}

function getFlowItem(model, id) {
  const component = model.components.find((item) => item.id === id);
  return component || { id: `literal-${id}`, label: id, shape: "", type: "compute", chapter: -1 };
}

function diagramPort(rect, side = "bottom") {
  if (side === "top") return { x: rect.x + rect.w / 2, y: rect.y };
  if (side === "left") return { x: rect.x, y: rect.y + rect.h / 2 };
  if (side === "right") return { x: rect.x + rect.w, y: rect.y + rect.h / 2 };
  return { x: rect.x + rect.w / 2, y: rect.y + rect.h };
}

function diagramPath(edge, fromRect, toRect) {
  const start = diagramPort(fromRect, edge.fromPort || "bottom");
  const end = diagramPort(toRect, edge.toPort || "top");
  if (edge.route === "left" || edge.route === "right") {
    const rail = edge.route === "left" ? Math.min(start.x, end.x) - 24 : Math.max(start.x, end.x) + 24;
    return `M ${start.x} ${start.y} L ${rail} ${start.y} L ${rail} ${end.y} L ${end.x} ${end.y}`;
  }
  if (edge.route === "above") {
    const rail = Math.min(start.y, end.y) - 20;
    return `M ${start.x} ${start.y} L ${start.x} ${rail} L ${end.x} ${rail} L ${end.x} ${end.y}`;
  }
  const middleY = start.y + (end.y - start.y) * 0.5;
  return `M ${start.x} ${start.y} L ${start.x} ${middleY} L ${end.x} ${middleY} L ${end.x} ${end.y}`;
}

function diagramEdgeLabel(edge, fromRect, toRect) {
  if (!edge.label) return "";
  const start = diagramPort(fromRect, edge.fromPort || "bottom");
  const end = diagramPort(toRect, edge.toPort || "top");
  return `<text class="diagram-edge-label diagram-edge-label-${edge.kind}" x="${(start.x + end.x) / 2}" y="${(start.y + end.y) / 2 - 5}">${edge.label}</text>`;
}

function renderArchitectureDiagram(model) {
  const diagram = window.ARCHITECTURE_DIAGRAMS?.[model.id];
  if (!diagram) return "";
  const nodes = diagram.nodes.filter(isModeVisible);
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const groups = diagram.groups.filter(isModeVisible);
  const edges = diagram.edges.filter((edge) => isModeVisible(edge) && nodeMap.has(edge.from) && nodeMap.has(edge.to));
  const focused = (item) => chapterIndex === 0 || item.chapter === chapterIndex;
  return `
    <section class="architecture-schematic" aria-label="Hierarchical ${pathwayMode} architecture diagram">
      <header><span>${pathwayMode.toUpperCase()} ARCHITECTURE</span><em>Containers are modules. Arrows terminate at the layer that consumes the tensor.</em></header>
      <div class="schematic-scroll">
        <svg viewBox="0 0 ${diagram.width} ${diagram.height}" role="img" aria-label="${model.reference} nested forward graph">
          <defs>
            ${["data", "residual", "cross", "condition", "skip", "loop"].map((kind) => `<marker class="marker-${kind}" id="arrow-${model.id}-${kind}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" /></marker>`).join("")}
          </defs>
          <g class="diagram-groups">${groups.map((group) => `<g class="diagram-group ${group.nested ? "nested" : ""} ${focused(group) ? "focus" : "muted"}"><rect x="${group.x}" y="${group.y}" width="${group.w}" height="${group.h}" rx="7" /><text x="${group.x + 14}" y="${group.y + 21}">${group.label}</text><text class="diagram-group-note" x="${group.x + 14}" y="${group.y + 37}">${group.note}</text></g>`).join("")}</g>
          <g class="diagram-edges">${edges.map((edge) => {
            const from = nodeMap.get(edge.from); const to = nodeMap.get(edge.to);
            const active = focused(from) || focused(to);
            return `<g class="diagram-edge ${edge.kind} ${active ? "focus" : "muted"}"><path d="${diagramPath(edge, from, to)}" marker-end="url(#arrow-${model.id}-${edge.kind})" />${diagramEdgeLabel(edge, from, to)}</g>`;
          }).join("")}</g>
          <g class="diagram-nodes">${nodes.map((node) => `<g class="diagram-node ${focused(node) ? "focus" : "muted"}" tabindex="0" role="button" data-diagram-component="${node.componentId || ""}" data-diagram-layer="${Number.isInteger(node.layerIndex) ? node.layerIndex : ""}" style="--node-color:${TYPE_COLORS[node.type]}"><rect x="${node.x}" y="${node.y}" width="${node.w}" height="${node.h}" rx="5" /><foreignObject x="${node.x + 9}" y="${node.y + 7}" width="${node.w - 18}" height="${node.h - 12}"><div xmlns="http://www.w3.org/1999/xhtml" class="diagram-node-copy"><b>${node.label}</b><small>${node.shape}</small></div></foreignObject></g>`).join("")}</g>
        </svg>
      </div>
      <footer class="diagram-legend">${(diagram.legend || []).map(([kind, label]) => `<span class="diagram-key ${kind}"><i></i>${label}</span>`).join("")}</footer>
    </section>`;
}

function renderLayerInventory(model) {
  if (chapterIndex === 0) return `<p class="architecture-hint">Choose a chapter to explode its modules into individual layers. Every canvas label shows the output tensor shape; click a slab for its complete operator specification.</p>`;
  const components = model.components.filter((component) => component.chapter === chapterIndex && isModeVisible(component));
  if (!components.length) return "";
  return `<section class="layer-inventory"><header><span>LAYER INVENTORY</span><em>Dimensions belong to the pinned reference above.</em></header>${components.map((component, componentIndex) => `
    <details ${componentIndex === 0 ? "open" : ""}>
      <summary><b>${component.label}</b><code>${component.output || component.shape}</code><small>${component.layers?.length ? `${component.layers.length} internal operations shown on canvas` : component.op}</small></summary>
      <div class="layer-table" role="table" aria-label="${component.label} layers">
        <div class="layer-row layer-head" role="row"><span>Layer / operation</span><span>Input</span><span>Output</span><span>Norm · activation</span></div>
        ${(component.layers?.length ? component.layers : [component]).map((detail) => `<div class="layer-row" role="row"><span><b>${detail.label || component.label}</b><small>${detail.op || component.op}</small></span><code>${detail.input || component.input || "—"}</code><code>${detail.output || detail.shape || component.output || component.shape}</code><span>${detail.norm || component.norm || "—"}<small>${detail.activation || component.activation || "—"}</small></span></div>`).join("")}
      </div>
    </details>`).join("")}</section>`;
}

function bindDiagramNodes() {
  el.chapterContent.querySelectorAll("[data-diagram-component]").forEach((button) => {
    const component = getModel().components.find((item) => item.id === button.dataset.diagramComponent);
    if (!component) return;
    const open = () => {
      const rawLayerIndex = button.dataset.diagramLayer;
      const layerIndex = rawLayerIndex === "" ? null : Number(rawLayerIndex);
      const detail = Number.isInteger(layerIndex) && component.layers?.[layerIndex];
      if (!detail) return openInspector(component);
      openInspector({ ...component, ...detail, nodeId: `${component.id}::diagram-${layerIndex}`, parentId: component.id, isDetail: true, instance: layerIndex, shape: detail.output, description: detail.description || `${detail.op}. Input ${detail.input}; output ${detail.output}.` });
    };
    button.addEventListener("click", open);
    button.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") open(); });
  });
}

function renderToc(model) {
  el.toc.innerHTML = `<div class="toc-list">${model.chapters.map((chapter, index) => `
    <button class="toc-button ${index === chapterIndex ? "active" : ""}" data-chapter="${index}">
      <span>${String(index + 1).padStart(2, "0")}</span>${chapter.name}
    </button>`).join("")}</div>`;
  el.toc.querySelectorAll(".toc-button").forEach((button) => {
    button.addEventListener("click", () => setChapter(Number(button.dataset.chapter)));
  });
}

function renderTimeline(model) {
  el.timeline.innerHTML = model.chapters.map((_, index) => `
    <button class="timeline-step ${index < chapterIndex ? "complete" : ""} ${index === chapterIndex ? "active" : ""}" data-chapter="${index}" aria-label="Open chapter ${index + 1}"></button>
  `).join("");
  el.timeline.querySelectorAll(".timeline-step").forEach((button) => {
    button.addEventListener("click", () => setChapter(Number(button.dataset.chapter)));
  });
}

function renderChapter() {
  const model = getModel();
  const chapter = model.chapters[chapterIndex];
  el.chapterNumber.textContent = String(chapterIndex + 1).padStart(2, "0");
  el.chapterName.textContent = chapter.name;
  el.modelClass.textContent = model.category.toUpperCase();
  el.chapterTitle.textContent = chapter.title;
  el.chapterLead.textContent = chapter.lead;
  el.chapterContent.innerHTML = `
    <h2>${chapterIndex === 0 ? "How to read this model" : chapter.name}</h2>
    ${chapter.paragraphs.map((paragraph) => `<p>${paragraph}</p>`).join("")}
    ${renderArchitectureDiagram(model)}
    ${renderLayerInventory(model)}`;
  bindDiagramNodes();
  el.takeaway.textContent = chapter.takeaway;
  el.previous.disabled = chapterIndex === 0;
  el.next.disabled = chapterIndex === model.chapters.length - 1;
  el.continueButton.textContent = chapterIndex === model.chapters.length - 1 ? "Restart walkthrough" : "Continue";
  el.stepReadout.textContent = `${chapterIndex + 1} / ${model.chapters.length}`;
  renderToc(model);
  renderTimeline(model);
  updateUrl();
}

function setChapter(nextIndex, options = {}) {
  const model = getModel();
  chapterIndex = (nextIndex + model.chapters.length) % model.chapters.length;
  selectedComponentId = null;
  el.inspector.classList.remove("open");
  sceneBlocks = expandScene(model);
  renderChapter();
  if (!options.keepScroll) el.lessonScroll.scrollTo({ top: 0, behavior: "smooth" });
  focusChapter();
}

function focusChapter() {
  const model = getModel();
  if (chapterIndex === 0) {
    camera.desiredTargetY = 0;
    camera.desiredZoom = 0.98;
    camera.desiredYaw = -0.58;
    camera.desiredPitch = -0.12;
    return;
  }
  const components = model.components.filter((component) => component.chapter === chapterIndex && isModeVisible(component));
  const meanY = components.reduce((sum, component) => sum + component.y + ((component.repeat || 1) - 1) * (component.gap || 0), 0) / Math.max(1, components.length);
  camera.desiredTargetY = meanY;
  camera.desiredZoom = components.length > 2 ? 1.18 : 1.34;
  camera.desiredYaw = chapterIndex === 3 ? -0.72 : -0.5;
  camera.desiredPitch = -0.1;
}

function expandScene(model) {
  const blocks = [];
  model.components.forEach((component) => {
    if (!isModeVisible(component)) return;
    if (chapterIndex > 0 && component.chapter === chapterIndex && component.layers?.length) {
      const detailGap = Math.min(0.36, 1.8 / Math.max(1, component.layers.length - 1));
      const startY = component.y - detailGap * (component.layers.length - 1) * 0.5;
      component.layers.forEach((detail, index) => {
        blocks.push({
          ...component,
          ...detail,
          id: component.id,
          nodeId: `${component.id}::${index}`,
          parentId: component.id,
          instance: index,
          label: detail.label,
          shape: detail.output,
          x: component.x + (index % 2 ? 0.06 : -0.06),
          y: startY + detailGap * index,
          z: component.z + index * 0.035,
          h: Math.min(component.h, 0.28),
          isDetail: true,
          isTop: true,
          description: detail.description || `${detail.op}. Input ${detail.input}; output ${detail.output}.`,
        });
      });
      return;
    }
    const repeat = component.repeat || 1;
    for (let index = 0; index < repeat; index += 1) {
      blocks.push({
        ...component,
        nodeId: component.id,
        parentId: component.id,
        instance: index,
        x: component.x + (component.twist || 0) * index,
        y: component.y + (component.gap || 0) * index,
        z: component.z + (component.twist || 0) * index * 0.6,
        isTop: index === repeat - 1,
      });
    }
  });
  return blocks;
}

function selectModel(id) {
  if (!validModel(id)) return;
  selectedModelId = id;
  chapterIndex = 0;
  selectedComponentId = null;
  pathwayMode = "inference";
  const model = getModel();
  setAccent(model.accent);
  sceneBlocks = expandScene(model);
  renderModelTabs();
  renderMiniDiagram(model);
  renderChapter();
  renderLegend();
  el.sceneCategory.textContent = model.category.toUpperCase();
  el.sceneModelName.textContent = model.sceneName;
  el.sceneYear.textContent = model.year;
  el.sceneReference.textContent = model.reference;
  el.aboutSourceLink.href = model.source;
  el.inspector.classList.remove("open");
  renderPathwayToggle();
  resetCamera();
}

function renderLegend() {
  const model = getModel();
  el.visualLegend.innerHTML = Object.entries(TYPE_LABELS).map(([type, label]) => `
    <span class="legend-item" style="--legend-color:${TYPE_COLORS[type]}"><i></i>${label}</span>
  `).join("") + `<div class="tensor-legend"><b>TENSOR SYMBOLS</b>${(model.symbols || []).map((symbol) => `<span>${symbol}</span>`).join("")}</div>`;
}

function renderPathwayToggle() {
  el.pathwayToggle.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.pathway === pathwayMode);
  });
}

function setPathway(mode) {
  if (!['inference', 'training'].includes(mode) || mode === pathwayMode) return;
  pathwayMode = mode;
  selectedComponentId = null;
  el.inspector.classList.remove("open");
  sceneBlocks = expandScene(getModel());
  renderPathwayToggle();
  renderChapter();
  focusChapter();
}

function resetCamera() {
  camera.desiredYaw = -0.58;
  camera.desiredPitch = -0.12;
  camera.desiredZoom = chapterIndex === 0 ? 0.98 : 1.18;
  camera.desiredTargetY = 0;
  focusChapter();
}

function resizeCanvas() {
  const rect = el.canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  el.canvas.width = Math.max(1, Math.round(rect.width * ratio));
  el.canvas.height = Math.max(1, Math.round(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function rotatePoint(point) {
  const cy = Math.cos(camera.yaw);
  const sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch);
  const sp = Math.sin(camera.pitch);
  const x1 = point.x * cy - point.z * sy;
  const z1 = point.x * sy + point.z * cy;
  const y1 = (point.y - camera.targetY) * cp - z1 * sp;
  const z2 = (point.y - camera.targetY) * sp + z1 * cp;
  return { x: x1, y: y1, z: z2 };
}

function project(point) {
  const rect = el.canvas.getBoundingClientRect();
  const rotated = rotatePoint(point);
  const base = Math.min(rect.width / 15.5, rect.height / 18.3) * camera.zoom;
  const perspective = Math.max(0.68, Math.min(1.32, 1 + rotated.z * 0.026));
  return {
    x: rect.width * 0.5 + rotated.x * base * perspective,
    y: rect.height * 0.56 - rotated.y * base * perspective,
    z: rotated.z,
  };
}

function blockVertices(block) {
  const x0 = block.x - block.w / 2;
  const x1 = block.x + block.w / 2;
  const y0 = block.y - block.h / 2;
  const y1 = block.y + block.h / 2;
  const z0 = block.z - block.d / 2;
  const z1 = block.z + block.d / 2;
  return [
    { x: x0, y: y0, z: z0 }, { x: x1, y: y0, z: z0 }, { x: x1, y: y1, z: z0 }, { x: x0, y: y1, z: z0 },
    { x: x0, y: y0, z: z1 }, { x: x1, y: y0, z: z1 }, { x: x1, y: y1, z: z1 }, { x: x0, y: y1, z: z1 },
  ];
}

function hexToRgb(hex) {
  const normalized = hex.replace("#", "");
  const value = Number.parseInt(normalized.length === 3 ? normalized.split("").map((char) => char + char).join("") : normalized, 16);
  return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
}

function shade(hex, amount, alpha = 1) {
  const { r, g, b } = hexToRgb(hex);
  const mix = amount >= 0 ? 255 : 0;
  const ratio = Math.abs(amount);
  return `rgba(${Math.round(r + (mix - r) * ratio)},${Math.round(g + (mix - g) * ratio)},${Math.round(b + (mix - b) * ratio)},${alpha})`;
}

function polygonPath(points) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let index = 1; index < points.length; index += 1) ctx.lineTo(points[index].x, points[index].y);
  ctx.closePath();
}

function componentOpacity(component) {
  if (selectedComponentId === component.nodeId || selectedComponentId === component.id) return 1;
  if (chapterIndex === 0) return 0.82;
  return component.chapter === chapterIndex ? 1 : 0.13;
}

function drawGrid(points, grid, opacity) {
  if (!grid || opacity < 0.35) return;
  const [cols, rows] = grid;
  const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
  ctx.save();
  ctx.strokeStyle = `rgba(255,255,255,${0.25 * opacity})`;
  ctx.lineWidth = 0.55;
  for (let index = 1; index < Math.min(cols, 15); index += 1) {
    const t = index / Math.min(cols, 15);
    const a = lerp(points[0], points[1], t);
    const b = lerp(points[3], points[2], t);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  for (let index = 1; index < Math.min(rows, 6); index += 1) {
    const t = index / Math.min(rows, 6);
    const a = lerp(points[0], points[3], t);
    const b = lerp(points[1], points[2], t);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  ctx.restore();
}

function getComponentCenter(model, id) {
  const component = model.components.find((item) => item.id === id);
  if (!component) return null;
  return {
    x: component.x + (component.twist || 0) * ((component.repeat || 1) - 1) * 0.5,
    y: component.y + (component.gap || 0) * ((component.repeat || 1) - 1) * 0.5,
    z: component.z,
  };
}

function drawConnections(time) {
  const model = getModel();
  model.links.filter(isModeVisible).forEach((linkItem, index) => {
    const fromId = Array.isArray(linkItem) ? linkItem[0] : linkItem.from;
    const toId = Array.isArray(linkItem) ? linkItem[1] : linkItem.to;
    const fromComponent = model.components.find((component) => component.id === fromId);
    const toComponent = model.components.find((component) => component.id === toId);
    if (!fromComponent || !toComponent || !isModeVisible(fromComponent) || !isModeVisible(toComponent)) return;
    const from = getComponentCenter(model, fromId);
    const to = getComponentCenter(model, toId);
    if (!from || !to) return;
    const start = project(from);
    const end = project(to);
    const active = chapterIndex === 0 || fromComponent.chapter === chapterIndex || toComponent.chapter === chapterIndex;
    ctx.save();
    ctx.strokeStyle = active ? shade(getModel().accent, 0.05, 0.42) : "rgba(87,103,98,0.08)";
    ctx.lineWidth = active ? 1.15 : 0.65;
    ctx.setLineDash(active ? [] : [3, 4]);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    if (active) {
      const t = (time * 0.00022 + index * 0.13) % 1;
      const x = start.x + (end.x - start.x) * t;
      const y = start.y + (end.y - start.y) * t;
      ctx.fillStyle = shade(getModel().accent, 0.12, 0.9);
      ctx.beginPath(); ctx.arc(x, y, 2.3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  });
}

function drawInternalConnections() {
  const groups = new Map();
  sceneBlocks.filter((block) => block.isDetail).forEach((block) => {
    if (!groups.has(block.parentId)) groups.set(block.parentId, []);
    groups.get(block.parentId).push(block);
  });
  groups.forEach((blocks) => {
    blocks.sort((a, b) => a.instance - b.instance);
    ctx.save();
    ctx.strokeStyle = shade(TYPE_COLORS[blocks[0].type], -0.05, 0.55);
    ctx.lineWidth = 1;
    for (let index = 1; index < blocks.length; index += 1) {
      const from = project({ x: blocks[index - 1].x, y: blocks[index - 1].y, z: blocks[index - 1].z });
      const to = project({ x: blocks[index].x, y: blocks[index].y, z: blocks[index].z });
      ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
    }
    ctx.restore();
  });
}

function drawBlock(block) {
  const vertices3d = blockVertices(block);
  const projected = vertices3d.map(project);
  const transformed = vertices3d.map(rotatePoint);
  const opacity = componentOpacity(block);
  const color = TYPE_COLORS[block.type];
  const faces = [
    { ids: [0, 1, 2, 3], shade: -0.28 },
    { ids: [0, 4, 7, 3], shade: -0.16 },
    { ids: [1, 5, 6, 2], shade: -0.08 },
    { ids: [3, 2, 6, 7], shade: 0.22 },
    { ids: [4, 5, 6, 7], shade: 0.03, front: true },
  ].map((face) => ({ ...face, depth: face.ids.reduce((sum, id) => sum + transformed[id].z, 0) / face.ids.length }))
    .sort((a, b) => a.depth - b.depth);

  faces.forEach((face) => {
    const points = face.ids.map((id) => projected[id]);
    polygonPath(points);
    ctx.fillStyle = shade(color, face.shade, opacity);
    ctx.fill();
    ctx.strokeStyle = shade(color, -0.28, Math.min(0.62, opacity + 0.08));
    ctx.lineWidth = selectedComponentId === block.nodeId ? 1.7 : 0.65;
    ctx.stroke();
    if (face.front) drawGrid(points, block.grid, opacity);
  });

  if (block.isTop) {
    hitTargets.push({ id: block.nodeId, polygon: [projected[4], projected[5], projected[6], projected[7]], depth: transformed[4].z });
  }
}

function drawLabel(component) {
  if (!component.isTop) return;
  if (chapterIndex !== 0 && component.chapter !== chapterIndex && selectedComponentId !== component.nodeId) return;
  const point = project({
    x: component.x,
    y: component.y + component.h / 2,
    z: component.z + component.d / 2,
  });
  const opacity = componentOpacity(component);
  if (opacity < 0.3) return;
  ctx.save();
  ctx.font = "650 8px Inter, sans-serif";
  const labelWidth = ctx.measureText(component.label).width;
  ctx.font = "600 7px ui-monospace, SFMono-Regular, Menlo, monospace";
  const shapeWidth = ctx.measureText(component.shape || "").width;
  const width = Math.max(labelWidth, shapeWidth) + 16;
  const x = point.x + 7;
  const y = point.y - 6;
  ctx.fillStyle = `rgba(250,251,248,${0.88 * opacity})`;
  ctx.strokeStyle = shade(TYPE_COLORS[component.type], -0.08, 0.58 * opacity);
  ctx.lineWidth = 0.7;
  ctx.beginPath(); ctx.roundRect(x, y - 13, width, component.shape ? 27 : 18, 3); ctx.fill(); ctx.stroke();
  ctx.fillStyle = `rgba(28,38,35,${0.86 * opacity})`;
  ctx.font = "650 8px Inter, sans-serif";
  ctx.fillText(component.label, x + 7, y + 1);
  if (component.shape) {
    ctx.fillStyle = `rgba(73,84,80,${0.78 * opacity})`;
    ctx.font = "600 7px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(component.shape, x + 7, y + 11);
  }
  ctx.restore();
}

function renderScene(time = 0) {
  const rect = el.canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  camera.yaw += (camera.desiredYaw - camera.yaw) * 0.09;
  camera.pitch += (camera.desiredPitch - camera.pitch) * 0.09;
  camera.zoom += (camera.desiredZoom - camera.zoom) * 0.09;
  camera.targetY += (camera.desiredTargetY - camera.targetY) * 0.09;
  hitTargets = [];
  drawConnections(time);
  drawInternalConnections();
  [...sceneBlocks]
    .sort((a, b) => rotatePoint({ x: a.x, y: a.y, z: a.z }).z - rotatePoint({ x: b.x, y: b.y, z: b.z }).z)
    .forEach(drawBlock);
  sceneBlocks.forEach(drawLabel);
  requestAnimationFrame(renderScene);
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x; const yi = polygon[i].y;
    const xj = polygon[j].x; const yj = polygon[j].y;
    const intersects = ((yi > point.y) !== (yj > point.y)) && (point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + Number.EPSILON) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

function openInspector(component) {
  const parent = getModel().components.find((item) => item.id === (component.parentId || component.id)) || component;
  selectedComponentId = component.nodeId || component.id;
  el.inspectorType.textContent = `${TYPE_LABELS[component.type].toUpperCase()} · CHAPTER ${parent.chapter + 1}${component.isDetail ? " · INTERNAL LAYER" : ""}`;
  el.inspectorTitle.textContent = component.label;
  el.inspectorBody.textContent = component.description || parent.description;
  const specs = [
    ["Input", component.input || parent.input || "—"],
    ["Output", component.output || component.shape || parent.output || parent.shape || "—"],
    ["Operation", component.op || parent.op || "—"],
    ["Normalization", component.norm || parent.norm || "—"],
    ["Activation", component.activation || parent.activation || "—"],
    ["Repetition", component.isDetail ? `layer ${component.instance + 1} of ${parent.layers.length} shown` : (parent.repeatLabel || (parent.repeat > 1 ? `×${parent.repeat}` : "once"))],
  ];
  el.inspectorSpecs.innerHTML = specs.map(([term, value]) => `<div><dt>${term}</dt><dd>${value}</dd></div>`).join("");
  el.inspector.style.setProperty("--accent", TYPE_COLORS[component.type]);
  el.inspector.classList.add("open");
}

function inspectAt(clientX, clientY) {
  const rect = el.canvas.getBoundingClientRect();
  const point = { x: clientX - rect.left, y: clientY - rect.top };
  const hit = [...hitTargets].sort((a, b) => b.depth - a.depth).find((target) => pointInPolygon(point, target.polygon));
  if (!hit) return;
  const component = sceneBlocks.find((item) => item.nodeId === hit.id);
  if (component) openInspector(component);
}

function togglePlayback() {
  isPlaying = !isPlaying;
  clearInterval(playTimer);
  el.playButton.classList.toggle("playing", isPlaying);
  el.playButton.textContent = isPlaying ? "Ⅱ" : "▶";
  el.playbackHint.textContent = isPlaying ? "Playing · Space to pause" : "Press Space to continue";
  if (isPlaying) {
    playTimer = setInterval(() => {
      setChapter(chapterIndex + 1);
    }, 6500);
  }
}

function continueWalkthrough() {
  const model = getModel();
  setChapter(chapterIndex === model.chapters.length - 1 ? 0 : chapterIndex + 1);
}

function renderComparison() {
  const modelA = getModel(el.compareA.value);
  const modelB = getModel(el.compareB.value);
  const flow = (model) => `<div class="compare-flow" style="--flow-color:${model.accent}">${model.components.filter((component) => [1, 2, 3, 4, 5].includes(component.chapter)).reduce((items, component) => {
    if (!items.some((item) => item.chapter === component.chapter)) items.push(component);
    return items;
  }, []).map((component, index) => `${index ? "<i>→</i>" : ""}<span title="${component.label}">${component.label}</span>`).join("")}</div>`;
  el.compareVisual.innerHTML = flow(modelA) + flow(modelB);
  const keys = Object.keys(modelA.compare);
  el.compareHead.innerHTML = `<tr><th>Design decision</th><th>${modelA.tab}</th><th>${modelB.tab}</th></tr>`;
  el.compareBody.innerHTML = keys.map((key) => `<tr><td>${key}</td><td>${modelA.compare[key]}</td><td>${modelB.compare[key]}</td></tr>`).join("");
}

function setupComparison() {
  const options = models.map((model) => `<option value="${model.id}">${model.tab}</option>`).join("");
  el.compareA.innerHTML = options;
  el.compareB.innerHTML = options;
  el.compareA.value = selectedModelId;
  el.compareB.value = selectedModelId === "dreamzero" ? "gr00t" : "dreamzero";
  renderComparison();
}

el.previous.addEventListener("click", () => setChapter(chapterIndex - 1));
el.next.addEventListener("click", () => setChapter(chapterIndex + 1));
el.continueButton.addEventListener("click", continueWalkthrough);
el.skipButton.addEventListener("click", () => setChapter(5));
el.playButton.addEventListener("click", togglePlayback);
el.pathwayToggle.addEventListener("click", (event) => {
  const button = event.target.closest("[data-pathway]");
  if (button) setPathway(button.dataset.pathway);
});
document.querySelector("#resetCamera").addEventListener("click", resetCamera);
document.querySelector("#fullscreenButton").addEventListener("click", () => {
  el.visualPanel.classList.toggle("fullscreen");
  setTimeout(resizeCanvas, 180);
});
document.querySelector("#closeInspector").addEventListener("click", () => {
  selectedComponentId = null;
  el.inspector.classList.remove("open");
});

el.canvas.addEventListener("pointerdown", (event) => {
  dragState = { x: event.clientX, y: event.clientY, yaw: camera.desiredYaw, pitch: camera.desiredPitch };
  didDrag = false;
  el.canvas.setPointerCapture(event.pointerId);
});

el.canvas.addEventListener("pointermove", (event) => {
  if (!dragState) return;
  const dx = event.clientX - dragState.x;
  const dy = event.clientY - dragState.y;
  if (Math.abs(dx) + Math.abs(dy) > 4) didDrag = true;
  camera.desiredYaw = dragState.yaw + dx * 0.008;
  camera.desiredPitch = Math.max(-0.75, Math.min(0.55, dragState.pitch + dy * 0.006));
});

el.canvas.addEventListener("pointerup", (event) => {
  if (!didDrag) inspectAt(event.clientX, event.clientY);
  dragState = null;
});

el.canvas.addEventListener("pointercancel", () => { dragState = null; });
el.canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  camera.desiredZoom = Math.max(0.58, Math.min(2.2, camera.desiredZoom * Math.exp(-event.deltaY * 0.0012)));
}, { passive: false });

document.addEventListener("keydown", (event) => {
  if (event.key === " " && !["SELECT", "INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    if (isPlaying) togglePlayback();
    else continueWalkthrough();
  }
  if (event.key === "ArrowRight") setChapter(chapterIndex + 1);
  if (event.key === "ArrowLeft") setChapter(chapterIndex - 1);
  if (event.key === "Escape" && el.visualPanel.classList.contains("fullscreen")) {
    el.visualPanel.classList.remove("fullscreen");
    setTimeout(resizeCanvas, 180);
  }
});

document.querySelector("#compareButton").addEventListener("click", () => {
  setupComparison();
  el.compareDialog.showModal();
});
document.querySelector("#aboutButton").addEventListener("click", () => el.aboutDialog.showModal());
document.querySelectorAll("[data-close]").forEach((button) => {
  button.addEventListener("click", () => document.querySelector(`#${button.dataset.close}Dialog`).close());
});
[el.compareDialog, el.aboutDialog].forEach((dialog) => {
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
});
el.compareA.addEventListener("change", renderComparison);
el.compareB.addEventListener("change", renderComparison);

new ResizeObserver(resizeCanvas).observe(el.visualPanel);
const initialChapter = chapterIndex;
const initialPathway = pathwayMode;
selectModel(selectedModelId);
if (initialPathway !== pathwayMode) setPathway(initialPathway);
setChapter(initialChapter, { keepScroll: true });
resizeCanvas();
requestAnimationFrame(renderScene);
