# MIT 6.S191 — Intro to Deep Learning: Study Plan

**Course:** MIT 6.S191 (January 2026 edition, Alexander & Ava Amini)
**Format:** Self-paced, following recorded lectures + slides
**Time budget:** 8–10 hrs/week dedicated to DL (separate from C++ track)
**Duration:** 10 weeks (April 7 – June 13, 2026)
**Frameworks:** Both PyTorch and TensorFlow (as the course uses them)
**Compute:** Local NVIDIA GPU (CUDA)
**Goal:** Build DL foundations + 3 portfolio projects targeting NVIDIA DRIVE Mapping roles

---

## Pre-requisites Checklist

Before Week 1, verify your local environment is ready:

- Python 3.10+ installed
- PyTorch installed with CUDA support (`torch.cuda.is_available()` returns True)
- TensorFlow installed with GPU support (`tf.config.list_physical_devices('GPU')` returns your GPU)
- Jupyter notebooks working (local or VS Code)
- Clone course labs: `git clone https://github.com/MITDeepLearning/introtodeeplearning`
- NumPy refresher: comfortable with array indexing, broadcasting, reshaping

---

## Week 1 (Apr 7–13): Foundations — The Perceptron to Neural Networks

**Lecture:** L1 — Introduction to Deep Learning (review, already completed)
**Hours:** ~8 hrs

### Theory (3 hrs)

- Re-watch L1 video with slides, take structured notes
- Focus on: perceptron math (weighted sum + bias + activation), loss functions, gradient descent, backpropagation
- Key equations to internalize: forward pass `ŷ = g(w₀ + Σwᵢxᵢ)`, loss `J(W)`, gradient update `W ← W - η∇J(W)`

### Coding (4 hrs)

- **NumPy from scratch:** Implement a single perceptron that learns AND, OR, XOR gates
- **PyTorch intro:** Rebuild the same perceptron using `torch.nn.Linear` and `torch.optim.SGD`
- **TensorFlow intro:** Same exercise in TF/Keras to compare APIs
- Install and run the course Lab 1 notebook to verify environment

### Quiz (1 hr)

- I'll quiz you on: activation functions (sigmoid vs ReLU vs softmax — when to use each), what happens when learning rate is too high/low, why we need non-linear activations, the chain rule in backpropagation

### Deliverable

- `week1-perceptron/` folder with numpy, pytorch, and tensorflow implementations
- Notes file: `week1-notes.md`

---

## Week 2 (Apr 14–20): Deep Sequence Modeling — RNNs to Transformers

**Lecture:** L2 — Deep Sequence Modeling (Ava Amini)
**Hours:** ~9 hrs

### Theory (3 hrs)

- Watch L2 video with slides
- Core concepts: why sequences need memory, RNN cell mechanics (hidden state recurrence), vanishing/exploding gradients, LSTM gates (forget, input, output), attention mechanism basics, self-attention and Transformers
- Draw the RNN unrolled computation graph by hand

### Coding (5 hrs)

- **Lab 1 — Music Generation:** Complete the course's music generation lab (build an RNN that learns Irish folk songs and generates new ones)
- Implement in both TF (course default) and then port key parts to PyTorch
- Experiment: change hyperparameters (sequence length, hidden units, temperature) and document what happens

### Quiz (1 hr)

- I'll quiz you on: why vanilla RNNs struggle with long sequences, how LSTM gates solve this, what "attention" means intuitively, how self-attention differs from RNN sequential processing

### Deliverable

- Completed Lab 1 with your own experiments documented
- `week2-notes.md` covering RNN → LSTM → Attention → Transformer progression

---

## Week 3 (Apr 21–27): Deep Computer Vision — CNNs

**Lecture:** L3 — Deep Computer Vision (Alexander Amini)
**Hours:** ~9 hrs

### Theory (3 hrs)

- Watch L3 video with slides
- Core concepts: convolution operation (filters, stride, padding), feature maps, pooling, how CNNs build hierarchical features (edges → textures → objects), classic architectures (LeNet, AlexNet, ResNet concept)
- Connect to AV context: this is exactly what NVIDIA DRIVE uses for perception

### Coding (5 hrs)

- **Lab 2 — Facial Detection Systems:** Complete the course's facial detection lab with bias mitigation
- Read the accompanying AAAI bias paper (`AAAI_MitigatingAlgorithmicBias.pdf`)
- Build a CNN classifier from scratch on CIFAR-10 or MNIST in PyTorch (not using pre-built models)
- Experiment with: number of conv layers, filter sizes, with/without batch normalization

### Quiz (1 hr)

- I'll quiz you on: what a convolution filter actually computes, why CNNs are better than fully-connected networks for images, what pooling does and why, how feature hierarchy emerges, what the bias paper's key finding was

### Deliverable

- Completed Lab 2
- Your own CNN classifier with training curves plotted
- `week3-notes.md`

---

## Week 4 (Apr 28 – May 4): Deep Generative Modeling — VAEs, GANs, Diffusion

**Lecture:** L4 — Deep Generative Modeling (Ava Amini)
**Hours:** ~9 hrs

### Theory (3 hrs)

- Watch L4 video with slides
- Core concepts: supervised vs unsupervised, latent variables, VAE architecture (encoder → latent space → decoder), the reparameterization trick, GAN architecture (generator vs discriminator), diffusion models (forward noise process → learned reverse denoising)
- Understand the key difference: VAEs optimize a reconstruction + KL divergence loss, GANs play a minimax game, diffusion models learn to denoise

### Coding (5 hrs)

- Implement a simple VAE on MNIST in PyTorch — generate new handwritten digits
- Implement a basic GAN on MNIST — compare training stability vs VAE
- Visualize the latent space of your VAE (2D latent space, plot digit clusters)

### Quiz (1 hr)

- I'll quiz you on: what "generative" means vs "discriminative", why GANs are hard to train (mode collapse, training instability), what the latent space represents, how diffusion models work at a high level

### Deliverable

- VAE and GAN implementations with generated samples
- Latent space visualization
- `week4-notes.md`

---

## Week 5 (May 5–11): Deep Reinforcement Learning

**Lecture:** L5 — Deep Reinforcement Learning (Alexander Amini)
**Hours:** ~9 hrs

### Theory (3 hrs)

- Watch L5 video with slides
- Core concepts: agent-environment loop, states/actions/rewards, policy vs value functions, Q-learning, deep Q-networks (DQN), policy gradient methods
- AV connection: RL is used for planning and decision-making in autonomous driving

### Coding (5 hrs)

- Implement DQN on CartPole (OpenAI Gym / Gymnasium) in PyTorch
- Experiment with: replay buffer size, epsilon-greedy exploration schedule, target network update frequency
- Stretch: try a slightly harder environment (LunarLander)

### Quiz (1 hr)

- I'll quiz you on: exploration vs exploitation, why we need experience replay, what a target network is and why it stabilizes training, policy gradient vs value-based methods

### Deliverable

- DQN agent that solves CartPole (>195 avg reward)
- Training reward curve plotted
- `week5-notes.md`

---

## Week 6 (May 12–18): LLM Fine-Tuning + New Frontiers

**Lectures:** L6 — Limitations & New Frontiers, plus L8 — Massively Parallel Training
**Hours:** ~10 hrs

### Theory (3 hrs)

- Watch L6 and L8 videos with slides
- L6 topics: limitations of current DL (adversarial examples, data efficiency, interpretability), emerging directions
- L8 topics: distributed training, data parallelism, model parallelism — directly relevant to NVIDIA's GPU infrastructure

### Coding (6 hrs)

- **Lab 3 — LLM Fine-Tuning:** Complete the course lab (fine-tune Gemma in a mystery style, evaluate with AI judge)
- This uses TensorFlow/Keras — follow the course notebook
- Document: what is fine-tuning vs training from scratch, what is LoRA/parameter-efficient fine-tuning, how the evaluation works

### Quiz (1 hr)

- I'll quiz you on: why fine-tuning is more practical than training from scratch, what adversarial examples are and why they matter for AV, data vs model parallelism, why NVIDIA GPUs dominate DL training

### Deliverable

- Completed Lab 3 with evaluation results
- `week6-notes.md`

---

## Week 7 (May 19–25): Portfolio Project 1 — Geospatial + Deep Learning

**Project:** Apply DL to satellite/aerial imagery — leverages your geospatial domain expertise
**Hours:** ~10 hrs

### Project Spec

- **Dataset:** Choose one of: EuroSAT (satellite image classification), SpaceNet (building footprint segmentation), or Inria Aerial Image Labeling
- **Model:** Transfer learning with a pre-trained ResNet or EfficientNet, fine-tuned on your chosen dataset
- **Goal:** Train a model that classifies land use or segments buildings from aerial imagery
- **Why this matters for NVIDIA:** DRIVE Mapping ingests aerial/satellite data to build HD maps — this is directly in the pipeline

### Deliverable

- GitHub repo with clean code, README, and results
- Training curves, confusion matrix, sample predictions visualized
- `portfolio-project-1-geospatial.md` write-up

---

## Week 8 (May 26 – Jun 1): Portfolio Project 2 — Computer Vision for Autonomous Vehicles

**Project:** Object detection or semantic segmentation on driving data
**Hours:** ~10 hrs

### Project Spec

- **Dataset:** Choose one of: KITTI (2D/3D object detection), nuScenes (multi-modal AV dataset), or BDD100K (driving video dataset)
- **Model:** Fine-tune a pre-trained model (YOLO, Faster R-CNN, or a segmentation model like DeepLab) on driving scenes
- **Goal:** Detect vehicles, pedestrians, cyclists in driving footage OR segment road/sidewalk/vehicle pixels
- **Why this matters for NVIDIA:** This is the core perception task for DRIVE — exactly what their team builds

### Deliverable

- GitHub repo with training pipeline, evaluation metrics (mAP or mIoU), sample detections
- Comparison: your model vs baseline
- `portfolio-project-2-av-perception.md` write-up

---

## Week 9 (Jun 2–8): Portfolio Project 3 — LLM Application for Mapping/GIS

**Project:** Fine-tune or prompt-engineer an LLM for a geospatial/mapping task
**Hours:** ~10 hrs

### Project Spec

- **Option A:** Fine-tune a small LLM to parse natural language location descriptions into structured geospatial queries (e.g., "find all intersections within 500m of a school" → GIS query)
- **Option B:** Build a RAG (Retrieval-Augmented Generation) system that answers questions about OpenStreetMap data or HD map specifications
- **Option C:** Create an LLM-powered tool that generates or validates map feature annotations

### Deliverable

- GitHub repo with working demo
- Evaluation metrics appropriate to your chosen task
- `portfolio-project-3-llm-geo.md` write-up

---

## Week 10 (Jun 9–13): Integration, Review & Portfolio Polish

**Hours:** ~8 hrs

### Review (3 hrs)

- Revisit any weak areas from quizzes throughout the course
- Re-read key lecture notes
- Final comprehensive quiz covering all 6 lectures

### Portfolio Polish (5 hrs)

- Clean up all 3 project repos: consistent READMEs, requirements.txt, clear documentation
- Write a short portfolio summary connecting all projects to the NVIDIA DRIVE Mapping role
- Update resume/LinkedIn with new DL skills and projects
- Prepare 2-minute verbal pitch for each project (interview prep)

### Deliverable

- 3 polished GitHub repos ready for resume/applications
- Portfolio summary document
- Updated resume bullet points

---

## Progress Tracking


| Week | Lecture     | Lab                 | Project          | Quiz        | Status |
| ---- | ----------- | ------------------- | ---------------- | ----------- | ------ |
| 1    | L1 (review) | Env setup           | —                | Foundations |        |
| 2    | L2          | Lab 1 (Music Gen)   | —                | Sequences   |        |
| 3    | L3          | Lab 2 (Facial Det.) | —                | CNNs        |        |
| 4    | L4          | —                   | —                | Generative  |        |
| 5    | L5          | —                   | —                | RL          |        |
| 6    | L6 + L8     | Lab 3 (LLM)         | —                | Frontiers   |        |
| 7    | —           | —                   | Geospatial+DL    | —           |        |
| 8    | —           | —                   | AV Perception    | —           |        |
| 9    | —           | —                   | LLM for Mapping  | —           |        |
| 10   | —           | —                   | Portfolio polish | Final       |        |


---

## Key Resources

- **Course website:** introtodeeplearning.com
- **Course GitHub (labs):** github.com/MITDeepLearning/introtodeeplearning
- **Bias paper:** `AAAI_MitigatingAlgorithmicBias.pdf` (in this folder)
- **PyTorch docs:** pytorch.org/docs
- **TensorFlow docs:** tensorflow.org/api_docs
- **NVIDIA DRIVE overview:** developer.nvidia.com/drive

## How This Connects to the C++ Track

Your C++ curriculum and this DL curriculum are parallel tracks that converge at the NVIDIA application. Here's how they reinforce each other:

- **C++ gives you the systems foundation** — memory management, performance, real-time constraints
- **DL gives you the AI/ML competence** — model architectures, training pipelines, perception systems
- **Together** — you'll understand both sides of the NVIDIA DRIVE stack: the C++/CUDA inference engine AND the Python-side model training pipeline
- **By June 1:** C++ fundamentals solid + 3 DL portfolio projects completed = credible application package

