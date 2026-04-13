Set up a new weekly session for this deep learning study plan. Creates the folder, exercises, quiz, and answers files — all tailored to the student's progress, past quiz scores, and career path toward NVIDIA DRIVE Mapping roles.

**Arguments:** The week's topic name (e.g. `perceptron` or `cnns`). If omitted, determine the next topic automatically from `study-plan.md`.

---

## Step 1 — Create the folder

1. List `weeks/` to find the highest existing `WNN_` prefix. Increment by 1 (zero-padded to 2 digits) to get `WNN`. If `weeks/` does not exist, create it and start at `W01`.
2. Create `weeks/WNN_$ARGUMENTS/`.

---

## Step 2 — Active Recall Review (Spaced Repetition)

Read `progress.md`. Find all dated entries that have a **Carry-Forward** section. For each, note the entry date and its carry-forward items (quiz questions and failed exercises).

Apply spaced repetition based on how old each entry is relative to today:

| Entry age | Quiz questions to include | Exercises to include |
|---|---|---|
| Most recent entry | Scored ≤ 0.5 only | All failed exercises |
| ~7 days ago | Scored ≤ 0.5 only | None |
| ~14+ days ago | Scored ≤ 0.5 only | None |

Questions scored 0.75 are not carried forward — partial credit means the concept is understood well enough to move on.

Collect all qualifying quiz questions as the **Active Recall Warm-Up** for the new exercises file, labelled by source week (e.g. *"From Week 2 — Review"*).

Collect all qualifying failed exercises as **Repeat Exercises**, labelled by source week.

If `progress.md` has no graded entries yet, skip this step.

---

## Step 2b — Read the student understanding snapshot

Read `memory.md` at the repo root. This file documents:
- What DL concepts the student currently understands solidly
- Known gaps in math/ML foundations and "why" reasoning
- Learning patterns and what kinds of explanations land well
- Carry-forward items not yet captured in `progress.md`
- Specific recommendations for upcoming weeks

Use this to calibrate:
- **Background section depth** — if a math or ML fundamental is listed as a gap, introduce it explicitly rather than assuming it
- **Exercise scaffolding** — lean toward more or less guidance based on the student's applied vs conceptual balance
- **Quiz question framing** — target question types that address noted gaps (e.g. "why" questions if reasoning lags implementation)
- **Framework context** — note whether the student is more comfortable in PyTorch or TensorFlow and adjust examples accordingly
- **Concept explainers** — if `memory.md` recommends pre-reading a `concepts/` file, link to it in the Background section

---

## Step 3 — Determine the next topic

Read `progress.md` to see what has been covered and the most recent Carry-Forward state.
Read `study-plan.md` for the full week-by-week roadmap.

Use the roadmap to identify the next uncompleted week. The sequence is:
1. Week 1 — Foundations: perceptron, loss functions, gradient descent, backpropagation
2. Week 2 — Deep Sequence Modeling: RNNs, LSTMs, attention, Transformers
3. Week 3 — Deep Computer Vision: CNNs, convolutions, feature hierarchy
4. Week 4 — Deep Generative Modeling: VAEs, GANs, diffusion models
5. Week 5 — Deep Reinforcement Learning: DQN, policy gradients, CartPole
6. Week 6 — LLM Fine-Tuning + New Frontiers: fine-tuning, LoRA, distributed training
7. Weeks 7–9 — Portfolio Projects (geospatial DL, AV perception, LLM for mapping)
8. Week 10 — Integration, review, portfolio polish

If $ARGUMENTS was provided, use that as the topic name and align the content to that week's spec in `study-plan.md`.

---

## Step 4 — Write the concepts pre-reading document(s)

**Before writing:** Check if a concepts file for this week's topic already exists in `concepts/` (e.g. `concepts/W01_perceptron.md`).
- If it exists: read it and skip creation unless carry-forward items from Step 2 suggest a gap that warrants a new companion concepts file.
- If it does not exist: create it.

Write one document per major conceptual area if the topic is broad enough to warrant splitting (e.g. Week 2 might produce `W02_rnns.md` and `W02_attention.md`). For most weeks, one file suffices.

File path: `concepts/WNN_topic.md` (or `concepts/WNN_topic_subtopic.md` if split).

Structure:

```
# Concepts: [Topic Name]

> Pre-reading for Week N exercises. Read this before opening the exercises file.

## What This Is About
[1 paragraph — the core problem this concept solves. Why does it exist? What was broken before it?]

## The Math (Plain English First)
[Explain the key equations intuitively before showing the symbols.
For each equation: state what it computes, what each term means, and what happens when you change it.
Keep LaTeX minimal — use inline code blocks for formulas where possible.]

## Intuition Builder
[An analogy or visual description that makes the concept stick.
Tie to something the student already knows: NumPy operations, coordinate systems, map data.]

## How It Works in Practice
[Short annotated code snippet in PyTorch and/or TensorFlow showing the concept in action.
Prefer runnable, minimal examples over exhaustive ones.]

## Common Pitfalls
[2–4 bullet points — mistakes beginners make with this concept and how to spot them]

## Connection to NVIDIA DRIVE / Mapping
[1 paragraph — how this concept appears in the AV or mapping stack. Be specific: which module, which task, what failure mode does understanding this prevent.]

## Quick-Check Questions
[3 short questions the student can answer mentally to verify they understood the pre-reading before starting exercises. No answers here — these are self-check prompts.]
```

After writing the concepts file(s), reference them explicitly at the top of the exercises file Background section with a link (e.g. `> Pre-reading: [concepts/W01_perceptron.md](../../concepts/W01_perceptron.md)`).

---

## Step 5 — Write the exercises file

**Before writing:** Check if `WNN_exercises.md` already exists in the week folder.
- If it exists: read it, check whether the Active Recall Warm-Up and Repeat Exercises sections are present and match the carry-forward items from Step 2. Add or update those sections only — do not delete or recreate the file.
- If it does not exist: create it.

File path: `weeks/WNN_topic/WNN_exercises.md`.

Structure:

```
# Week N — [Topic Name]

**Goal:** [One sentence — what the student will be able to do after this week, tied to the NVIDIA DRIVE Mapping context]

## Active Recall Warm-Up
[Only if there are carried-over questions from Step 2. Label each with its source week.
Direct the student to write answers in the previous week's answers file before reading on.]

---

## Background
[2–4 paragraphs. Explain the concept from first principles.
Use concrete intuition: diagrams described in text, analogies to Python/NumPy the student already knows.
Connect to the NVIDIA DRIVE / mapping domain concretely — e.g., how CNNs underpin perception in DRIVE, how sequence models relate to sensor time-series.]

## Exercise 1 — [Name]
**File:** Create `weeks/WNN_topic/filename.py` (new file) OR work in `filename.ipynb` (notebook).
[Clear task. Include a code scaffold where helpful — e.g., class stubs, data loading boilerplate.]
**What to observe:** [Connect the output or behavior to the broader concept and career relevance]

## Exercise 2 — [Name]
**File:** Update `filename.py` (add to existing file) OR Create `filename2.py` — whichever fits.
...

## Exercise 3 — [Name]
**File:** Update `filename.py` OR Create `filename3.py` — whichever fits.
...

[Add Exercise 4 and 5 if the topic warrants it]

[If there are repeat exercises from Step 2, add a clearly labeled section:]
## Repeat Exercises (from Week X)
[Re-state the exercise with a note that it was previously incomplete]

## Checkpoint
You've passed Week N when you can:
- [bullet — implementation ability]
- [bullet — conceptual understanding]
- [bullet — connect to the broader DL/AV context]
```

---

## Step 6 — Write the quiz file

**Before writing:** Check if `WNN_topic_quiz.md` already exists in the week folder.
- If it exists: read it, check whether carried-over warm-up questions are present. Add missing ones only — do not delete existing questions, answers, or the Grade Log.
- If it does not exist: create it.

File path: `weeks/WNN_topic/WNN_topic_quiz.md`.

Structure:

```
# Quiz — Week N: [Topic Name]

---

[If there are carried-over questions from Step 2:]
## Warm-Up (From Week X — Review)

**QR1:** [Carried-over question verbatim]

## Answer QR1
[blank for student]

---

## Question 1
[New question for this week's topic]
...

## Answer 1
[blank for student]

---

[Continue for 4–5 new questions]

---

## Grade Log
```

**Quiz question guidelines:**
- Test *reasoning*, not just recall — ask "why" and "when", not only "what"
- Use realistic AV/mapping scenarios: camera feeds, LiDAR point clouds, HD map tile updates, sensor fusion, model deployment on DRIVE hardware
- Mix question types: explain the concept, spot the flaw in this architecture, predict the training behavior, choose between two design options and justify it
- Each question should be answerable in 2–5 sentences; avoid yes/no questions
- At least one question per week should connect the topic to autonomous driving or geospatial mapping

---

## Step 7 — Write the answers file

**Before writing:** Check if `WNN_topic_answers.md` already exists in the week folder.
- If it exists: read it, check whether answers for any carried-over warm-up questions are present. Add missing answers only — do not modify or delete existing content.
- If it does not exist: create it.

File path: `weeks/WNN_topic/WNN_topic_answers.md`.

Structure:

```
# Answers — Week N: [Topic Name]

---

## Question [N]

**[The correct answer / concept / design choice]**

[Explanation of *why* — 2–5 sentences. Focus on intuition, not just correctness.
For carried-over questions, note which week they came from.
Include a short code snippet (PyTorch or TensorFlow) for any question that involves implementation.]

---
```

Tone: encouraging, builds intuition over memorisation, connects each answer to the broader DL mental model and the NVIDIA DRIVE Mapping career context.

---

## Final Output

After creating all files, print:
- The new week folder path
- The topic chosen and why (one sentence)
- Concepts file(s) created (list paths)
- Carried-over warm-up questions, if any (list by source week + question number)
- Repeated exercises, if any (list by source week + exercise number)
- The full list of files created or updated (note which were new vs modified)
