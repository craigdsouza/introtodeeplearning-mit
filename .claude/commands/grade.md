Grade the student's quiz and coding exercises for the week specified in $ARGUMENTS (e.g. `01`). If no argument is given, ask which week to grade.

---

## Part 1 — Grade the Quiz

1. Find the quiz file at `weeks/<id>_*/` matching the argument (e.g. `weeks/01_perceptron/01_perceptron_quiz.md`).
2. Check the **Grade Log** section of the quiz file. If an entry already exists for today's date, skip grading and inform the user: "Quiz already graded today — run `/grade` again tomorrow or on the next session."
3. Find the corresponding answers file in the same folder (e.g. `01_perceptron_answers.md`).
4. Grade each question using this rubric:
   - **1.0** — correct answer + complete reasoning
   - **0.75** — correct answer + partial reasoning (key insight present but something missing)
   - **0.5** — correct answer + no reasoning
   - **0.0** — wrong answer or unanswered
5. Show a breakdown table to the user: one row per question with score and a one-line note on what was missing (if anything).
6. Show the total quiz score out of the maximum.
7. Append a new entry to the **Grade Log** section at the bottom of the quiz file with today's date, the total score, and the per-question breakdown table (Q, Score, Note). If no Grade Log section exists yet, create one.

---

## Part 2 — Grade the Exercises

1. Find the exercises file in the same week folder (e.g. `01_exercises.md`).
2. Check the **Exercise Results** section of the exercises file. If an entry already exists for today's date, skip grading and inform the user: "Exercises already graded today."
3. Read all exercises in the file.
4. For each exercise, look for evidence of completion:
   - Is there a `.py` or `.ipynb` file in the week folder that corresponds to this exercise?
   - Does the code in that file appear to address the exercise goal?
   - Is there any explicit pass/fail marker already written in the exercises file?
5. For each exercise assign **Pass** or **Fail**:
   - **Pass** — a corresponding `.py` or `.ipynb` exists and addresses the exercise goal, OR the exercise is explicitly marked as passed
   - **Fail** — no file found, or the file clearly doesn't address the exercise goal, OR explicitly marked as failed
   - **Not attempted** — treat as Fail for carry-forward purposes
6. Show an exercise summary table to the user: one row per exercise with Pass/Fail and a one-line note.
7. Append a **Exercise Results** section to the exercises file using this format (add after the last exercise, before any existing status section; if already present, update it):

```markdown
## Exercise Results — [DATE]

| Exercise | Result | Note |
|----------|--------|------|
| Exercise 1 — [Name] | Pass | [brief note or blank] |
| Exercise 2 — [Name] | Fail | [what was missing] |
```

---

## Part 3 — Update progress.md

Read `progress.md`. Append a new dated entry at the end of the file in this format:

```markdown
## [DATE] — Week [N]: [Topic Name]

### Quiz Score
[Total] / [Max] — [one sentence characterising performance, e.g. "Strong on architecture intuition, gaps in backprop math"]

| Q | Score | Note |
|---|-------|------|
| Q1 | X.X | ... |
...

### Exercises
| Exercise | Result |
|----------|--------|
| Exercise 1 — [Name] | Pass |
| Exercise 2 — [Name] | Fail |
...

### Concepts Confirmed
[2–4 bullet points of what the student demonstrably understands based on their quiz answers — not aspirational, only what the graded answers actually show]

### Carry-Forward
[List quiz questions scored ≤ 0.5 and any failed exercises as one-line items — these will be picked up by /project for the next week's warm-up. Questions scored 0.75 are NOT listed here. If nothing qualifies, write "None — clean sweep."]
```

---

## Part 4 — Update memory.md

Read `memory.md` at the repo root. Update it to reflect what was learned and observed this session. This is not a log — it is a **living snapshot** of the student's current state. Rewrite sections in place rather than appending.

Apply the following updates:

1. **Update "Last updated"** to today's date and week number.

2. **Promote concepts to "What is solid"** if quiz answers or exercises this session demonstrated clear understanding — correct answer + complete reasoning (score 1.0), or correct code written without prompting.

3. **Remove gaps that are now closed** — if a gap listed under "Known gaps" was addressed this session (correct answer on a previously missed concept, or demonstrated in an exercise), move it to solid or remove it.

4. **Add new gaps** surfaced this session — any concept where the student scored 0.0 or 0.5, showed confusion, or asked a clarifying question that revealed a missing foundation. Note *what specifically* was missing, not just the topic.

5. **Update "Curriculum position"** — mark the just-completed week as Complete and update the "Next" row.

6. **Update "Coming into Week N+1"** — summarise what the student is carrying into the next week: what's solid, what's a gap, and any specific concept from carry-forward that will be relevant.

7. **Update "Recommendations"** — revise or add recommendations based on patterns observed this session. Remove recommendations that are no longer relevant.

Do not append dated log entries — `progress.md` handles the log. `memory.md` should always read as a current snapshot, not a history.

---

## Tone

Encouraging. Point out what was done well before noting gaps. The carry-forward section is framed as "things to revisit" not "things you got wrong."
