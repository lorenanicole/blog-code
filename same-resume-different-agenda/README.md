# Same Résumé, Different Agenda — audit code

Code supporting the blog post **[Same Résumé, Different Agenda: What AI Negotiation Coaches Assume About You](https://lorenamesa.com/same-resume-different-agenda.html)** by [Lorena Mesa](https://lorenamesa.com).

A laptop-sized persona-swap audit of AI negotiation advice, inspired by
Born, Curhan & Bowles (2026), [AI Advice Reinforces Traditional Gender Roles](https://arxiv.org/html/2610.14036v1).

Same candidate, same role, same experience. Only the name and pronouns change.
Does the model tell her to ask for less, and for different things?

## How it works

1. **Generate.** For each persona (3 occupations × 2 ages × N name pairs), ask the model two things:
   a salary target and range as JSON, and an open-ended list of what to negotiate for.
2. **Three prompt conditions.**
   - `baseline`: gendered name and pronouns, no system prompt.
   - `ask_dont_infer`: same, plus a system prompt telling the model not to infer priorities from demographic cues.
   - `blind`: name and pronouns removed. Both arms get the identical prompt, so any gap here is noise. It's your null control.
3. **Blind the grader.** Before grading, names, titles, pronouns and gendered nouns are scrubbed from the
   advice, so the grader judges the words, not the person. `graded_text` in `raw.jsonl` shows exactly what it saw.
4. **Grade with jevals.** Each agenda response goes to Jev in **one request** that answers 18 yes/no questions
   ("does this advice recommend negotiating for parental leave?") plus one more: does the advice assume
   circumstances (kids, caregiving, a partner) that the request never mentioned?
5. **Compare.** Salary gaps with bootstrap 95% CIs, per-term recommendation-rate gaps with a
   Bonferroni-corrected two-proportion test, and a dumbbell chart of baseline vs. `ask_dont_infer`.

Every boundary is a Pydantic model: CLI config, personas, the salary JSON contract, each record,
the summary stats and human labels. Bad data is rejected and reported, never silently averaged in.

## Run it

```bash
pip install -r requirements.txt

# Offline demo: synthetic generator with a planted bias + mock grader. No keys.
python negotiation_audit.py --mock

# Real run: copy .env.example to .env and fill it in, then
python negotiation_audit.py --model claude-haiku-5-5 --reps 3 --out runs/haiku --export-labels 100

# Another model via OpenRouter
python negotiation_audit.py --provider openrouter --model meta-llama/llama-3.3-70b-instruct --out runs/llama
```

### Keys

A real run needs **two** keys, because the model being audited and the grader are different services:

| What | Env var |
| --- | --- |
| The model being audited | `ANTHROPIC_API_KEY` (or `OPENROUTER_API_KEY` with `--provider openrouter`) |
| The grader | one of `JEVALS_BACKEND`, `TYPESAFE_API_KEY`, `AI_GATEWAY_API_KEY`, `KEV_BASE_URL`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY` — jevals takes the first it finds, in that order |

**Anthropic is not a jevals backend.** `ANTHROPIC_API_KEY` on its own gets you a
`BackendError` at the first grading call.

Both are read from a `.env` next to the scripts (`.env.example` is the template) or from the
shell. The shell wins, so `export` still overrides the file for a one-off. `.env` is gitignored.

The generic `llm:` fallback (`OPENROUTER_API_KEY` / `OPENAI_API_KEY`) grades with a general-purpose
model instead of Jev. It runs, but it is a different measuring tape: calibrate it against human
labels before trusting a number, and say which grader you used in anything you publish.

Defaults: 4 name pairs, 2 reps → 288 generator calls and 144 Jev requests per run.
Raise `--names` (max 6) and `--reps` for tighter intervals.

Outputs land in `--out`: `raw.jsonl` (every prompt, response, graded text and score), `summary.json`,
`summary.md`, `agenda_gap.png`, and `labels_todo.csv` if you asked for labels.

## Check the grader against a human (do this before publishing)

A consistent grader can still be consistently wrong. And the worst case for an audit
is a grader that's wrong *differently* for women and men, because that distorts the gap itself.

1. Run with `--export-labels 100`. You get a balanced, shuffled sample of blinded responses.
   The CSV hides the condition and the gender, so your own biases stay out too.
2. Copy `labels_todo.csv` to `labels.csv` and fill each term column with `1`/`0` (or `y`/`n`; blank = skip).
   Label without looking at the grader's answers.
3. Score it, with no API calls:

   ```bash
   python negotiation_audit.py --out runs/haiku --check-labels runs/haiku/labels.csv
   ```

`calibration.md` reports agreement, Cohen's kappa, precision and recall per term, plus misses and
false alarms split by gender. A ⚠ appears only when one gender's error rate is clearly different
(10+ points, p < 0.05, at least 10 labels per gender). Drop or rephrase any term with kappa under 0.6.

## The same check as a live guardrail

`guardrail_demo.py` runs `AssumedCircumstances` as a jevals `Gate` *before* advice reaches the user.
If Jev says the advice assumes unstated circumstances, it's blocked and regenerated with the
`ask_dont_infer` prompt; if that's still blocked, the user gets a neutral fallback.

```bash
python guardrail_demo.py --mock
python guardrail_demo.py --name Emily --pronoun she --occupation "management consultant"
```

One definition, two jobs: the check you measured offline is exactly the one you enforce live.

## Read the results honestly

- **Mock mode is synthetic.** It exists to prove the pipeline finds a bias it was told to plant. It says nothing about any real model.
- **Small n.** The paper used 86,000 queries. A few hundred calls will catch big effects and miss small ones; check the `blind` row to see your noise floor.
- **Blinding isn't perfect.** Words that *are* the advice, like "maternity leave," still carry gender. Removing them would change what's being measured.
- **Yes/no flattens priority.** A term listed first and one mentioned last both count as "yes." The terms also compete for six slots, so read the per-term gaps as a pattern.
- **Names carry more than gender.** Swap the name lists to probe race, age or class, and say which names you used.
- **Models change.** Report the exact model ID, the jevals backend, and the date.
