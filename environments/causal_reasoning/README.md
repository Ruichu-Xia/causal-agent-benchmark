# causal-reasoning

Single-turn causal reasoning over small random causal systems, with exact answers. A Verifiers (v1)
taskset built on `causalagentbench.reasoning`.

Each task describes a closed causal system of 3–8 variables in words and asks one question:

| Family | Question | Answer | Verifier |
| --- | --- | --- | --- |
| independence | Are X and Y dependent given Z? | yes/no | d-separation |
| adjustment | Does adjusting for Z recover the causal effect of X on Y? | yes/no, or which of two analysts | complete adjustment criterion (Perković et al.) |
| effect | P(y \| do(x)) contrasts, observed contrasts, signs | number (±0.011) or yes/no | exact enumeration |
| discovery | Given all independences, must/could X cause Y (or share a cause, ...)? | yes/no | Markov equivalence class |

The model answers in plain text, with no tools, and ends with `Answer: <answer>`. The reward `correct`
is 1.0 for a right final answer and 0.0 otherwise. The metrics `answered` and `truncated` record
whether an answer line was given and whether the reply hit a limit.

## Splits

| `split` | What changes from `iid` |
| --- | --- |
| `train`, `iid` | nothing (use different seeds) |
| `ood_wording` | every phrase comes from held-out templates |
| `ood_size` | larger graphs |
| `ood_vocab` | variable names from held-out concepts |

With one `seed`, `iid`, `ood_wording` and `ood_vocab` pose the same problems item by item, so their
differences measure wording and vocabulary alone.

Labels are balanced, and each family's most plausible shortcut (e.g. "adjust for every common cause
and nothing downstream of the treatment") scores at chance by construction. `scripts/audit_reasoning.py`
in the parent repository re-derives every label and measures every shortcut. See
`docs/environment_v2.md` for the design.

## Usage

```bash
uv run vf-eval causal-reasoning -m <model> -n 100 --env.taskset.split ood_size --env.taskset.seed 200
```

Taskset options: `split`, `seed`, `num_tasks`, `families`.

Variable names include concepts from CauseNet-Precision (Heindorf et al., CIKM 2020; CC BY 4.0). In
these tasks they are names only: the causal relations among them are random.
