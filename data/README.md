# Example dataset

`example.json` and `example.npz` contain ten tasks, each with five candidates and 2,000 observations.
They are ready for evaluator use without downloading CauseNet or generating names with a model.
The JSON includes hidden keys and must not be exposed to agents.

The prompts use signed total-effect estimation as the objective, with `normalized-error-v1` evaluation
reporting and a separate `effect-v1` RL reward. Prompts were refreshed as these metrics changed. Graphs, names,
coefficients and sampled observations were preserved, so the statistical references use the same tasks.

Names are exact concepts from **CauseNet-Precision**, by Stefan Heindorf, Yan Scholten, Henning Wachsmuth,
Axel-Cyrille Ngonga Ngomo, and Martin Potthast, *CauseNet: Towards a Causality Graph Extracted from the Web*,
CIKM 2020. Source and attribution: <https://causenet.org/>.

Every simulated edge matches a directed relation in that dump. We select a subgraph and generate
synthetic linear-Gaussian coefficients and observations; source relation frequencies are not used as
causal effect sizes. Additional source edges can be omitted. The example includes automatically selected
concepts and has not undergone human domain curation.

Source dump SHA-256:

```
357b78317cd62960a4bffc616d414f859f997d4976a4af4d6c48d538a2ea3a13
```

Build command (package 0.1.0, Python 3.12.10, locked dependencies):

```bash
uv run causalagentbench build --causenet /path/to/causenet-precision.jsonl.bz2 \
  --out data/example --seed 0 -n 10
```

The dataset files are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
following the source data's license. Code in the repository is licensed separately under Apache-2.0.
