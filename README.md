# CausalAgentBench

A verifiable causal-reasoning benchmark and RL environment for tool-using agents.

Data is generated from a **hidden linear-Gaussian structural causal model (SCM)**, so the SCM is the
exact answer key. An agent must rank the drivers of an outcome by causal effect, choose valid adjustment
sets, and (in later stages) abstain when an effect is not identifiable.

**Stage 0** (this repo's current scope) is the generator foundation: a single canonical 6-node scaffold
with an engineered confounder, a mediator (bad-control trap), and a wired-zero decoy; exact effects via
`τ = (I − W)⁻¹ − I`; UUMC weight sampling for neutral sortability; and a reject loop whose five soundness
gates guarantee every instance requires genuine causal reasoning (naive correlation *and*
"adjust-for-everything" both provably fail). See [`stage0_plan.md`](stage0_plan.md) for the full spec.

## Quickstart

```bash
uv sync                      # create the environment
uv run pytest                # run the soundness-gate test suite
uv run causalagentbench-smoke   # generate gated instances and print acceptance/gate stats
```

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src              # strict type check
```

## Layout

```
src/causalagentbench/
  config.py        GeneratorConfig (validated knobs)
  types.py         Role, SCM (hidden answer key), GateReport
  scaffold.py      the canonical 6-node role graph
  effects.py       total-effect matrix τ and marginal covariance Σ
  generate.py      UUMC structural generation (unit-variance, neutral sortability)
  sample.py        draw a standardized data table from an SCM
  sortability.py   varsortability / R²-sortability metrics
  baselines.py     oracle / correlation / full-adjustment reference solvers
  gates.py         the five soundness gates + rejection-sampling generator
  smoke.py         acceptance-rate / gate-statistics report
```

License: Apache-2.0.
