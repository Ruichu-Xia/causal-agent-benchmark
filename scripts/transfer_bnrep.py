r"""Near-transfer tasks: published Gaussian Bayesian networks from bnRep, cast as CausalAgentBench tasks.

    python scripts/transfer_bnrep.py --per-graph 50 --out outputs/transfer/bnrep
    causalagentbench eval --dataset outputs/transfer/bnrep --model hosted_vllm/trained ...

Each network in ``data/bnrep`` (exported by ``scripts/bnrep_export.R``) is a linear-Gaussian SCM with
published coefficients and the authors' variable names. Standardizing it gives the correlation matrix that
``realize`` turns back into the same graph's standardized weights, so tasks use the benchmark's exact
effects, scorer and prompt; only the graph, coefficients and names come from outside the generator. A task
is an outcome and five other nodes as candidates, every node is a column, and none of the generator's
gates apply except that every true effect is below 1 and at least two candidates have an effect of 0.1
or more.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from causalagentbench.benchmark import N_ROWS
from causalagentbench.scm.effects import marginal_covariance
from causalagentbench.scm.generate import realize
from causalagentbench.scm.sample import sample
from causalagentbench.task.dataset import save_dataset
from causalagentbench.task.instance import Instance, InstanceKey, build_instance
from causalagentbench.task.labeling import Labeler, Labeling
from causalagentbench.types import SCM, Role, Scaffold

ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ("expenditure", "lexical", "algal2", "liquefaction")
CANDIDATES = 5


def correlation(network: dict) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], np.ndarray]:
    """Node names, edges and the exact correlation matrix of one exported network."""
    nodes = tuple(entry["node"] for entry in network["nodes"])
    index = {name: i for i, name in enumerate(nodes)}
    weights = np.zeros((len(nodes), len(nodes)))
    noise_var = np.array([entry["sd"] ** 2 for entry in network["nodes"]])
    edges = []
    for entry in network["nodes"]:
        for parent, coefficient in entry["parents"].items():
            weights[index[parent], index[entry["node"]]] = coefficient
            edges.append((parent, entry["node"]))
    sigma = marginal_covariance(weights, noise_var)
    scale = np.sqrt(np.diag(sigma))
    return nodes, tuple(edges), sigma / np.outer(scale, scale)


def scms(name: str) -> list[SCM]:
    """Every (outcome, five candidates) choice on one network that passes the effect-size filter."""
    network = json.loads((ROOT / "data" / "bnrep" / f"{name}.json").read_text(encoding="utf-8"))
    nodes, edges, sigma = correlation(network)
    kept = []
    for outcome in nodes:
        others = [node for node in nodes if node != outcome]
        for drivers in itertools.combinations(others, CANDIDATES):
            # Roles are nominal here: external graphs have no motif positions.
            roles = dict.fromkeys(nodes, Role.CONFOUNDER) | dict.fromkeys(drivers, Role.DRIVER_CLEAN)
            scaffold = Scaffold(nodes, roles | {outcome: Role.OUTCOME}, edges, drivers, outcome)
            scm = realize(scaffold, sigma)
            effects = np.array([abs(scm.total_effect_on_outcome(d)) for d in drivers])
            if effects.max() < 1 and (effects >= 0.1).sum() >= 2:
                kept.append(scm)
    return kept


def own_names(name: str) -> Labeler:
    """Label every node with the network authors' own variable name."""

    def labeler(scm: SCM, _rng: np.random.Generator) -> Labeling:
        """Identity labeling."""
        return Labeling({node: node for node in scm.scaffold.node_names}, producer=f"bnrep:{name}")

    return labeler


def build(per_graph: int, seed: int) -> list[tuple[Instance, InstanceKey]]:
    """``per_graph`` tasks from each network, chosen at random among those that pass the filter."""
    tasks = []
    for g, name in enumerate(GRAPHS):
        candidates = scms(name)
        rng = np.random.default_rng([seed, g])
        chosen = rng.choice(len(candidates), size=min(per_graph, len(candidates)), replace=False)
        for i, child in zip(sorted(chosen), np.random.SeedSequence([seed, g]).spawn(len(chosen)), strict=True):
            data_seed, label_seed = child.spawn(2)
            data = sample(candidates[i], N_ROWS, data_seed).astype(np.float32).astype(np.float64)
            tasks.append(build_instance(candidates[i], data, label_seed, labeler=own_names(name)))
        print(f"{name}: {len(chosen)} of {len(candidates)} eligible tasks")
    return tasks


def main() -> None:
    """Build and save the near-transfer dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-graph", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/transfer/bnrep")
    args = parser.parse_args()
    save_dataset(args.out, build(args.per_graph, args.seed), split="bnrep")


if __name__ == "__main__":
    main()
