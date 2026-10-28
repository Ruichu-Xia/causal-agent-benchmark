"""Task families over random causal worlds: each draws a world, asks one question, and computes its answer.

- ``independence``: are two variables associated given a conditioning set? (d-separation)
- ``adjustment``: does adjusting for a set identify the causal effect? (the adjustment criterion)
- ``effect``: causal or observed effect from stated probabilities. (truncated factorization, Bayes' rule)
- ``discovery``: what must or might be true of the structure given all its independences? (equivalence class)

Every family varies its wording, graph description and answer format, and balances yes and no answers.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from causalagentbench.reasoning.worlds import (
    World,
    causal_effect,
    d_separated,
    observed_difference,
    random_dag,
    random_world,
    valid_adjustment,
)
from causalagentbench.scm.identifiability import equivalence_class
from causalagentbench.types import Role, Scaffold

Namer = Callable[[np.random.Generator, int], tuple[str, ...]]
"""Draws ``n`` distinct variable names."""

FORMATS = {
    "yesno": "Yes or No",
    "number": "a number rounded to two decimal places",
    "letters": "the letters of every correct option separated by commas, or None",
    "choice": "1, 2, Both or Neither",
}
MAX_TRIES = 200


@dataclass(frozen=True)
class Task:
    """One question with its exact answer.

    Attributes:
        family: The task family.
        kind: The answer format, a key of ``FORMATS``.
        prompt: The full user message.
        answer: The canonical answer: yes/no, a two-decimal number, sorted letters or none, or a choice.
        meta: Generation details for analysis (graph, variant, query).
    """

    family: str
    kind: str
    prompt: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


def _join(words: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _prompt(context: str, question: str, kind: str) -> str:
    """Context, question and the final-line instruction for the answer format."""
    instruction = (
        f'Reason step by step, then end with a final line "Answer: <answer>", where <answer> is {FORMATS[kind]}.'
    )
    return f"{context}\n\n{question}\n\n{instruction}"


def describe_graph(names: tuple[str, ...], edges: Sequence[tuple[int, int]], rng: np.random.Generator) -> str:
    """The causal structure in one of four styles, statements in random order, closed-world assumption stated."""
    style = int(rng.integers(4))
    order = list(rng.permutation(len(names)))
    if style == 0:
        lines = [f"{names[p].capitalize()} directly causes {names[c]}." for p, c in edges]
    elif style == 1:
        lines = [f"{names[c].capitalize()} is directly affected by {names[p]}." for p, c in edges]
    elif style == 2:
        lines = []
        for c in order:
            parents = [names[p] for p, child in edges if child == c]
            if parents:
                lines.append(f"{names[c].capitalize()} depends directly on {_join(parents)}.")
    else:
        lines = [f"{names[p]} -> {names[c]}" for p, c in edges]
    if style != 2:
        lines = [lines[i] for i in rng.permutation(len(lines))]
    listed = _join([names[i] for i in order])
    header = f"A closed system has {len(names)} variables: {listed}. Its causal structure is:"
    closing = "There are no other causal relationships and no hidden common causes."
    separator = "\n" if style == 3 else " "
    return f"{header}{separator}{separator.join(lines)}{separator}{closing}"


def _subset_text(names: tuple[str, ...], z: Sequence[int]) -> str:
    """'nothing' or the joined names of a set."""
    return _join([names[i] for i in sorted(z)]) if z else "nothing"


def _balanced(rng: np.random.Generator, draw: Callable[[], tuple[bool, Any] | None]) -> Any:
    """Call ``draw`` until it returns a case whose label equals a fair coin flip; return that case."""
    target = bool(rng.random() < 0.5)
    for _ in range(MAX_TRIES):
        result = draw()
        if result is not None and result[0] == target:
            return result
    raise RuntimeError("could not draw a case with the target label")


def independence(rng: np.random.Generator, namer: Namer) -> Task:
    """Are two variables associated within strata of a conditioning set?"""

    def draw() -> tuple[bool, Any]:
        """A random graph, pair and conditioning set; the label is d-connection."""
        n = int(rng.integers(3, 7))
        edges = random_dag(rng, n, float(rng.uniform(0.3, 0.6)))
        x, y = (int(v) for v in rng.choice(n, size=2, replace=False))
        others = [v for v in range(n) if v not in (x, y)]
        size = int(rng.integers(0, min(3, len(others)) + 1))
        z = {int(v) for v in rng.choice(others, size=size, replace=False)}
        graph = nx.DiGraph(edges)
        graph.add_nodes_from(range(n))
        connected = not d_separated(graph, x, y, z)
        return connected, (n, edges, x, y, z, connected)

    _, (n, edges, x, y, z, connected) = _balanced(rng, draw)
    names = namer(rng, n)
    context = describe_graph(names, edges, rng) + " Every causal relationship is real, with no exact cancellations."
    where = (
        "in the population as a whole"
        if not z
        else rng.choice(
            [
                f"among units that share the same values of {_subset_text(names, z)}",
                f"after conditioning on {_subset_text(names, z)}",
                f"within groups defined by {_subset_text(names, z)}",
            ]
        )
    )
    ask_independent = bool(rng.random() < 0.5)
    if ask_independent:
        question = f"Are {names[x]} and {names[y]} statistically independent {where}?"
        answer = not connected
    else:
        question = f"Would {names[x]} and {names[y]} be statistically associated {where}?"
        answer = connected
    meta = {
        "edges": edges,
        "x": x,
        "y": y,
        "z": sorted(z),
        "variant": "independent" if ask_independent else "associated",
    }
    return Task("independence", "yesno", _prompt(context, question, "yesno"), "yes" if answer else "no", meta)


def _effect_pair(rng: np.random.Generator, n: int) -> tuple[int, int]:
    """A random pair ``x < y``, so ``y`` never causes ``x``."""
    x, y = sorted(int(v) for v in rng.choice(n, size=2, replace=False))
    return x, y


def adjustment(rng: np.random.Generator, namer: Namer) -> Task:
    """Does adjusting for a set identify the causal effect of one variable on another?"""
    variant = str(rng.choice(["single", "options", "compare"]))
    n = int(rng.integers(4, 7))
    edges = random_dag(rng, n, float(rng.uniform(0.3, 0.6)))
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(range(n))
    x, y = _effect_pair(rng, n)
    others = [v for v in range(n) if v not in (x, y)]
    subsets = [set(c) for k in range(len(others) + 1) for c in itertools.combinations(others, k)]
    valid = [valid_adjustment(graph, x, y, z) for z in subsets]

    def pick(label: bool) -> set[int] | None:
        """A random subset with the given validity, if any."""
        matching = [z for z, v in zip(subsets, valid, strict=True) if v == label]
        return matching[int(rng.integers(len(matching)))] if matching else None

    if variant == "single":
        target = bool(rng.random() < 0.5)
        z = pick(target)
        if z is None:
            target, z = not target, pick(not target)
        assert z is not None
        question = (
            "Researchers want the causal effect of {x} on {y} from observational data. They compare units with "
            "different values of {x} {within}. Does this comparison give an unbiased estimate of the causal effect?"
        )
        kind, answer, chosen = "yesno", "yes" if target else "no", [sorted(z)]
    elif variant == "options":
        k = min(len(subsets), int(rng.integers(3, 5)))
        chosen_sets = [subsets[i] for i in rng.choice(len(subsets), size=k, replace=False)]
        letters = "ABCD"[:k]
        right = [letter for letter, z in zip(letters, chosen_sets, strict=True) if valid_adjustment(graph, x, y, z)]
        question = (
            "Researchers want the causal effect of {x} on {y} from observational data by adjusting for a set of "
            "variables. Which of these adjustment sets give an unbiased estimate?"
        )
        kind, answer, chosen = "letters", ",".join(right) if right else "none", [sorted(z) for z in chosen_sets]
    else:
        z1, z2 = (subsets[i] for i in rng.choice(len(subsets), size=2, replace=False))
        v1, v2 = valid_adjustment(graph, x, y, z1), valid_adjustment(graph, x, y, z2)
        question = (
            "Two analysts estimate the causal effect of {x} on {y} from observational data. Analyst 1 adjusts for "
            "{z1}; analyst 2 adjusts for {z2}. Whose estimate is unbiased: 1, 2, Both or Neither?"
        )
        choice = {(True, True): "both", (True, False): "1", (False, True): "2", (False, False): "neither"}[(v1, v2)]
        kind, answer, chosen = "choice", choice, [sorted(z1), sorted(z2)]
    names = namer(rng, n)
    fill = {"x": names[x], "y": names[y]}
    if variant == "single":
        z_text = _subset_text(names, chosen[0])
        fill["within"] = (
            f"within groups that share the same values of {z_text}" if chosen[0] else "directly, adjusting for nothing"
        )
    elif variant == "compare":
        fill["z1"], fill["z2"] = _subset_text(names, chosen[0]), _subset_text(names, chosen[1])
    question = question.format(**fill)
    if variant == "options":
        question += "\n" + "\n".join(f"{'ABCD'[i]}. {_subset_text(names, z)}" for i, z in enumerate(chosen))
    meta = {"edges": edges, "x": x, "y": y, "sets": chosen, "variant": variant}
    return Task("adjustment", kind, _prompt(describe_graph(names, edges, rng), question, kind), answer, meta)


def describe_probabilities(world: World, rng: np.random.Generator) -> str:
    """Every conditional probability table, one sentence per node, nodes in random order."""
    names = world.names
    sentences = []
    for node in rng.permutation(len(names)):
        parents = world.parents(int(node))
        parts = []
        for k, p in enumerate(world.cpts[node]):
            given = ", ".join(f"{names[q]}={(k >> j) & 1}" for j, q in enumerate(parents))
            parts.append(f"P({names[node]}=1 | {given}) = {p:.2f}" if parents else f"P({names[node]}=1) = {p:.2f}")
        sentences.append("; ".join(parts) + ".")
    return " ".join(sentences)


def effect(rng: np.random.Generator, namer: Namer) -> Task:
    """Causal or observed effect of one binary variable on another, from the stated mechanism."""
    variant = str(rng.choice(["causal", "observed", "sign", "compare"]))
    for _ in range(MAX_TRIES):
        # Comparing both effects takes the longest replies; 4-variable worlds overflowed 4,096 tokens.
        n = 3 if variant == "compare" else int(rng.integers(3, 5))
        world = random_world(rng, tuple(str(i) for i in range(n)), float(rng.uniform(0.4, 0.8)))
        x, y = _effect_pair(rng, n)
        ate, observed = causal_effect(world, x, y), observed_difference(world, x, y)
        if variant == "sign" and abs(ate) < 0.03:
            continue
        if variant == "compare" and abs(observed - ate) < 0.03:
            continue
        break
    else:
        raise RuntimeError("no world met the margin")
    names = namer(rng, n)
    world = World(names, world.edges, world.cpts)
    context = (
        describe_graph(names, world.edges, rng)
        + " Every variable is binary (1 = present, 0 = absent). The causal mechanisms are: "
        + describe_probabilities(world, rng)
    )
    nx_, ny = names[x], names[y]
    if variant == "causal":
        question = (
            f"If {nx_} were set to 1 for everyone, rather than to 0 for everyone, by how much would the probability "
            f"that {ny}=1 change? Give P({ny}=1 | do({nx_}=1)) - P({ny}=1 | do({nx_}=0))."
        )
        kind, answer = "number", f"{ate:.2f}"
    elif variant == "observed":
        question = f"What is P({ny}=1 | {nx_}=1) - P({ny}=1 | {nx_}=0) in this population?"
        kind, answer = "number", f"{observed:.2f}"
    elif variant == "sign":
        question = f"Would setting {nx_} to 1 for everyone, rather than to 0, make {ny}=1 more likely?"
        kind, answer = "yesno", "yes" if ate > 0 else "no"
    else:
        question = (
            f"Is the observed difference P({ny}=1 | {nx_}=1) - P({ny}=1 | {nx_}=0) larger than the causal effect "
            f"P({ny}=1 | do({nx_}=1)) - P({ny}=1 | do({nx_}=0))?"
        )
        kind, answer = "yesno", "yes" if observed > ate else "no"
    meta = {"edges": world.edges, "x": x, "y": y, "ate": ate, "observed": observed, "variant": variant}
    return Task("effect", kind, _prompt(context, question, kind), answer, meta)


def _relations(edges: Sequence[tuple[str, str]], nodes: Sequence[str]) -> dict[tuple[str, str, str], bool]:
    """Every structural statement about one DAG: direct cause, cause, common cause, common effect."""
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(nodes)
    ancestors = {v: nx.ancestors(graph, v) for v in nodes}
    facts = {}
    for a, b in itertools.permutations(nodes, 2):
        facts[("direct", a, b)] = graph.has_edge(a, b)
        facts[("cause", a, b)] = a in ancestors[b]
        facts[("common_cause", a, b)] = bool(ancestors[a] & ancestors[b] - {a, b})
        facts[("common_effect", a, b)] = any(graph.has_edge(a, c) and graph.has_edge(b, c) for c in nodes)
    return facts


HYPOTHESES = {
    "direct": "{a} directly causes {b}",
    "cause": "{a} causes {b}, directly or through other variables",
    "common_cause": "some other variable in the system causes both {a} and {b}",
    "common_effect": "{a} and {b} both directly cause some third variable",
}


def describe_independences(names: tuple[str, ...], graph: nx.DiGraph) -> str:
    """Every (conditional) independence among the variables; all other pairs are dependent given any set."""
    n = len(names)
    lines = []
    for a, b in itertools.combinations(range(n), 2):
        others = [v for v in range(n) if v not in (a, b)]
        for k in range(len(others) + 1):
            for z in itertools.combinations(others, k):
                if d_separated(graph, a, b, set(z)):
                    given = f" given {_join([names[v] for v in z])}" if z else ""
                    lines.append(f"{names[a]} and {names[b]} are independent{given}.")
    if not lines:
        return "No independences hold: every pair of variables is dependent given any set of the others."
    rest = "Every other pair is dependent given any other set of variables."
    return "The independences that hold are exactly these: " + " ".join(lines) + " " + rest


def discovery(rng: np.random.Generator, namer: Namer) -> Task:
    """What must, or could, be true of the causal structure given all of its independences?"""

    def draw() -> tuple[bool, Any]:
        """A random graph and statement; the label is whether it holds in every (or some) equivalent DAG."""
        n = int(rng.integers(3, 5))  # with 5 variables most base replies overflowed 4,096 tokens
        edges = random_dag(rng, n, float(rng.uniform(0.3, 0.6)))
        nodes = tuple(str(i) for i in range(n))
        scaffold = Scaffold(
            nodes, dict.fromkeys(nodes, Role.CONFOUNDER), tuple((str(p), str(c)) for p, c in edges), (), nodes[0]
        )
        members = [_relations(member, nodes) for member in equivalence_class(scaffold)]
        relation = str(rng.choice(list(HYPOTHESES)))
        a, b = (str(v) for v in rng.choice(nodes, size=2, replace=False))
        mode = str(rng.choice(["must", "could"]))
        holds = [facts[(relation, a, b)] for facts in members]
        label = all(holds) if mode == "must" else any(holds)
        return label, (n, edges, relation, int(a), int(b), mode, label, len(members))

    _, (n, edges, relation, a, b, mode, label, size) = _balanced(rng, draw)
    names = namer(rng, n)
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(range(n))
    context = (
        f"A study measured {n} variables: {_join([names[i] for i in rng.permutation(n)])}. They form a closed causal "
        "system with no hidden common causes, and every dependence among them reflects the causal structure "
        "(no exact cancellations). " + describe_independences(names, graph)
    )
    statement = HYPOTHESES[relation].format(a=names[a], b=names[b])
    question = f"Must it be true that {statement}?" if mode == "must" else f"Is it possible that {statement}?"
    meta = {"edges": edges, "relation": relation, "a": a, "b": b, "variant": mode, "class_size": size}
    return Task("discovery", "yesno", _prompt(context, question, "yesno"), "yes" if label else "no", meta)


FAMILIES: dict[str, Callable[[np.random.Generator, Namer], Task]] = {
    "independence": independence,
    "adjustment": adjustment,
    "effect": effect,
    "discovery": discovery,
}
