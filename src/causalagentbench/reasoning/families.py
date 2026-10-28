"""Task families over random causal worlds: each draws a world, asks one question, and computes its answer.

- ``independence``: are two variables associated given a conditioning set? (d-separation)
- ``adjustment``: does adjusting for a set identify the causal effect? (the adjustment criterion)
- ``effect``: causal or observed effect from stated probabilities. (truncated factorization, Bayes' rule)
- ``discovery``: what must or might be true of the structure given all its independences? (equivalence class)

Two random streams drive a task. ``rng`` draws the problem (graph, query, answer); ``Surface`` draws how it
reads (templates from ``wording.SLOTS``, statement order, variable names). Answers depend on the problem
alone, so the same problem seed gives the same problem under any wording or vocabulary. Yes/no labels
are balanced, and where a family has a plausible shortcut (``shortcuts.py``) the shortcut's agreement
with the label is drawn independently of the label (``Spec.agree``).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from causalagentbench.reasoning.grading import grade
from causalagentbench.reasoning.shortcuts import confounder_rule, undirected_separated
from causalagentbench.reasoning.wording import GRAPH_STYLES, SLOTS
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
    "choice": "1, 2, Both or Neither",
}
MAX_TRIES = 100_000
"""Rejection-sampling budget. The rarest target is two invalid sets the confounder rule accepts (M-bias-like),
present in 0.4% of training-size graphs: about 250 tries on average, and failure odds about e^-400."""
MARGIN = 0.03
"""Minimum size of an effect, or of a gap between effects, whose sign or order is asked about."""
P_NONZERO = 0.9
"""Share of numeric effect questions whose answer is not 0.00, so answering zero is not a shortcut."""

SIZES = {
    "independence": ((3, 6), (7, 8)),
    "adjustment": ((4, 6), (7, 8)),
    "effect": ((3, 4), (5, 5)),
    "discovery": ((3, 4), (5, 5)),
}
"""Per family: (training, ood_size) inclusive ranges of the number of variables."""

RELATIONS = ("direct", "cause", "common_cause", "common_effect")


@dataclass(frozen=True)
class Spec:
    """How a split draws problems.

    Attributes:
        large: Draw graph sizes from the ood_size range instead of the training range.
        agree: Probability that the family's shortcut gives the right answer.
    """

    large: bool = False
    agree: float = 0.5


@dataclass
class Surface:
    """Random draws that change only how a task reads: templates, statement order and variable names.

    Attributes:
        rng: The surface stream, separate from the problem stream.
        namer: Draws variable names from this stream.
        held_out: Use held-out templates instead of training ones.
    """

    rng: np.random.Generator
    namer: Namer
    held_out: bool = False

    def pick(self, options: tuple[tuple[Any, ...], tuple[Any, ...]]) -> Any:
        """A random option from the training or held-out pool."""
        pool = options[1] if self.held_out else options[0]
        return pool[int(self.rng.integers(len(pool)))]

    def phrase(self, slot: str) -> str:
        """A random template of ``slot``."""
        return str(self.pick(SLOTS[slot]))

    def names(self, n: int) -> tuple[str, ...]:
        """``n`` distinct variable names."""
        return self.namer(self.rng, n)

    def order(self, n: int) -> list[int]:
        """A random order of ``n`` items."""
        return [int(i) for i in self.rng.permutation(n)]


@dataclass(frozen=True)
class Task:
    """One question with its exact answer.

    Attributes:
        family: The task family.
        kind: The answer format, a key of ``FORMATS``.
        prompt: The full user message.
        answer: The canonical answer: yes/no, a two-decimal number, or a choice.
        meta: The problem (graph, query, variable names) for analysis and audits.
    """

    family: str
    kind: str
    prompt: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


Case = tuple[bool | None, bool | None, Any]
"""(label, shortcut agrees, payload) of one drawn case; None means "not matched"."""


def _join(words: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _prompt(context: str, question: str, kind: str) -> str:
    """Context, question and the final-line instruction for the answer format."""
    instruction = (
        f'Reason step by step, then end with a final line "Answer: <answer>", where <answer> is {FORMATS[kind]}.'
    )
    return f"{context}\n\n{question}\n\n{instruction}"


def _size(rng: np.random.Generator, family: str, spec: Spec) -> int:
    """A graph size from the family's range for this split."""
    lo, hi = SIZES[family][spec.large]
    return int(rng.integers(lo, hi + 1))


def _graph(edges: Sequence[tuple[int, int]], n: int) -> nx.DiGraph:
    """A DAG on nodes 0..n-1."""
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(range(n))
    return graph


def _draw(rng: np.random.Generator, draw: Callable[[], Case | None], agree: float, p_label: float = 0.5) -> Any:
    """Draw cases until one falls in a random cell; return its payload.

    ``draw`` returns a ``Case`` or None (rejected). The cell's label is true with probability
    ``p_label`` and its shortcut agreement is true with probability ``agree``.
    """
    label, agrees = bool(rng.random() < p_label), bool(rng.random() < agree)
    for _ in range(MAX_TRIES):
        case = draw()
        if case is not None and case[0] in (None, label) and case[1] in (None, agrees):
            return case[2]
    raise RuntimeError("could not draw a case in the target cell")


def describe_graph(names: tuple[str, ...], edges: Sequence[tuple[int, int]], surface: Surface) -> str:
    """The causal structure in one style, statements in random order, closed-world assumption stated."""
    kind, template = surface.pick(GRAPH_STYLES)
    order = surface.order(len(names))
    if kind in ("edge", "arrows"):
        lines = [
            template.format(P=names[p].capitalize(), p=names[p], C=names[c].capitalize(), c=names[c])
            for p, c in edges
        ]
        lines = [lines[i] for i in surface.order(len(lines))]
    elif kind == "parents":
        lines = [
            template.format(C=names[c].capitalize(), ps=_join([names[p] for p, child in edges if child == c]))
            for c in order
            if any(child == c for _, child in edges)
        ]
    else:
        lines = [
            template.format(P=names[p].capitalize(), cs=_join([names[c] for parent, c in edges if parent == p]))
            for p in order
            if any(parent == p for parent, _ in edges)
        ]
    header = surface.phrase("header").format(n=len(names), names=_join([names[i] for i in order]))
    separator = "\n" if kind == "arrows" else " "
    return f"{header}{separator}{separator.join(lines)}{separator}{surface.phrase('closing')}"


def _subset_text(names: tuple[str, ...], z: Sequence[int]) -> str:
    """'nothing' or the joined names of a set."""
    return _join([names[i] for i in sorted(z)]) if z else "nothing"


def independence(rng: np.random.Generator, surface: Surface, spec: Spec) -> Task:
    """Are two variables associated within strata of a conditioning set?"""

    def draw() -> Case:
        """A random graph, pair and conditioning set; the label is d-separation."""
        n = _size(rng, "independence", spec)
        edges = random_dag(rng, n, float(rng.uniform(0.3, 0.6)))
        x, y = (int(v) for v in rng.choice(n, size=2, replace=False))
        others = [v for v in range(n) if v not in (x, y)]
        z = {int(v) for v in rng.choice(others, size=int(rng.integers(0, min(3, len(others)) + 1)), replace=False)}
        graph = _graph(edges, n)
        separated = d_separated(graph, x, y, z)
        return separated, undirected_separated(graph, x, y, z) == separated, (n, edges, x, y, z, separated)

    n, edges, x, y, z, separated = _draw(rng, draw, spec.agree)
    ask_independent = bool(rng.random() < 0.5)
    answer = separated if ask_independent else not separated
    names = surface.names(n)
    context = describe_graph(names, edges, surface) + " " + surface.phrase("faithful")
    where = surface.phrase("where_given").format(z=_subset_text(names, z)) if z else surface.phrase("where_none")
    template = surface.phrase("ask_independent" if ask_independent else "ask_associated")
    question = template.format(x=names[x], y=names[y], where=where)
    meta = {
        "names": names,
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


def adjustment(rng: np.random.Generator, surface: Surface, spec: Spec) -> Task:
    """Does adjusting for a set identify the causal effect of one variable on another?

    Each queried set's validity is a fair coin and the confounder rule's agreement with it is true with
    probability ``spec.agree``, independently for each set.
    """
    variant = str(rng.choice(["single", "compare"]))
    targets = [(bool(rng.random() < 0.5), bool(rng.random() < spec.agree)) for _ in range(1 + (variant == "compare"))]
    for _ in range(MAX_TRIES):
        n = _size(rng, "adjustment", spec)
        edges = random_dag(rng, n, float(rng.uniform(0.3, 0.6)))
        graph = _graph(edges, n)
        x, y = _effect_pair(rng, n)
        others = [v for v in range(n) if v not in (x, y)]
        cells: dict[tuple[bool, bool], list[set[int]]] = {}
        for k in range(len(others) + 1):
            for c in itertools.combinations(others, k):
                ok = valid_adjustment(graph, x, y, set(c))
                cells.setdefault((ok, confounder_rule(graph, x, y, set(c)) == ok), []).append(set(c))
        if all(len(cells.get(t, [])) >= targets.count(t) for t in targets):
            break
    else:
        raise RuntimeError("no graph had sets in the target cells")
    chosen: list[set[int]] = []
    for t in targets:
        pool = [z for z in cells[t] if z not in chosen]
        chosen.append(pool[int(rng.integers(len(pool)))])
    valid = [t[0] for t in targets]
    names = surface.names(n)
    fill = {"x": names[x], "y": names[y]}

    def procedure(z: set[int], short: bool) -> str:
        """How one analysis adjusts for ``z``, in a random template."""
        slot = ("short_" if short else "proc_") + ("set" if z else "empty")
        return surface.phrase(slot).format(z=_subset_text(names, sorted(z)), **fill)

    if variant == "single":
        question = surface.phrase("ask_single").format(proc=procedure(chosen[0], False), **fill)
        kind, answer = "yesno", "yes" if valid[0] else "no"
    else:
        proc1, proc2 = procedure(chosen[0], True), procedure(chosen[1], True)
        question = surface.phrase("ask_compare").format(proc1=proc1, proc2=proc2, **fill)
        kind = "choice"
        answer = {(True, True): "both", (True, False): "1", (False, True): "2", (False, False): "neither"}[
            (valid[0], valid[1])
        ]
    meta = {"names": names, "edges": edges, "x": x, "y": y, "sets": [sorted(z) for z in chosen], "variant": variant}
    # without cancellations an invalid set is biased, so "no" means biased, not merely "not guaranteed"
    context = describe_graph(names, edges, surface) + " " + surface.phrase("faithful")
    return Task("adjustment", kind, _prompt(context, question, kind), answer, meta)


def describe_probabilities(world: World, surface: Surface) -> str:
    """Every conditional probability table, one sentence per node, nodes in random order."""
    names = world.names
    row, root = surface.phrase("cpt_row"), surface.phrase("cpt_root")
    sentences = []
    for node in surface.order(len(names)):
        parents = world.parents(node)
        template = row if parents else root
        parts = []
        for k, p in enumerate(world.cpts[node]):
            given = ", ".join(f"{names[q]}={(k >> j) & 1}" for j, q in enumerate(parents))
            parts.append(template.format(v=names[node], given=given, p=f"{p:.2f}"))
        sentence = "; ".join(parts) + "."
        # capitalize a sentence that opens with template text, never one that opens with a variable name
        sentences.append(sentence if template.startswith("{") else sentence[0].upper() + sentence[1:])
    return " ".join(sentences)


def _number(value: float) -> str:
    """Two-decimal canonical form; never "-0.00"."""
    return f"{round(value, 2) + 0.0:.2f}"


def effect(rng: np.random.Generator, surface: Surface, spec: Spec) -> Task:
    """Causal or observed effect of one binary variable on another, from the stated mechanism.

    Shortcuts: reporting the observed difference as the causal effect (``causal``), and reading the sign
    of the causal effect off the observed difference (``sign``). For numeric questions the label is
    "the answer is not 0.00", true with probability ``P_NONZERO``.
    """
    variant = str(rng.choice(["causal", "observed", "sign", "gap"]))

    def draw() -> Case | None:
        """A random world and pair, with this variant's label and shortcut agreement."""
        # Comparing both effects takes the longest replies, so it uses the smallest graphs.
        n = (4 if spec.large else 3) if variant == "gap" else _size(rng, "effect", spec)
        world = random_world(rng, tuple(str(i) for i in range(n)), float(rng.uniform(0.4, 0.8)))
        x, y = _effect_pair(rng, n)
        ate, observed = causal_effect(world, x, y), observed_difference(world, x, y)
        payload = (world, x, y, ate, observed)
        if variant == "causal":  # the shortcut is right when the grader accepts the observed difference
            shortcut_right = grade("number", _number(ate), f"Answer: {_number(observed)}") == 1.0
            return _number(ate) != "0.00", shortcut_right, payload
        if variant == "observed":
            return _number(observed) != "0.00", None, payload
        if variant == "sign":
            if min(abs(ate), abs(observed)) < MARGIN:
                return None
            return ate > 0, (observed > 0) == (ate > 0), payload
        return (observed > ate, None, payload) if abs(observed - ate) >= MARGIN else None

    numeric = variant in ("causal", "observed")
    world, x, y, ate, observed = _draw(rng, draw, spec.agree, P_NONZERO if numeric else 0.5)
    if variant == "causal":
        kind, answer = "number", _number(ate)
    elif variant == "observed":
        kind, answer = "number", _number(observed)
    elif variant == "sign":
        kind, answer = "yesno", "yes" if ate > 0 else "no"
    else:
        kind, answer = "yesno", "yes" if observed > ate else "no"
    names = surface.names(len(world.names))
    world = World(names, world.edges, world.cpts)
    context = (
        describe_graph(names, world.edges, surface)
        + " "
        + surface.phrase("mechanisms")
        + " "
        + describe_probabilities(world, surface)
    )
    question = surface.phrase(f"ask_{variant}").format(x=names[x], y=names[y])
    meta = {
        "names": names,
        "edges": world.edges,
        "cpts": world.cpts,
        "x": x,
        "y": y,
        "ate": ate,
        "observed": observed,
        "variant": variant,
    }
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


def describe_independences(names: tuple[str, ...], graph: nx.DiGraph, surface: Surface) -> str:
    """Every (conditional) independence among the variables; all other pairs are dependent given any set."""
    n = len(names)
    given, none = surface.phrase("indep_given"), surface.phrase("indep_none")
    lines = []
    for a, b in itertools.combinations(range(n), 2):
        others = [v for v in range(n) if v not in (a, b)]
        for k in range(len(others) + 1):
            for z in itertools.combinations(others, k):
                if d_separated(graph, a, b, set(z)):
                    fill = {"a": names[a], "b": names[b], "z": _join([names[v] for v in z]) if z else ""}
                    lines.append((given if z else none).format(**fill))
    if not lines:
        return surface.phrase("no_independences")
    return f"{surface.phrase('independences')} {' '.join(lines)} {surface.phrase('dependent_rest')}"


def discovery(rng: np.random.Generator, surface: Surface, spec: Spec) -> Task:
    """What must, or could, be true of the causal structure given all of its independences?

    The statement type and the mode are drawn first and the label is balanced within them, so no fixed
    answer per question type beats chance.
    """
    relation, mode = str(rng.choice(RELATIONS)), str(rng.choice(["must", "could"]))

    def draw() -> Case:
        """A random graph and pair; the label is whether the statement holds in every (or some) equivalent DAG."""
        n = _size(rng, "discovery", spec)
        edges = random_dag(rng, n, float(rng.uniform(0.3, 0.6)))
        nodes = tuple(str(i) for i in range(n))
        scaffold = Scaffold(
            nodes, dict.fromkeys(nodes, Role.CONFOUNDER), tuple((str(p), str(c)) for p, c in edges), (), nodes[0]
        )
        members = [_relations(member, nodes) for member in equivalence_class(scaffold)]
        a, b = (str(v) for v in rng.choice(nodes, size=2, replace=False))
        holds = [facts[(relation, a, b)] for facts in members]
        label = all(holds) if mode == "must" else any(holds)
        return label, None, (n, edges, int(a), int(b), label, len(members))

    n, edges, a, b, label, size = _draw(rng, draw, spec.agree)
    names = surface.names(n)
    study = surface.phrase("study").format(n=n, names=_join([names[i] for i in surface.order(n)]))
    context = study + " " + describe_independences(names, _graph(edges, n), surface)
    question = surface.phrase(f"ask_{mode}").format(s=surface.phrase(relation).format(a=names[a], b=names[b]))
    meta = {
        "names": names,
        "edges": edges,
        "relation": relation,
        "a": a,
        "b": b,
        "variant": mode,
        "class_size": size,
    }
    return Task("discovery", "yesno", _prompt(context, question, "yesno"), "yes" if label else "no", meta)


FAMILIES: dict[str, Callable[[np.random.Generator, Surface, Spec], Task]] = {
    "independence": independence,
    "adjustment": adjustment,
    "effect": effect,
    "discovery": discovery,
}
