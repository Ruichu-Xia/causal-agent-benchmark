"""Build text-reasoning datasets: tasks from every family, named with nonsense words or real concepts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from causalagentbench.generation.causenet import is_near_duplicate
from causalagentbench.reasoning.families import FAMILIES, FORMATS, Namer, Spec, Surface, _prompt
from causalagentbench.reasoning.wording import GRAPH_STYLES, SLOTS

_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z"]
_VOWELS = ["a", "e", "i", "o", "u"]


def nonsense_names(rng: np.random.Generator, n: int) -> tuple[str, ...]:
    """``n`` distinct pronounceable two- or three-syllable nonsense words."""
    names: list[str] = []
    while len(names) < n:
        syllables = int(rng.integers(2, 4))
        word = "".join(str(rng.choice(_ONSETS)) + str(rng.choice(_VOWELS)) for _ in range(syllables))
        if word not in names:
            names.append(word)
    return tuple(names)


def concept_namer(concepts: Sequence[str]) -> Namer:
    """Draw distinct real concepts at random; their causal relations in the world are arbitrary."""

    def draw(rng: np.random.Generator, n: int) -> tuple[str, ...]:
        """``n`` concepts, no two near-duplicates (such as "disease" and "lung disease")."""
        names: list[str] = []
        for _ in range(100 * n):
            concept = str(concepts[int(rng.integers(len(concepts)))])
            # is_near_duplicate splits words on "_" (CauseNet's form); pool names use spaces
            if all(not is_near_duplicate(concept.replace(" ", "_"), other.replace(" ", "_")) for other in names):
                names.append(concept)
            if len(names) == n:
                return tuple(names)
        raise ValueError(f"could not draw {n} distinct concepts from a pool of {len(concepts)}")

    return draw


RESERVED = frozenset({"nothing", "none", "both", "neither", "yes", "no"})
"""Words the code writes into prompts outside the templates: empty sets and answer options."""

TASK_WORDS = frozenset(
    re.findall(
        "[a-z]+",
        " ".join(
            [t.lower() for pools in SLOTS.values() for pool in pools for t in pool]
            + [t.lower() for pool in GRAPH_STYLES for _, t in pool]
            + [_prompt("", "", kind).lower() for kind in FORMATS]
        ),
    )
)
"""Every word of the fixed task text (templates, instruction, answer formats), placeholders included."""


def plain(name: str) -> bool:
    """Whether a name reads unambiguously in a prompt.

    It must be one to three letter-only words, none of them "and", "or", a reserved word, or any word of
    the task text: "failure to do" would make "from {p} to {c}" parse two ways, and "climate change"
    would collide with "change the probability".
    """
    words = name.split()
    if name in RESERVED or {"and", "or"} & set(words) or TASK_WORDS & set(words):
        return False
    return 1 <= len(words) <= 3 and all(w.isalpha() for w in words)


def concept_pool(graph: nx.DiGraph, min_degree: int = 20) -> list[str]:
    """Common CauseNet concepts usable as variable names, lower case and sorted."""
    pool = []
    for node, degree in graph.degree():
        words = str(node).replace("_", " ").lower()
        if degree >= min_degree and plain(words):
            pool.append(words)
    return sorted(set(pool))


CONCEPTS = Path(__file__).with_name("concepts.txt")
"""The concept-name pool: ``concept_pool`` of CauseNet-Precision (CC BY 4.0; see ``data/README.md``)."""


def load_concepts() -> list[str]:
    """The packaged concept-name pool."""
    return CONCEPTS.read_text(encoding="utf-8").splitlines()


SPLITS = ("train", "iid", "ood_wording", "ood_size", "ood_vocab")
"""Each held-out split changes one factor from ``iid``: wording, graph size, or concept vocabulary."""


def held_out_concept(concept: str) -> bool:
    """Whether a concept is one of the fifth reserved for ``ood_vocab`` (a stable hash, independent of order)."""
    return int.from_bytes(hashlib.sha256(concept.encode("utf-8")).digest()[:8], "big") % 5 == 0


def build(
    n: int, seed: int, concepts: Sequence[str], split: str = "train", families: Sequence[str] = tuple(FAMILIES)
) -> list[dict[str, Any]]:
    """``n`` tasks of one split, cycling through ``families``.

    Problems come from one stream seeded by ``seed`` and wording and names from another, so ``iid``,
    ``ood_wording`` and ``ood_vocab`` built with the same seed pose the same problems item by item.
    Names are nonsense words or concepts with equal probability; ``ood_vocab`` uses held-out concepts
    only, and every other split uses only the rest.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    spec = Spec(large=split == "ood_size")
    pool = [c for c in concepts if plain(c) and held_out_concept(c) == (split == "ood_vocab")]
    namers: dict[str, Namer] = {"nonsense": nonsense_names, "concepts": concept_namer(pool)}
    rng, surface_rng = np.random.default_rng(seed), np.random.default_rng([seed, 1])
    tasks = []
    for i in range(n):
        family = families[i % len(families)]
        naming = "concepts" if split == "ood_vocab" or surface_rng.random() < 0.5 else "nonsense"
        surface = Surface(surface_rng, namers[naming], held_out=split == "ood_wording")
        task = FAMILIES[family](rng, surface, spec)
        tasks.append({"id": f"{split}-{seed}-{i}", "split": split, "naming": naming, **asdict(task)})
    return tasks


def save(tasks: list[dict[str, Any]], path: str | Path) -> None:
    """Write tasks as JSON lines."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")


def load(path: str | Path) -> list[dict[str, Any]]:
    """Read tasks written by ``save``."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
