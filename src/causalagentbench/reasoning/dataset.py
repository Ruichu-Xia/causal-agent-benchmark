"""Build text-reasoning datasets: tasks from every family, named with nonsense words or real concepts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from causalagentbench.generation.causenet import is_near_duplicate
from causalagentbench.reasoning.families import FAMILIES, Namer

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
        """``n`` concepts, no two near-duplicates."""
        names: list[str] = []
        while len(names) < n:
            concept = str(concepts[int(rng.integers(len(concepts)))])
            if all(not is_near_duplicate(concept, other) for other in names):
                names.append(concept)
        return tuple(names)

    return draw


def concept_pool(graph: nx.DiGraph, min_degree: int = 20) -> list[str]:
    """Common CauseNet concepts usable as variable names: one to three plain words, sorted."""
    pool = []
    for node, degree in graph.degree():
        words = str(node).replace("_", " ")
        if degree >= min_degree and 1 <= len(words.split()) <= 3 and words.replace(" ", "").isalpha():
            pool.append(words.lower())
    return sorted(set(pool))


def build(
    n: int, seed: int, concepts: Sequence[str], families: Sequence[str] = tuple(FAMILIES)
) -> list[dict[str, Any]]:
    """``n`` tasks cycling through ``families``, half named with nonsense words and half with concepts."""
    rng = np.random.default_rng(seed)
    namers: dict[str, Namer] = {"nonsense": nonsense_names, "concepts": concept_namer(concepts)}
    tasks = []
    for i in range(n):
        family = families[i % len(families)]
        naming = "nonsense" if rng.random() < 0.5 else "concepts"
        task = FAMILIES[family](rng, namers[naming])
        tasks.append({"id": f"{seed}-{i}", "naming": naming, **asdict(task)})
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
