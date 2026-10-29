import itertools
import re

import networkx as nx
import numpy as np
import pytest

from causalagentbench.reasoning.audit import audit
from causalagentbench.reasoning.dataset import (
    SPLITS,
    build,
    concept_namer,
    held_out_concept,
    load_concepts,
    nonsense_names,
    plain,
)
from causalagentbench.reasoning.families import FAMILIES, _join, _number, _relations
from causalagentbench.reasoning.grading import grade
from causalagentbench.reasoning.wording import BANNED, GRAPH_STYLES, SLOTS
from causalagentbench.reasoning.worlds import World, causal_effect, d_separated, observed_difference, valid_adjustment
from causalagentbench.scm.identifiability import equivalence_class
from causalagentbench.types import Role, Scaffold


def _graph(edges: list[tuple[int, int]], n: int) -> nx.DiGraph:
    """A DAG on nodes 0..n-1."""
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(range(n))
    return graph


def test_effects_of_a_confounded_world_match_hand_computation():
    """Z -> X, Z -> Y, X -> Y: do() averages over P(Z); conditioning weights by P(Z | X)."""
    # node 0 = Z, 1 = X, 2 = Y; Y's parents are (Z, X): bit 0 is Z, bit 1 is X
    world = World(("z", "x", "y"), ((0, 1), (0, 2), (1, 2)), ((0.3,), (0.2, 0.8), (0.1, 0.5, 0.4, 0.9)))
    ate = 0.7 * (0.4 - 0.1) + 0.3 * (0.9 - 0.5)
    assert causal_effect(world, 1, 2) == pytest.approx(ate)
    p_z_given_x1 = 0.3 * 0.8 / (0.3 * 0.8 + 0.7 * 0.2)
    p_z_given_x0 = 0.3 * 0.2 / (0.3 * 0.2 + 0.7 * 0.8)
    p_y_x1 = p_z_given_x1 * 0.9 + (1 - p_z_given_x1) * 0.4
    p_y_x0 = p_z_given_x0 * 0.5 + (1 - p_z_given_x0) * 0.1
    assert observed_difference(world, 1, 2) == pytest.approx(p_y_x1 - p_y_x0)


@pytest.mark.parametrize(
    ("edges", "n", "z", "valid"),
    [
        ([(0, 1), (0, 2), (1, 2)], 3, set(), False),  # confounder Z=0 left open
        ([(0, 1), (0, 2), (1, 2)], 3, {0}, True),
        ([(1, 0), (0, 2)], 3, {0}, False),  # X=1 -> M=0 -> Y=2: adjusting for a mediator
        ([(1, 2), (1, 0), (2, 0)], 3, {0}, False),  # collider X -> C <- Y... and descendant of Y
        ([(1, 2), (1, 0)], 3, {0}, True),  # X -> W off the causal path: valid, though not back-door
        ([(3, 1), (3, 0), (4, 0), (4, 2), (1, 2)], 5, {0}, False),  # M-bias: conditioning on the collider
        ([(3, 1), (3, 0), (4, 0), (4, 2), (1, 2)], 5, {0, 3}, True),
    ],
)
def test_adjustment_criterion_known_cases(edges, n, z, valid):
    """Classic cases of the adjustment criterion for the effect of node 1 on node 2."""
    assert valid_adjustment(_graph(edges, n), 1, 2, z) is valid


def test_conditioning_on_a_collider_connects_its_parents():
    """X -> C <- Y: independent marginally, dependent given C."""
    graph = _graph([(0, 2), (1, 2)], 3)
    assert d_separated(graph, 0, 1, set())
    assert not d_separated(graph, 0, 1, {2})


def _members(edges: list[tuple[str, str]], nodes: tuple[str, ...]) -> list[dict]:
    """Structural facts of every DAG equivalent to ``edges``."""
    scaffold = Scaffold(nodes, dict.fromkeys(nodes, Role.CONFOUNDER), tuple(edges), (), nodes[0])
    return [_relations(member, nodes) for member in equivalence_class(scaffold)]


def test_chain_direction_is_unknown_but_a_collider_is_fixed():
    """A - B - C chain: 3 equivalent DAGs; A -> C <- B collider: 1."""
    chain = _members([("a", "b"), ("b", "c")], ("a", "b", "c"))
    assert len(chain) == 3
    assert not all(m[("cause", "a", "c")] for m in chain)
    assert any(m[("cause", "a", "c")] for m in chain)
    collider = _members([("a", "c"), ("b", "c")], ("a", "b", "c"))
    assert len(collider) == 1
    assert collider[0][("direct", "a", "c")] and collider[0][("common_effect", "a", "b")]


@pytest.mark.parametrize(
    ("kind", "answer", "reply", "score"),
    [
        ("yesno", "yes", "so...\nAnswer: Yes", 1.0),
        ("yesno", "yes", "**Answer: No**", 0.0),
        ("yesno", "no", "Answer: yes\nWait. Answer: no.", 1.0),
        ("yesno", "no", "### Final Answer:\n\nAnswer: No", 1.0),  # a heading above the answer line
        ("yesno", "yes", "Final answer:\nAnswer: Yes", 1.0),
        ("yesno", "yes", "**Answer**: Yes", 1.0),
        ("yesno", "yes", "Answer:\nYes", 0.0),  # the value must be on the answer line
        ("yesno", "yes", "Answer: <answer> Yes.", 1.0),  # the instruction's placeholder, echoed
        ("yesno", "no", "**Final Answer**: <answer> No </answer>", 1.0),
        ("number", "0.53", "Answer: <answer>0.53</answer>", 1.0),
        ("number", "0.14", "$$\\boxed{0.14}$$", 0.0),  # no answer line
        ("number", "-0.12", "Answer: $-0.12$", 1.0),
        ("number", "0.12", "Answer: 0.13", 1.0),
        ("number", "0.12", "Answer: 0.14", 0.0),
        ("letters", "A,C", "Answer: C, A", 1.0),
        ("letters", "none", "Answer: None", 1.0),
        ("letters", "A,C", "Answer: A", 0.0),
        ("choice", "both", "Answer: Both", 1.0),
        ("choice", "1", "the answer is 1", 0.0),
    ],
)
def test_grade(kind, answer, reply, score):
    """The last "Answer:" line decides; numbers within the tolerance count."""
    assert grade(kind, answer, reply) == score


CONCEPTS = load_concepts()


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    """200 tasks of every split, built with one seed (iid, ood_wording and ood_vocab are paired)."""
    return {split: build(200, 5, CONCEPTS, split) for split in SPLITS}


def test_build_is_deterministic_and_covers_every_family(splits):
    """The same seed gives the same tasks."""
    assert build(200, 5, CONCEPTS, "iid") == splits["iid"]
    assert {t["family"] for t in splits["iid"]} == set(FAMILIES)


def test_paired_splits_pose_the_same_problems_in_other_words(splits):
    """With one seed, ood_wording and ood_vocab ask iid's problems item by item, with different text."""
    problem_keys = ("edges", "x", "y", "z", "sets", "variant", "relation", "a", "b", "cpts")
    for split in ("ood_wording", "ood_vocab"):
        for base, other in zip(splits["iid"], splits[split], strict=True):
            assert other["answer"] == base["answer"]
            assert {k: other["meta"].get(k) for k in problem_keys} == {k: base["meta"].get(k) for k in problem_keys}
            assert other["prompt"] != base["prompt"]


def _fragments(template: str) -> list[str]:
    """The literal pieces of a template between placeholders."""
    return [piece for i, piece in enumerate(re.split(r"\{\w+\}", template)) if i % 2 == 0 or piece]


def _held_out_markers() -> list[str]:
    """Literal fragments of held-out templates (12+ characters) that occur in no training template."""
    training = [t for train, _ in SLOTS.values() for t in train] + [t for _, t in GRAPH_STYLES[0]]
    held = [t for _, out in SLOTS.values() for t in out] + [t for _, t in GRAPH_STYLES[1]]
    markers = [f.strip() for t in held for f in _fragments(t) if len(f.strip()) >= 12]
    return [m for m in markers if not any(m in t for t in training)]


def test_held_out_wording_appears_only_in_its_split(splits):
    """No training-pool split contains a held-out template; every ood_wording prompt does."""
    markers = _held_out_markers()
    assert len(markers) > 20
    for split in ("train", "iid", "ood_size", "ood_vocab"):
        assert not [t["id"] for t in splits[split] for m in markers if m in t["prompt"]]
    assert all(any(m in t["prompt"] for m in markers) for t in splits["ood_wording"])


def test_held_out_concepts_appear_only_in_their_split(splits):
    """Concept names are partitioned by the stable hash; ood_vocab uses only held-out concepts."""
    for split, tasks in splits.items():
        concept_names = {name for t in tasks if t["naming"] == "concepts" for name in t["meta"]["names"]}
        assert concept_names
        assert all(held_out_concept(name) == (split == "ood_vocab") for name in concept_names)


def test_no_template_quotes_the_far_transfer_benchmark():
    """No template contains a phrase specific to CLadder."""
    templates = [t for pools in SLOTS.values() for pool in pools for t in pool]
    templates += [t for pool in GRAPH_STYLES for _, t in pool]
    assert not [(t, b) for t in templates for b in BANNED if b in t.lower()]


def test_reserved_and_list_words_are_not_names():
    """Names that would read ambiguously in a template are excluded."""
    assert not plain("wear and tear") and not plain("loss or damage") and not plain("none")
    assert not plain("effect") and not plain("failure to do")  # words of the task text
    assert plain("crop failure")


def test_audit_rederives_every_label_and_finds_no_gross_shortcut(splits):
    """Re-derived answers match exactly; no shortcut or constant is far above chance.

    The bounds are loose for this small sample; ``scripts/audit_reasoning.py`` applies the strict ones
    to 10,000 tasks per split.
    """
    for split in ("iid", "ood_size"):
        report = audit(splits[split])
        assert report["label_mismatches"] == []
        for name, group in report["groups"].items():
            kind = name.split("/")[1]
            if kind == "yesno":
                assert 0.3 <= group["best_constant"] <= 0.7, (split, name, group)
                assert group["shortcut"] is None or 0.3 <= group["shortcut"] <= 0.7, (split, name, group)
            elif kind == "choice":
                assert group["best_constant"] <= 0.45 and group["shortcut"] <= 0.45, (split, name, group)
            else:
                assert group["best_constant"] <= 0.25, (split, name, group)


def _pattern(template: str) -> re.Pattern:
    """A regex matching one rendering of ``template``, capturing each placeholder."""
    parts = re.split(r"\{(\w+)\}", template)
    body = "".join(re.escape(p) if i % 2 == 0 else f"(?P<{p}>.+?)" for i, p in enumerate(parts))
    return re.compile(body + "$", re.IGNORECASE)


def _names(text: str) -> list[str]:
    """Split 'a, b and c' into names."""
    return re.split(r", | and ", text.lower())


def _parse_edges(prompt: str, names: list[str]) -> set[tuple[int, int]]:
    """The edges a prompt's graph description states, read back with whichever style matches."""
    context = prompt.split("\n\n")[0]
    closings = [c for pool in SLOTS["closing"] for c in pool]
    # statements run from the header's last colon to the closing sentence (a header may hold two colons)
    body = next(context.split(c)[0] for c in closings if c in context).rsplit(":", 1)[1].strip()
    index = {name.lower(): i for i, name in enumerate(names)}
    for kind, template in [s for pool in GRAPH_STYLES for s in pool]:
        lines = body.split("\n") if kind == "arrows" else [s.rstrip(".") + "." for s in body.split(". ")]
        matches = [_pattern(template).match(line.strip()) for line in lines]
        if not all(matches):
            continue
        edges = set()
        for match in matches:
            g = {k: v.lower() for k, v in match.groupdict().items()}
            if kind == "parents":
                edges |= {(index[p], index[g["C"]]) for p in _names(g["ps"])}
            elif kind == "children":
                edges |= {(index[g["P"]], index[c]) for c in _names(g["cs"])}
            else:
                edges.add((index[g.get("p", g.get("P"))], index[g.get("c", g.get("C"))]))
        return edges
    raise AssertionError(f"no graph style matches: {body!r}")


def test_graph_descriptions_read_back_to_the_true_graph(splits):
    """Every family's prompt states exactly the task's edges, under training and held-out wording."""
    for split in ("iid", "ood_wording", "ood_size"):
        for task in splits[split]:
            if task["family"] != "discovery":
                edges = _parse_edges(task["prompt"], task["meta"]["names"])
                assert edges == {tuple(e) for e in task["meta"]["edges"]}, task["id"]


def test_questions_name_exactly_the_queried_variables(splits):
    """The question paragraph mentions the queried variables and no others."""
    for split in ("iid", "ood_wording", "ood_size"):
        for task in splits[split]:
            meta, names = task["meta"], task["meta"]["names"]
            question = task["prompt"].split("\n\n")[1].lower()
            mentioned = {i for i, name in enumerate(names) if re.search(rf"\b{re.escape(name.lower())}\b", question)}
            if task["family"] == "discovery":
                expected = {meta["a"], meta["b"]}
            elif task["family"] == "independence":
                expected = {meta["x"], meta["y"], *meta["z"]}
            elif task["family"] == "adjustment":
                expected = {meta["x"], meta["y"], *(v for z in meta["sets"] for v in z)}
            else:
                expected = {meta["x"], meta["y"]}
            assert mentioned == expected, task["id"]


def _parse_world(prompt: str, names: list[str], edges: list[list[int]]) -> World:
    """The world whose probability tables a prompt states, read back with whichever template matches."""
    context = prompt.split("\n\n")[0]
    mechanisms = [m for pool in SLOTS["mechanisms"] for m in pool]
    text = next(context.split(m)[1] for m in mechanisms if m in context).strip().rstrip(".")
    rows = [_pattern(t) for pool in SLOTS["cpt_row"] for t in pool]
    roots = [_pattern(t) for pool in SLOTS["cpt_root"] for t in pool]
    index = {name.lower(): i for i, name in enumerate(names)}
    world = World(tuple(names), tuple(tuple(e) for e in edges), ())
    tables: dict[int, dict[int, float]] = {i: {} for i in range(len(names))}
    for part in re.split(r"; |\. ", text):
        match = next(m for p in rows + roots if (m := p.match(part)))
        node = index[match["v"].lower()]
        k = 0
        if "given" in match.groupdict():
            values = dict(item.split("=") for item in match["given"].lower().split(", "))
            k = sum(int(values[names[q].lower()]) << j for j, q in enumerate(world.parents(node)))
        tables[node][k] = float(match["p"])
    cpts = tuple(tuple(tables[i][k] for k in range(2 ** len(world.parents(i)))) for i in range(len(names)))
    return World(tuple(names), world.edges, cpts)


def test_effect_prompts_state_tables_that_give_the_answer(splits):
    """Reading the probability tables back from the text reproduces every effect answer."""
    for split in ("iid", "ood_wording", "ood_size"):
        for task in splits[split]:
            if task["family"] != "effect":
                continue
            meta = task["meta"]
            world = _parse_world(task["prompt"], meta["names"], meta["edges"])
            assert world.cpts == tuple(tuple(c) for c in meta["cpts"]), task["id"]
            ate, observed = causal_effect(world, meta["x"], meta["y"]), observed_difference(world, meta["x"], meta["y"])
            expected = {
                "causal": _number(ate),
                "observed": _number(observed),
                "sign": "yes" if ate > 0 else "no",
                "gap": "yes" if observed > ate else "no",
            }[meta["variant"]]
            assert task["answer"] == expected, task["id"]


def test_discovery_prompts_list_exactly_the_independences(splits):
    """Each discovery prompt states every d-separation of the true graph and nothing else."""
    for split in ("iid", "ood_wording", "ood_size"):
        for task in splits[split]:
            if task["family"] != "discovery":
                continue
            names, n = task["meta"]["names"], len(task["meta"]["names"])
            graph = _graph([tuple(e) for e in task["meta"]["edges"]], n)
            facts = [
                (a, b, z)
                for a, b in itertools.combinations(range(n), 2)
                for k in range(n - 1)
                for z in itertools.combinations([v for v in range(n) if v not in (a, b)], k)
                if d_separated(graph, a, b, set(z))
            ]
            context = task["prompt"].split("\n\n")[0]
            for a, b, z in facts:
                pool = SLOTS["indep_given" if z else "indep_none"]
                fill = {"a": names[a], "b": names[b], "z": _join([names[v] for v in z]) if z else ""}
                assert any(t.format(**fill) in context for t in pool[0] + pool[1]), (task["id"], a, b, z)
            assert len(re.findall(r"\bindependent\b", context)) == len(facts), task["id"]


def test_concept_names_in_one_task_are_not_near_duplicates():
    """Multi-word concepts are compared word by word, so "disease" and "lung disease" never meet."""
    names = concept_namer(["disease", "lung disease", "human diseases", "rain"])(np.random.default_rng(0), 2)
    assert set(names) in ({"disease", "rain"}, {"lung disease", "rain"}, {"human diseases", "rain"})


def test_nonsense_names_are_distinct():
    """Names within one task never repeat."""
    names = nonsense_names(np.random.default_rng(0), 6)
    assert len(set(names)) == 6


def _adjusted(world: World, x: int, y: int, z: tuple[int, ...]) -> float:
    """The adjustment formula sum_z P(z) (P(y=1 | x=1, z) - P(y=1 | x=0, z)), by enumeration."""
    from causalagentbench.reasoning.worlds import probability

    total = 0.0
    for values in np.ndindex(*(2,) * len(z)):
        stratum = dict(zip(z, (int(v) for v in values), strict=True))
        weight = probability(world, stratum) if z else 1.0
        total += weight * (
            probability(world, {y: 1}, {**stratum, x: 1}) - probability(world, {y: 1}, {**stratum, x: 0})
        )
    return total


@pytest.mark.parametrize(("worlds", "low", "high"), [(30, 3, 5), (3, 7, 7)])
def test_verifiers_agree_with_brute_force_probability(worlds, low, high):
    """Valid sets reproduce the causal effect exactly and invalid ones do not; d-separation matches independence.

    The second case covers the ood_size graph sizes.
    """
    from itertools import combinations

    from causalagentbench.reasoning.worlds import probability, random_world

    rng = np.random.default_rng(low)
    for _ in range(worlds):
        n = int(rng.integers(low, high + 1))
        world = random_world(rng, tuple(str(i) for i in range(n)), 0.5)
        graph = world.graph()
        x, y = sorted(int(v) for v in rng.choice(n, size=2, replace=False))
        others = [v for v in range(n) if v not in (x, y)]
        for k in range(len(others) + 1):
            for z in combinations(others, k):
                bias = _adjusted(world, x, y, z) - causal_effect(world, x, y)
                assert (abs(bias) < 1e-9) is valid_adjustment(graph, x, y, set(z))
                stratum_gaps = [
                    probability(world, {y: 1}, {**dict(zip(z, v, strict=True)), x: 1})
                    - probability(world, {y: 1}, {**dict(zip(z, v, strict=True)), x: 0})
                    for v in np.ndindex(*(2,) * len(z))
                ]
                independent = max(abs(g) for g in stratum_gaps) < 1e-9
                assert independent is d_separated(graph, x, y, set(z))
