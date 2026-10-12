"""Tests for the no-LLM scripted baseline.

This baseline is a control, so the properties that matter are the ones that make it a *fair* comparison:
it goes through the same answer contract and the same scorer as a model, and it is provably blind to the
column names.
"""

import pytest

from causalagentbench import build_instance, sample
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.scoring.scripted_agent import AmbiguityPolicy, scripted_answer, scripted_scores
from causalagentbench.task.answer import validate_answer_contract

pytest.importorskip("causallearn", reason="the scripted baseline needs the discovery extra")


def _instance(seed: int):
    scm, _ = generate_gated_procedural(seed)
    data = sample(scm, 200, seed=seed)
    return build_instance(scm, data, seed=seed)


def test_the_script_satisfies_the_same_answer_contract_a_model_must():
    """It is only a fair control if it is graded through the identical path, not a private one."""
    instance, _key = _instance(0)
    answer = scripted_answer(instance, policy=AmbiguityPolicy.EXCLUDE)
    validated = validate_answer_contract(
        answer, instance.candidate_drivers, columns=instance.columns, outcome=instance.outcome
    )
    assert len(validated.verdicts) == len(instance.candidate_drivers)
    assert all(v.effect is not None for v in validated.verdicts)
    assert all(v.identifiable_from_data is not None for v in validated.verdicts)


def test_the_two_ambiguity_policies_actually_differ():
    """If they never diverged there would be no orientation ambiguity to measure in the first place."""
    instance, _key = _instance(0)
    exclude = scripted_answer(instance, policy=AmbiguityPolicy.EXCLUDE)
    include = scripted_answer(instance, policy=AmbiguityPolicy.INCLUDE)
    assert [v.adjustment_set for v in exclude.verdicts] != [v.adjustment_set for v in include.verdicts]


def _renamed(instance, key, alignment, prefix):
    """A second naming skin of the SAME instance: identical data and identical column ORDER, new names.

    Built by renaming rather than by re-running ``build_instance`` with a different seed, because a
    different seed also reshuffles the display order - and PC is order-dependent, so that would compare two
    things at once. This mirrors how the real pipeline builds paired skins (one seed, several labelers).
    """
    node_to_label = {node: f"{prefix}{i}" for i, node in enumerate(key.node_to_label)}
    columns = tuple(node_to_label[key.label_to_node[c]] for c in instance.columns)
    provenance = type(key.provenance)(
        alignment=alignment,
        n_aligned=0,
        n_neutral=0,
        n_misaligned=0,
        node_kind=dict(key.provenance.node_kind),
        domain=key.provenance.domain,
        producer=key.provenance.producer,
    )
    renamed_instance = type(instance)(
        columns=columns,
        data=instance.data,
        outcome=node_to_label[key.label_to_node[instance.outcome]],
        candidate_drivers=tuple(node_to_label[key.label_to_node[d]] for d in instance.candidate_drivers),
        prompt=instance.prompt,
        context=instance.context,
        column_schema={},
    )
    renamed_key = type(key)(
        scm=key.scm,
        label_to_node={label: node for node, label in node_to_label.items()},
        node_to_label=node_to_label,
        provenance=provenance,
    )
    return renamed_instance, renamed_key


def test_paired_skins_share_the_column_order():
    """A precondition for every naming contrast in the benchmark, and it is not self-evident.

    PC is order-dependent (Colombo & Maathuis 2014), so if two skins of one SCM presented their columns in
    different orders, discovery would differ between them and a naming contrast would pick that up as a
    semantic effect. ``build_instance`` splits its label and shuffle RNGs from one seed precisely so this
    cannot happen; this pins it.
    """
    from causalagentbench.task.labeling import abstract_labels

    scm, _ = generate_gated_procedural(0)
    data = sample(scm, 200, seed=0)
    first_instance, first_key = build_instance(scm, data, seed=3, labeler=abstract_labels)
    second_instance, second_key = build_instance(scm, data, seed=3, labeler=abstract_labels)
    order_of = lambda i, k: tuple(k.label_to_node[c] for c in i.columns)  # noqa: E731
    assert order_of(first_instance, first_key) == order_of(second_instance, second_key)


def test_the_script_is_blind_to_the_column_names():
    """The design's own negative control.

    The script reads only the numbers, so its score must be identical across naming skins of one SCM. A
    non-zero contrast here would mean the *pipeline* leaks naming into the score, which would invalidate any
    Gain or Harm measured on a model.
    """
    scm, _ = generate_gated_procedural(0)
    data = sample(scm, 200, seed=0)
    base = build_instance(scm, data, seed=0)
    skins = [_renamed(*base, alignment, prefix) for alignment, prefix in ((1.0, "aa"), (-1.0, "bb"), (None, "cc"))]
    scores = scripted_scores(skins, policy=AmbiguityPolicy.EXCLUDE)
    assert len({round(s.overall, 12) for s in scores}) == 1
