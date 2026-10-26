import hashlib
import json
import re
import sys

import numpy as np
import pytest

from causalagentbench import cli
from causalagentbench.agent import eval as evaluator
from causalagentbench.agent.loop import MAX_TOOL_CALLS
from causalagentbench.benchmark import N_ROWS, make_task
from causalagentbench.cli import main
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.baselines import BASELINES
from causalagentbench.scoring.gates import generate_scm
from causalagentbench.scoring.score import SCORING_VERSION
from causalagentbench.task.dataset import concept_pairs, load_dataset, save_dataset
from causalagentbench.task.instance import build_instance


@pytest.fixture
def dataset(tmp_path):
    """A frozen two-task dataset with placeholder names and small tables."""
    tasks = []
    for seed in range(2):
        scm, _ = generate_scm(np.random.SeedSequence(seed))
        data = sample(scm, 300, seed).astype(np.float32).astype(np.float64)
        tasks.append(build_instance(scm, data, seed))
    save_dataset(tmp_path / "dev", tasks, split="dev")
    return tmp_path / "dev"


def _run(monkeypatch, *args) -> int:
    """Run the command line with ``args`` and return its exit status."""
    monkeypatch.setattr(sys, "argv", ["causalagentbench", *map(str, args)])
    return main()


def _records(path) -> list[dict]:
    """Every record of a JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episodes(path) -> list[dict]:
    """The episode records of a results file, in file order."""
    return [r for r in _records(path) if r["type"] == "episode"]


def _printed(text: str) -> list[dict]:
    """The JSON objects a command printed to stdout, one after another."""
    decoder, objects, rest = json.JSONDecoder(), [], text.strip()
    while rest:
        obj, end = decoder.raw_decode(rest)
        objects.append(obj)
        rest = rest[end:].lstrip()
    return objects


def _baseline(monkeypatch, dataset, out, agent: str = "oracle", *extra) -> int:
    """Run one baseline over ``dataset`` into ``out``."""
    return _run(monkeypatch, "baseline", "--dataset", dataset, "--agent", agent, "--out", out, *extra)


def test_baseline_writes_a_header_episodes_and_a_summary(dataset, tmp_path, monkeypatch, capsys):
    out = tmp_path / "zero.jsonl"
    assert _baseline(monkeypatch, dataset, out, "zero", "--workers", 2) == 0
    header, *rest = _records(out)
    assert header["type"] == "run" and header["scoring_version"] == SCORING_VERSION
    assert header["dataset"] == str(dataset) and re.fullmatch(r"[0-9a-f]{64}", header["dataset_sha256"])
    assert header["tasks"] == 2 and header["models"] == ["zero"] and not header["placeholder_names"]
    assert header["max_tool_calls"] == MAX_TOOL_CALLS
    assert {"git_commit", "git_dirty", "python", "packages"} <= set(header)
    assert sorted(r["index"] for r in rest) == [0, 1]
    assert all(r["type"] == "episode" and r["task_error"] == 1.0 and r["reward"] == 0.0 for r in rest)
    [summary] = _printed(capsys.readouterr().out)
    assert summary["model"] == "zero" and summary["tasks"] == 2
    assert summary["capped_error"]["value"] == 1.0
    assert not out.with_suffix(".transcripts.jsonl").exists()  # baselines have no conversation


@pytest.mark.parametrize("agent", BASELINES)
def test_every_baseline_runs_from_the_command_line(dataset, tmp_path, monkeypatch, agent):
    out = tmp_path / f"{agent}.jsonl"
    assert _baseline(monkeypatch, dataset, out, agent) == 0
    episodes = _episodes(out)
    assert len(episodes) == 2 and all(e["parse_ok"] and e["model"] == agent for e in episodes)


def test_eval_scores_each_model_and_saves_transcripts(dataset, tmp_path, monkeypatch, fake_llm):
    out = tmp_path / "models.jsonl"
    assert _run(monkeypatch, "eval", "--dataset", dataset, "--model", "a", "--model", "b", "--out", out) == 0
    assert _records(out)[0]["models"] == ["a", "b"]
    episodes = _episodes(out)
    assert sorted((e["model"], e["index"]) for e in episodes) == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    assert all(e["parse_ok"] and e["served_models"] == ["fake-served"] and e["tool_calls"] == 1 for e in episodes)
    transcripts = _records(out.with_suffix(".transcripts.jsonl"))
    assert len(transcripts) == 4
    answers = {(e["model"], e["index"]): e["answer"] for e in episodes}
    assert all(t["final_answer"] == answers[(t["model"], t["index"])] for t in transcripts)


def test_an_invalid_model_answer_counts_as_zero_estimates(dataset, tmp_path, monkeypatch, fake_llm, capsys):
    monkeypatch.setattr(evaluator, "run_agent", lambda *args, **kwargs: "I could not decide.")
    out = tmp_path / "invalid.jsonl"
    assert _run(monkeypatch, "eval", "--dataset", dataset, "--model", "m", "--out", out) == 0
    assert all(not e["parse_ok"] and e["reward"] == -1.0 and e["task_error"] == 1.0 for e in _episodes(out))
    [summary] = _printed(capsys.readouterr().out)
    assert summary["answer_valid_rate"] == 0.0 and summary["capped_error"]["value"] == 1.0


def test_provider_errors_are_recorded_and_retried_on_resume(dataset, tmp_path, monkeypatch, fake_llm, capsys):
    failing_columns = load_dataset(dataset)[0][0].columns
    real_run_agent = evaluator.run_agent

    def flaky(instance, client, **kwargs):
        """Fail on task 0 as a provider outage would; run every other task normally."""
        if instance.columns == failing_columns:
            raise RuntimeError("provider unavailable")
        return real_run_agent(instance, client, **kwargs)

    out = tmp_path / "flaky.jsonl"
    monkeypatch.setattr(evaluator, "run_agent", flaky)
    assert _run(monkeypatch, "eval", "--dataset", dataset, "--model", "m", "--out", out) == 1
    assert "rerun with --resume" in capsys.readouterr().err
    assert [(r["type"], r["index"]) for r in _records(out)[1:]] == [("error", 0), ("episode", 1)]

    monkeypatch.setattr(evaluator, "run_agent", real_run_agent)
    assert _run(monkeypatch, "eval", "--dataset", dataset, "--model", "m", "--out", out, "--resume") == 0
    assert [(r["type"], r["index"]) for r in _records(out)[1:]] == [("error", 0), ("episode", 1), ("episode", 0)]


def test_a_run_with_no_scored_episode_exits_nonzero(dataset, tmp_path, monkeypatch, fake_llm, capsys):
    """``_report_file`` documents "return 1 if any task has no scored episode"; here no task has one."""
    monkeypatch.setattr(fake_llm, "fail", True)
    out = tmp_path / "down.jsonl"
    status = _run(monkeypatch, "eval", "--dataset", dataset, "--model", "m", "--out", out)
    assert [r["type"] for r in _records(out)[1:]] == ["error", "error"]
    assert status == 1
    assert "rerun with --resume" in capsys.readouterr().err


def test_resume_runs_only_the_missing_episodes(dataset, tmp_path, monkeypatch):
    out = tmp_path / "oracle.jsonl"
    assert _baseline(monkeypatch, dataset, out) == 0
    header, *episodes = _records(out)
    kept = next(e for e in episodes if e["index"] == 0)
    out.write_text(json.dumps(header) + "\n" + json.dumps(kept) + "\n", encoding="utf-8")  # an interrupted run

    evaluated = []
    real_evaluate_task = cli.evaluate_task

    def spy(index, task, **kwargs):
        """Record which task is evaluated, then evaluate it for real."""
        evaluated.append(index)
        return real_evaluate_task(index, task, **kwargs)

    monkeypatch.setattr(cli, "evaluate_task", spy)
    assert _baseline(monkeypatch, dataset, out, "oracle", "--resume") == 0
    assert evaluated == [1]
    assert sorted(e["index"] for e in _episodes(out)) == [0, 1]
    assert _records(out)[0] == header


def test_an_existing_results_file_needs_resume(dataset, tmp_path, monkeypatch):
    out = tmp_path / "oracle.jsonl"
    _baseline(monkeypatch, dataset, out)
    with pytest.raises(FileExistsError, match="--resume"):
        _baseline(monkeypatch, dataset, out)


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        (("--max-tool-calls", 7), "max_tool_calls"),
        (("--placeholder-names",), "placeholder_names"),
        (("--limit", 1), "tasks"),
    ],
)
def test_resume_refuses_a_mismatched_header(dataset, tmp_path, monkeypatch, extra, field):
    out = tmp_path / "oracle.jsonl"
    _baseline(monkeypatch, dataset, out)
    with pytest.raises(ValueError, match=f"header differs in \\['{field}'\\]"):
        _baseline(monkeypatch, dataset, out, "oracle", "--resume", *extra)


def test_placeholder_names_change_the_labels_but_not_a_name_blind_score(dataset, tmp_path, monkeypatch):
    named, blind = tmp_path / "named.jsonl", tmp_path / "blind.jsonl"
    _baseline(monkeypatch, dataset, named)
    _baseline(monkeypatch, dataset, blind, "oracle", "--placeholder-names")
    assert _records(blind)[0]["placeholder_names"]
    by_index = {e["index"]: e for e in _episodes(named)}
    for episode in _episodes(blind):
        twin = by_index[episode["index"]]
        assert all(re.fullmatch(r"X\d+", c["label"]) for c in episode["candidates"])
        assert [c["node"] for c in episode["candidates"]] == [c["node"] for c in twin["candidates"]]
        assert episode["task_error"] == pytest.approx(twin["task_error"], abs=1e-12)


def test_report_prints_the_summary_a_comparison_and_the_names_ablation(dataset, tmp_path, monkeypatch, capsys):
    oracle, zero, blind = tmp_path / "oracle.jsonl", tmp_path / "zero.jsonl", tmp_path / "blind.jsonl"
    _baseline(monkeypatch, dataset, oracle)
    _baseline(monkeypatch, dataset, zero, "zero")
    _baseline(monkeypatch, dataset, blind, "oracle", "--placeholder-names")
    capsys.readouterr()

    assert _run(monkeypatch, "report", oracle) == 0
    [summary] = _printed(capsys.readouterr().out)
    assert summary["model"] == "oracle" and summary["tasks"] == 2

    assert _run(monkeypatch, "report", oracle, "--compare", zero) == 0
    _, comparison = _printed(capsys.readouterr().out)
    assert (comparison["first"], comparison["second"], comparison["tasks"]) == ("oracle", "zero", 2)
    assert comparison["difference"]["value"] == pytest.approx(summary["capped_error"]["value"] - 1.0)

    assert _run(monkeypatch, "report", oracle, "--ablation", blind) == 0
    _, ablation = _printed(capsys.readouterr().out)
    assert ablation["model"] == "oracle"
    assert ablation["names_ablation"]["difference_in_differences"]["value"] == pytest.approx(0.0, abs=1e-9)


def test_report_refuses_mismatched_pairs(dataset, tmp_path, monkeypatch):
    oracle, blind, partial = tmp_path / "oracle.jsonl", tmp_path / "blind.jsonl", tmp_path / "partial.jsonl"
    _baseline(monkeypatch, dataset, oracle)
    _baseline(monkeypatch, dataset, blind, "oracle", "--placeholder-names")
    with pytest.raises(ValueError, match="real-names run first"):
        _run(monkeypatch, "report", blind, "--ablation", oracle)
    header, *rest = _records(oracle)
    partial.write_text("\n".join(json.dumps(r) for r in [{**header, "dataset_sha256": "0" * 64}, *rest]) + "\n")
    with pytest.raises(ValueError, match="different datasets"):
        _run(monkeypatch, "report", oracle, "--compare", partial)


@pytest.mark.parametrize("placeholder", [False, True])
def test_rescore_reproduces_every_episode(dataset, tmp_path, monkeypatch, fake_llm, placeholder):
    out, rescored = tmp_path / "run.jsonl", tmp_path / "rescored.jsonl"
    flags = ("--placeholder-names",) if placeholder else ()
    assert _run(monkeypatch, "eval", "--dataset", dataset, "--model", "m", "--out", out, *flags) == 0
    assert _run(monkeypatch, "rescore", out, "--out", rescored) == 0
    header = _records(rescored)[0]
    assert header["rescored_from"] == str(out) and header["placeholder_names"] is placeholder
    assert _episodes(rescored) == _episodes(out)


def test_rescore_refuses_a_changed_dataset(dataset, tmp_path, monkeypatch):
    out = tmp_path / "run.jsonl"
    _baseline(monkeypatch, dataset, out)
    with dataset.with_suffix(".json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="dataset files changed"):
        _run(monkeypatch, "rescore", out, "--out", tmp_path / "rescored.jsonl")


def test_build_freezes_the_tasks_make_task_defines(tmp_path, monkeypatch, causenet_file):
    out = tmp_path / "built"
    assert _run(monkeypatch, "build", "--causenet", causenet_file, "--split", "dev", "-n", 2, "--out", out) == 0
    labeler = CauseNetLabeler.from_path(causenet_file)
    digest = hashlib.sha256(causenet_file.read_bytes()).hexdigest()
    for index, (instance, key) in enumerate(load_dataset(out)):
        expected, expected_key = make_task("dev", index, labeler=labeler)
        assert key.producer == f"causenet:sha256:{digest}"
        assert instance.columns == expected.columns and instance.prompt == expected.prompt
        assert instance.data.shape == (N_ROWS, key.scm.n_nodes)
        assert np.array_equal(instance.data, expected.data)  # exact: tasks are built at float32 precision
        assert key.scm.scaffold.edges == expected_key.scm.scaffold.edges


def test_build_train_avoids_the_concept_pairs_of_the_excluded_datasets(tmp_path, monkeypatch, causenet_file):
    dev, train = tmp_path / "dev", tmp_path / "train"
    assert _run(monkeypatch, "build", "--causenet", causenet_file, "--split", "dev", "-n", 2, "--out", dev) == 0
    args = ("build", "--causenet", causenet_file, "--split", "train", "-n", 2, "--out", train)
    assert _run(monkeypatch, *args, "--exclude", dev) == 0
    excluded = concept_pairs(load_dataset(dev))
    for _, key in load_dataset(train):
        edges = {frozenset((key.node_to_label[a], key.node_to_label[b])) for a, b in key.scm.scaffold.edges}
        assert not edges & excluded


def test_build_requires_exclude_exactly_for_the_train_split(tmp_path, monkeypatch, causenet_file):
    base = ("build", "--causenet", causenet_file, "-n", 1, "--out", tmp_path / "x")
    with pytest.raises(ValueError, match="--exclude"):
        _run(monkeypatch, *base, "--split", "train")
    with pytest.raises(ValueError, match="--exclude"):
        _run(monkeypatch, *base, "--split", "dev", "--exclude", tmp_path / "dev")
