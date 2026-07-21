import pytest
from pydantic import ValidationError

from causalagentbench.generation.config import BuildConfig


def test_minimal_config_and_defaults():
    config = BuildConfig(gen_model="openai/gpt-5")
    assert config.critic_model is None  # critic optional
    assert config.max_revisions >= 0
    assert config.n_scms > 0


def test_critic_and_revisions_settable():
    config = BuildConfig(gen_model="openai/gpt-5", critic_model="openai/gpt-5-mini", max_revisions=3)
    assert config.critic_model == "openai/gpt-5-mini"
    assert config.max_revisions == 3


def test_rejects_nonpositive_n_scms():
    with pytest.raises(ValidationError):
        BuildConfig(gen_model="a", n_scms=0)


def test_rejects_unknown_field():
    # The old judge_model is gone; unknown fields are forbidden.
    with pytest.raises(ValidationError):
        BuildConfig(gen_model="a", judge_model="b")


def test_from_yaml(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "c.yaml"
    path.write_text("gen_model: openai/gpt-5\ncritic_model: openai/gpt-5\nn_scms: 4\n", encoding="utf-8")
    config = BuildConfig.from_yaml(path)
    assert config.gen_model == "openai/gpt-5"
    assert config.n_scms == 4
