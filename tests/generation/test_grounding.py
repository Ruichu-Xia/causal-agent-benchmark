import bz2
import json

import pytest

from causalagentbench.generation.grounding import load_causenet


@pytest.mark.parametrize("compressed", [False, True])
def test_load_relations_preserves_direction_and_support(tmp_path, compressed):
    path = tmp_path / ("relations.jsonl.bz2" if compressed else "relations.jsonl")
    records = [
        {"causal_relation": {"cause": {"concept": "smoking"}, "effect": {"concept": "lung_cancer"}}, "support": 4},
        {"causal_relation": {"cause": {"concept": "smoking"}, "effect": {"concept": "lung_cancer"}}, "support": 2},
    ]
    content = "\n".join(json.dumps(r) for r in records)
    path.write_bytes(bz2.compress(content.encode()) if compressed else content.encode())
    graph = load_causenet(str(path))
    assert graph["smoking"]["lung_cancer"]["weight"] == 6
    assert not graph.has_edge("lung_cancer", "smoking")
