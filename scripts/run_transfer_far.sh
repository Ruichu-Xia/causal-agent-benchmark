#!/bin/zsh
# Far transfer, pre-registered sets (docs/transfer_preregistration.md), for the models given as arguments.
cd "${0:A:h}/.."
models=(); for m in "$@"; do models+=(--model $m); done
ev() { uv run python scripts/transfer_eval.py run "$@" --workers 120; }
ev primary $models --samples 4
ev c2c_test $models --samples 2
ev cladder_correlation $models --samples 2
ev cladder_ate $models --samples 2
