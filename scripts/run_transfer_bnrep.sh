#!/bin/zsh
# Near transfer: the bnRep tasks under the in-domain protocol, real and placeholder names, for each model given.
# Failed episodes are retried once with --resume.
cd "${0:A:h}/.."
run() {
  uv run --extra agents causalagentbench eval --dataset outputs/transfer/bnrep --model hosted_vllm/$1 --workers 6 \
    --timeout 300 ${=3} --out outputs/transfer/bnrep_$2.jsonl >> outputs/transfer/bnrep_$2.log 2>&1
}
for m in "$@"; do
  run $m $m "" &
  run $m ${m}_x --placeholder-names &
done
wait
for m in "$@"; do
  run $m $m --resume &
  run $m ${m}_x "--placeholder-names --resume" &
done
wait
