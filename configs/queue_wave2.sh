#!/usr/bin/env bash
# GPU 7 queue (D60). Waits for the 32B main run, then: 32B wave 2 and all-regime run, Mistral-24B-2501, Qwen3-8B thinking
# off/on, Qwen2.5-7B. Every stage resumes from its jsonl, so the script is safe to re-run after an interruption.
set -u
source $HOME/miniforge3/etc/profile.d/conda.sh
export PROJ=$HOME/coalrag PYTHONPATH=$HOME/coalrag/src
Q=$PROJ/logs/queue_wave2.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
served() { curl -s 127.0.0.1:8007/v1/models | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; }
serve() {  # $1 checkpoint dir, $2 served name, rest = vllm args
  local M=$1 N=$2; shift 2
  [ "$(served)" = "$N" ] && { say "reuse server $N"; return 0; }
  [ -d "$M" ] || { say "SKIP $N: $M not on disk"; return 1; }
  pkill -u $USER -f "vllm serve"; sleep 20
  conda activate crag-vllm
  CUDA_VISIBLE_DEVICES=7 nohup vllm serve $M --served-model-name $N --host 127.0.0.1 --port 8007 --seed 0 "$@" > $PROJ/logs/vllm_$N.log 2>&1 &
  echo $! > $PROJ/logs/vllm.pid
  conda activate crag
  for i in $(seq 1 240); do
    [ "$(served)" = "$N" ] && { say "served $N | $(grep -h 'KV cache size' $PROJ/logs/vllm_$N.log | tail -1 | sed 's/.*GPU //')"; return 0; }
    kill -0 $(cat $PROJ/logs/vllm.pid) 2>/dev/null || break; sleep 5
  done
  say "SERVE FAILED $N"; tail -5 $PROJ/logs/vllm_$N.log >> $Q; return 1
}
run() {  # $1 served name, rest = e5_wave2 args; two passes so transient errors are retried
  local N=$1; shift; say "start $N $*"
  for pass in 1 2; do python -W ignore -m coalrag.experiments.e5_wave2 --served $N "$@" >> $PROJ/logs/e5_$N.log 2>&1; done
  say "end   $N $* | $(grep -E '^rows=' $PROJ/logs/e5_$N.log | tail -1)"
}

say "queue started; waiting for the 32B main run"
while pgrep -u $USER -f "e4_llm.*--tag main" >/dev/null; do sleep 120; done
say "32B main run finished"
if serve $HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8 qwen2.5-32b-gptq-int8 --max-model-len 8192 --gpu-memory-utilization 0.93 --max-num-seqs 12 --max-num-batched-tokens 4096; then
  run qwen2.5-32b-gptq-int8 --conds wave2 --tag wave2 --conc 4
  run qwen2.5-32b-gptq-int8 --conds AR --queries adopted --tag allreg --conc 4
fi
if serve $HOME/models/Mistral-Small-24B-Instruct-2501-FP8-dynamic mistral-small-24b-2501-fp8 --max-model-len 8192 --gpu-memory-utilization 0.92 --max-num-seqs 24; then
  run mistral-small-24b-2501-fp8 --conds main --tag main --conc 16
  run mistral-small-24b-2501-fp8 --conds F,M --tag wave2 --conc 16
fi
if serve $HOME/models/Qwen3-8B qwen3-8b --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 48; then
  run qwen3-8b --conds THINK --think off --tag think-off --conc 48
  run qwen3-8b --conds THINK --think on --tag think-on --conc 48 --budget 1024
fi
if serve $HF_MODELS/Qwen/Qwen2.5-7B-Instruct qwen2.5-7b --max-model-len 16384 --gpu-memory-utilization 0.85; then
  run qwen2.5-7b --conds main --tag main --conc 48
  run qwen2.5-7b --conds F,M --tag wave2 --conc 48
fi
say "queue complete"
