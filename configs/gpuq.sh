#!/usr/bin/env bash
# Two-GPU queue (D66). Replaces queue_wave2.sh, which assumed one GPU and stopped every vLLM server of this user.
#   bash gpuq.sh lane9   GPU 9, port 8001: 32B main run (pre-registered e4 runner, resumes), then 32B wave 2
#   bash gpuq.sh lane7   GPU 7, port 8007: Mistral-24B-2501 main + F,M; Qwen3-8B thinking off/on; Qwen2.5-7B main + F,M
#   bash gpuq.sh tp2     GPUs 7+9, port 8007: waits for both lanes and all clients, then Llama-3.3-70B-FP8 (tensor parallel 2)
# A lane only stops servers on its own port and only this user's processes on its own GPUs. Every run resumes from its jsonl.
set -u
LANE=${1:?usage: gpuq.sh lane9|lane7|tp2}
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate crag
export PROJ=$HOME/coalrag PYTHONPATH=$HOME/coalrag/src CUDA_DEVICE_ORDER=PCI_BUS_ID
L=$PROJ/logs/gpuq.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] [$LANE] $*" | tee -a $L; }
served() { curl -s 127.0.0.1:$1/v1/models | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; }
free_gpu() {  # this user's processes on the listed GPUs, plus their 'vllm serve' parent
  for g in ${1//,/ }; do
    for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); do
      [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = "$USER" ] || continue
      pp=$(ps -o ppid= -p $p | tr -d ' '); ps -o cmd= -p $pp | grep -q "vllm serve" && kill $pp; kill $p 2>/dev/null
    done
  done; sleep 30
}
serve() {  # $1 GPUs, $2 port, $3 checkpoint dir, $4 served name, rest = vllm args
  local G=$1 P=$2 M=$3 N=$4; shift 4
  [ "$(served $P)" = "$N" ] && { say "reuse $N on :$P"; return 0; }
  [ -f "$M/config.json" ] || { say "SKIP $N: $M missing"; return 1; }
  pkill -u $USER -f "vllm serve.*--port $P( |$)"; free_gpu $G
  conda activate crag-vllm
  CUDA_VISIBLE_DEVICES=$G nohup vllm serve $M --served-model-name $N --host 127.0.0.1 --port $P --seed 0 --enable-prefix-caching "$@" \
    > $PROJ/logs/vllm_${N}_$P.log 2>&1 &
  local pid=$!; conda activate crag
  for i in $(seq 1 360); do
    [ "$(served $P)" = "$N" ] && { say "served $N on GPU $G :$P | $(grep -h 'KV cache size' $PROJ/logs/vllm_${N}_$P.log | tail -1 | sed 's/.*GPU //')"; return 0; }
    kill -0 $pid 2>/dev/null || break; sleep 5
  done
  say "SERVE FAILED $N"; tail -5 $PROJ/logs/vllm_${N}_$P.log >> $L; return 1
}
e5() {  # $1 port, $2 served name, rest = e5_wave2 args; two passes so transient errors are retried
  local P=$1 N=$2; shift 2; say "start $N $*"
  for pass in 1 2; do python -W ignore -m coalrag.experiments.e5_wave2 --served $N --base http://127.0.0.1:$P/v1 "$@" >> $PROJ/logs/e5_$N.log 2>&1; done
  say "end   $N $* | $(grep -E '^rows=' $PROJ/logs/e5_$N.log | tail -1)"
}

lane9() {
  local G=9 P=8001 N=qwen2.5-32b-gptq-int8
  serve $G $P $HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8 $N --max-model-len 8192 --gpu-memory-utilization 0.93 --max-num-seqs 16 || return 1
  say "start $N main (pre-registered e4 runner)"
  for pass in 1 2; do python -W ignore -m coalrag.experiments.e4_llm --served $N --base http://127.0.0.1:$P/v1 --tag main --conc 8 >> $PROJ/logs/e4_qwen32b_main.log 2>&1; done
  say "end   $N main | ok rows $(grep -c '"error": null' $PROJ/results/store/e4_${N}_main.jsonl) (target 4620)"
  e5 $P $N --conds wave2 --tag wave2 --conc 10
  say "lane9 complete"
}
lane7() {
  local G=7 P=8007
  pkill -u $USER -f "queue_wave2.sh"
  pgrep -u $USER -af "coalrag.experiments.e[45]" | grep -v ":8001" | awk '{print $1}' | xargs -r kill
  if serve $G $P $HOME/models/Mistral-Small-24B-Instruct-2501-FP8-dynamic mistral-small-24b-2501-fp8 --max-model-len 8192 --gpu-memory-utilization 0.92 --max-num-seqs 24; then
    e5 $P mistral-small-24b-2501-fp8 --conds main --tag main --conc 16
    e5 $P mistral-small-24b-2501-fp8 --conds F,M --tag wave2 --conc 16
  fi
  if tail -1 $PROJ/logs/fetch_qwen3.log 2>/dev/null | grep -q "Qwen3-8B" && serve $G $P $HOME/models/Qwen3-8B qwen3-8b --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 48; then
    e5 $P qwen3-8b --conds THINK --think off --tag think-off --conc 48
    e5 $P qwen3-8b --conds THINK --think on --tag think-on --conc 48 --budget 1024
  else say "Qwen3-8B skipped (download incomplete or serve failed)"; fi
  if serve $G $P $HF_MODELS/Qwen/Qwen2.5-7B-Instruct qwen2.5-7b --max-model-len 16384 --gpu-memory-utilization 0.85; then
    e5 $P qwen2.5-7b --conds main --tag main --conc 48
    e5 $P qwen2.5-7b --conds F,M --tag wave2 --conc 48
  fi
  say "lane7 complete"
}
tp2() {
  local G=7,9 P=8007 N=llama-3.3-70b-fp8 M=$HF_MODELS/MetaAI/Llama-3.3-70B-Instruct-FP8
  say "waiting for lane7 and lane9 to complete and for every LLM client (including the AR run) to finish"
  until grep -q "\[lane7\] lane7 complete" $L && grep -q "\[lane9\] lane9 complete" $L && ! pgrep -u $USER -f "coalrag.experiments.e[45]" >/dev/null; do sleep 300; done
  local q; q=$(python -c "import json,sys; c=json.load(open(sys.argv[1]+'/config.json')); print((c.get('quantization_config') or {}).get('quant_method'))" $M)
  [ "$q" = "None" ] && { say "SKIP $N: checkpoint is not pre-quantized (no quantization_config)"; return 1; }
  pkill -u $USER -f "vllm serve.*--port 8001( |$)"
  serve $G $P $M $N --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.92 --max-num-seqs 8 || return 1
  e5 $P $N --conds B0-choi,B0-std,CF-LTorig,D-bge-coal,D-oracle-coal,D-random-coal,D-strat4-coal --tag main --conc 8
  e5 $P $N --conds F --tag wave2 --conc 8
  say "tp2 complete"
}
say "started"; $LANE
