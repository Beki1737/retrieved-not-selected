#!/usr/bin/env bash
# Two-GPU queue v2 (D68, D69). Replaces gpuq.sh. Memory-safe serving for echo log-likelihood scoring:
# V0 engine, eager mode, gpu-memory-utilization <= 0.80, max-num-batched-tokens = max-model-len, and a preflight that
# scores 8 near-maximum-length prompts before any run. If a configuration fails, the next one in the chain is tried.
#   bash gpuq2.sh stop    stop every lane, LLM client and vLLM server of this user on GPUs 7 and 9 (ports 8001, 8007)
#   bash gpuq2.sh diag    coverage, errors, prompt lengths, Llama format, template audit
#   bash gpuq2.sh lane9   GPU 9 :8001  32B main (pre-registered e4 runner) -> 32B all-regime -> 32B wave 2
#   bash gpuq2.sh lane7   GPU 7 :8007  Mistral-24B-2501 main + F,M -> Qwen3-8B thinking off/on
#   bash gpuq2.sh tp2     GPUs 7+9 :8007  waits for both lanes, then Llama-3.3-70B-FP8 (reduced set + F)
set -u
LANE=${1:?usage: gpuq2.sh stop|diag|lane9|lane7|tp2}
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate crag
export PROJ=$HOME/coalrag PYTHONPATH=$HOME/coalrag/src CUDA_DEVICE_ORDER=PCI_BUS_ID
L=$PROJ/logs/gpuq2.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] [$LANE] $*" | tee -a $L; }
served() { curl -s -m 10 127.0.0.1:$1/v1/models | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; }
cov() { python -m coalrag.tools.diag 2>/dev/null | grep "^e4_$1_$2.jsonl" | cut -d'|' -f1-3; }
free_gpu() {  # this user's processes on the listed GPUs, plus their 'vllm serve' parent
  for g in ${1//,/ }; do
    for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); do
      [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = "$USER" ] || continue
      pp=$(ps -o ppid= -p $p | tr -d ' '); ps -o cmd= -p $pp | grep -q "vllm serve" && kill $pp; kill $p 2>/dev/null
    done
  done; sleep 30
}
stop_port() { pkill -u $USER -f "vllm serve.*--port $1( |$)"; sleep 5; }
serve_chain() {  # $1 GPUs, $2 port, $3 checkpoint, $4 served name, then one quoted string of vllm args per configuration
  local G=$1 P=$2 M=$3 N=$4; shift 4
  [ -f "$M/config.json" ] || { say "SKIP $N: $M missing"; return 1; }
  for cfg in "$@"; do
    stop_port $P; free_gpu $G
    conda activate crag-vllm
    VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$G nohup vllm serve $M --served-model-name $N --host 127.0.0.1 --port $P --seed 0 \
      --enforce-eager --max-num-seqs 16 $cfg > $PROJ/logs/vllm_${N}_$P.log 2>&1 &
    local pid=$! ok=0; conda activate crag
    for i in $(seq 1 360); do
      [ "$(served $P)" = "$N" ] && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 5
    done
    if [ $ok = 1 ]; then
      local ml; ml=$(echo "$cfg" | grep -o "max-model-len [0-9]*" | awk '{print $2}')
      local pf; pf=$(python -W ignore -m coalrag.tools.preflight $N http://127.0.0.1:$P/v1 $ml 2>&1 | tail -1)
      say "$N on GPU $G :$P [$cfg] | $(grep -h 'GPU blocks\|KV cache size' $PROJ/logs/vllm_${N}_$P.log | tail -1 | sed 's/.*INFO[^]]*] //') | $pf"
      echo "$pf" | grep -q " 8/8 " && return 0
    else
      say "$N failed to start [$cfg]: $(grep -hE 'Error|error' $PROJ/logs/vllm_${N}_$P.log | tail -1 | cut -c1-160)"
    fi
  done
  stop_port $P; say "SERVE FAILED $N (all configurations)"; return 1
}
e4main() {  # $1 port, $2 served name, $3 concurrency
  say "start $2 main (pre-registered e4 runner)"
  for pass in 1 2; do python -W ignore -m coalrag.experiments.e4_llm --served $2 --base http://127.0.0.1:$1/v1 --tag main --conc $3 >> $PROJ/logs/e4_$2_main.log 2>&1; done
  say "end   $2 main | $(cov $2 main)"
}
e5() {  # $1 port, $2 served name, $3 tag, rest = e5_wave2 args; two passes so transient errors are retried
  local P=$1 N=$2 T=$3; shift 3; say "start $N $T $*"
  for pass in 1 2; do python -W ignore -m coalrag.experiments.e5_wave2 --served $N --base http://127.0.0.1:$P/v1 --tag $T "$@" >> $PROJ/logs/e5_$N.log 2>&1; done
  say "end   $N $T | $(cov $N $T)"
}

stop() {
  pkill -u $USER -f "gpuq2?\.sh (lane|tp2)"; pkill -u $USER -f "queue_wave2.sh"
  pkill -u $USER -f "coalrag.experiments.e[45]"; sleep 3
  stop_port 8001; stop_port 8007; free_gpu 7,9
  for g in 7 9; do echo "GPU $g in use: $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader)"; done
}
diag() { python -W ignore -m coalrag.tools.diag; }
lane9() {
  local G=9 P=8001 N=qwen2.5-32b-gptq-int8
  serve_chain $G $P $HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8 $N \
    "--gpu-memory-utilization 0.78 --max-model-len 6144 --max-num-batched-tokens 6144" \
    "--gpu-memory-utilization 0.80 --max-model-len 6144 --max-num-batched-tokens 6144" \
    "--gpu-memory-utilization 0.80 --max-model-len 4096 --max-num-batched-tokens 4096" \
    "--gpu-memory-utilization 0.80 --max-model-len 4096 --max-num-batched-tokens 4096 --enable-prefix-caching" || return 1
  e4main $P $N 8
  e5 $P $N allreg --conds AR --queries adopted --conc 8
  e5 $P $N wave2 --conds wave2 --conc 8
}
lane7() {
  local G=7 P=8007 N=mistral-small-24b-2501-fp8
  if serve_chain $G $P $HOME/models/Mistral-Small-24B-Instruct-2501-FP8-dynamic $N \
       "--gpu-memory-utilization 0.75 --max-model-len 8192 --max-num-batched-tokens 8192" \
       "--gpu-memory-utilization 0.70 --max-model-len 6144 --max-num-batched-tokens 6144"; then
    e5 $P $N main --conds main --conc 12
    e5 $P $N wave2 --conds F,M --conc 12
  fi
  N=qwen3-8b
  if serve_chain $G $P $HOME/models/Qwen3-8B $N \
       "--gpu-memory-utilization 0.65 --max-model-len 8192 --max-num-batched-tokens 8192" \
       "--gpu-memory-utilization 0.60 --max-model-len 6144 --max-num-batched-tokens 6144"; then
    e5 $P $N think-off --conds THINK --think off --conc 32
    e5 $P $N think-on --conds THINK --think on --budget 1024 --conc 32
  fi
}
tp2() {
  local G=7,9 P=8007 N=llama-3.3-70b-fp8 M=$HF_MODELS/MetaAI/Llama-3.3-70B-Instruct-FP8 Q=""
  say "waiting for lane9 and lane7 to finish and for every LLM client to exit"
  until grep -q "\[lane9\] lane9 done" $L && grep -q "\[lane7\] lane7 done" $L && ! pgrep -u $USER -f "coalrag.experiments.e[45]" >/dev/null; do sleep 300; done
  if [ -f $M/hf_quant_config.json ]; then Q="--quantization modelopt"
  elif ! python -c "import json,sys; sys.exit(0 if json.load(open('$M/config.json')).get('quantization_config') else 1)"; then
    say "SKIP $N: neither hf_quant_config.json (ModelOpt) nor quantization_config found"; return 1; fi
  stop_port 8001
  serve_chain $G $P $M $N \
    "--tensor-parallel-size 2 --gpu-memory-utilization 0.82 --max-model-len 6144 --max-num-batched-tokens 6144 $Q" \
    "--tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-batched-tokens 4096 $Q" || return 1
  e5 $P $N main --conds B0-choi,B0-std,CF-LTorig,D-bge-coal,D-oracle-coal,D-random-coal,D-strat4-coal --conc 8
  e5 $P $N wave2 --conds F --conc 8
}
case $LANE in
  stop|diag) $LANE ;;
  lane9|lane7|tp2) trap 'say "$LANE done"' EXIT; say "started"; $LANE ;;
  *) echo "unknown lane $LANE"; exit 2 ;;
esac
