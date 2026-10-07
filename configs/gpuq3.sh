#!/usr/bin/env bash
# Two-GPU queue v3 (D73). Single serving standard for every model:
#   vLLM V1 engine, --generation-config vllm (neutral sampling: no model-default repetition penalty),
#   chunked prefill with --max-num-batched-tokens 2048 (bounds the prompt log-prob tensor), --max-model-len 8192,
#   and a preflight that scores 8 near-maximum-length prompts concurrently before any run.
#   bash gpuq3.sh stop    stop all lanes (v1-v3), LLM clients and this user's servers on GPUs 7 and 9
#   bash gpuq3.sh diag    coverage, errors, prompt lengths, Llama format, template audit
#   bash gpuq3.sh lane9   GPU 9 :8001  32B main (pre-registered e4 runner) -> 32B all-regime -> 32B wave 2
#   bash gpuq3.sh lane7   GPU 7 :8007  Mistral-24B-2501 main + F,M -> Qwen2.5-7B main + F,M -> Qwen3-8B thinking off/on
#   bash gpuq3.sh tp2     GPUs 7+9 :8007  waits for both lanes, then Llama-3.3-70B-FP8 (reduced set + F)
#   bash gpuq3.sh lane7w  GPU 7 :8007  second 32B server for the 32B wave 2 (D77)
#   bash gpuq3.sh lane6   GPU 6 :8006  E7 selection experiment on Mistral, 7B, 32B; then releases GPU 6 (D81)
#   bash gpuq3.sh tp2b    waits for lane7w and the all-regime client, retries all-regime, then Llama (D77)
set -u
LANE=${1:?usage: gpuq3.sh stop|diag|lane9|lane7|tp2|lane7w|tp2b|lane6}
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate crag
export PROJ=$HOME/coalrag PYTHONPATH=$HOME/coalrag/src CUDA_DEVICE_ORDER=PCI_BUS_ID
L=$PROJ/logs/gpuq3.log
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
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$G nohup vllm serve $M --served-model-name $N --host 127.0.0.1 --port $P --seed 0 \
      --generation-config vllm --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 16 $cfg \
      > $PROJ/logs/vllm_${N}_$P.log 2>&1 &
    local pid=$! ok=0; conda activate crag
    for i in $(seq 1 360); do
      [ "$(served $P)" = "$N" ] && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 5
    done
    if [ $ok = 1 ]; then
      local pf; pf=$(python -W ignore -m coalrag.tools.preflight $N http://127.0.0.1:$P/v1 8192 2>&1 | tail -1)
      say "$N on GPU $G :$P [$cfg] | $(grep -h 'KV cache size' $PROJ/logs/vllm_${N}_$P.log | tail -1 | sed 's/.*INFO[^]]*] //') | $pf"
      echo "$pf" | grep -q " 8/8 " && return 0
      say "  server log tail: $(grep -hE 'Error|error' $PROJ/logs/vllm_${N}_$P.log | tail -1 | cut -c1-200)"
    else
      say "$N failed to start [$cfg]: $(grep -hE 'Error|error' $PROJ/logs/vllm_${N}_$P.log | tail -1 | cut -c1-200)"
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

e7() {  # $1 port, $2 served name: E7 selection experiment (D81), two passes
  say "start $2 e7"
  for pass in 1 2; do python -W ignore -m coalrag.experiments.e7_select --served $2 --base http://127.0.0.1:$1/v1 --conc 16 >> $PROJ/logs/e7_$2.log 2>&1; done
  say "end   $2 e7 | $(grep -E '^wall' $PROJ/logs/e7_$2.log | tail -1)"
}
lane6() {  # D81: borrowed GPU 6 (port 8006) for E7 on Mistral-24B, Qwen2.5-7B, Qwen2.5-32B; releases GPU 6 at the end
  local G=6 P=8006
  serve_chain $G $P $HOME/models/Mistral-Small-24B-Instruct-2501-FP8-dynamic mistral-small-24b-2501-fp8 \
    "--gpu-memory-utilization 0.85" "--gpu-memory-utilization 0.80 --enforce-eager" && e7 $P mistral-small-24b-2501-fp8
  serve_chain $G $P $HF_MODELS/Qwen/Qwen2.5-7B-Instruct qwen2.5-7b \
    "--gpu-memory-utilization 0.80" "--gpu-memory-utilization 0.75 --enforce-eager" && e7 $P qwen2.5-7b
  serve_chain $G $P $HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8 qwen2.5-32b-gptq-int8 \
    "--gpu-memory-utilization 0.85" "--gpu-memory-utilization 0.80 --enforce-eager" && e7 $P qwen2.5-32b-gptq-int8
  stop_port $P; free_gpu $G; say "GPU 6 released"
}
stop() {
  pkill -u $USER -f "gpuq[23]?\.sh (lane|tp2)"; pkill -u $USER -f "queue_wave2.sh"
  pkill -u $USER -f "coalrag.experiments.e[457]"; pkill -u $USER -f "coalrag.tools.preflight"; sleep 3
  stop_port 8001; stop_port 8006; stop_port 8007; free_gpu 6,7,9
  for g in 7 9; do echo "GPU $g in use: $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader)"; done
}
diag() { python -W ignore -m coalrag.tools.diag; }
lane9() {
  local G=9 P=8001 N=qwen2.5-32b-gptq-int8
  serve_chain $G $P $HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8 $N \
    "--gpu-memory-utilization 0.85" "--gpu-memory-utilization 0.80 --enforce-eager" || return 1
  e4main $P $N 12
  e5 $P $N allreg --conds AR --queries adopted --conc 16
  e5 $P $N wave2 --conds wave2 --conc 12
}
lane7() {
  local G=7 P=8007 N=mistral-small-24b-2501-fp8
  if serve_chain $G $P $HOME/models/Mistral-Small-24B-Instruct-2501-FP8-dynamic $N \
       "--gpu-memory-utilization 0.85" "--gpu-memory-utilization 0.80 --enforce-eager"; then
    e5 $P $N main --conds main --conc 16
    e5 $P $N wave2 --conds F,M --conc 16
  fi
  N=qwen2.5-7b
  if serve_chain $G $P $HF_MODELS/Qwen/Qwen2.5-7B-Instruct $N \
       "--gpu-memory-utilization 0.80" "--gpu-memory-utilization 0.75 --enforce-eager"; then
    e5 $P $N main --conds main --conc 32
    e5 $P $N wave2 --conds F,M --conc 32
  fi
  N=qwen3-8b
  if serve_chain $G $P $HOME/models/Qwen3-8B $N \
       "--gpu-memory-utilization 0.80" "--gpu-memory-utilization 0.75 --enforce-eager"; then
    e5 $P $N think-off --conds THINK --think off --conc 32
    e5 $P $N think-on --conds THINK --think on --budget 1024 --conc 32
  fi
}
llama() {  # Llama-3.3-70B-FP8 (ModelOpt) across GPUs 7+9, reduced condition set + F
  local G=7,9 P=8007 N=llama-3.3-70b-fp8 M=$HF_MODELS/MetaAI/Llama-3.3-70B-Instruct-FP8
  [ -f $M/hf_quant_config.json ] || { say "SKIP $N: no hf_quant_config.json"; return 1; }
  stop_port 8001; stop_port 8007
  serve_chain $G $P $M $N \
    "--tensor-parallel-size 2 --quantization modelopt --gpu-memory-utilization 0.88" \
    "--tensor-parallel-size 2 --quantization modelopt --gpu-memory-utilization 0.85 --enforce-eager" || return 1
  e5 $P $N main --conds B0-choi,B0-std,CF-LTorig,D-bge-coal,D-oracle-coal,D-random-coal,D-strat4-coal --conc 12
  e5 $P $N wave2 --conds F --conc 12
}
tp2() {
  say "waiting for lane9 and lane7 to finish and for every LLM client to exit"
  until grep -q "\[lane9\] lane9 done" $L && grep -q "\[lane7\] lane7 done" $L && ! pgrep -u $USER -f "coalrag.experiments.e[45]" >/dev/null; do sleep 300; done
  llama
}
lane7w() {  # D77: second 32B server on GPU 7 takes the 32B wave 2 while GPU 9 finishes the all-regime run
  local G=7 P=8007 N=qwen2.5-32b-gptq-int8
  serve_chain $G $P $HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8 $N \
    "--gpu-memory-utilization 0.85" "--gpu-memory-utilization 0.80 --enforce-eager" || return 1
  e5 $P $N wave2 --conds wave2 --conc 12
}
tp2b() {  # waits for lane7w and for no LLM client in two checks 5 minutes apart; retries the all-regime run; then Llama
  say "waiting for lane7w to finish and for every LLM client (including the all-regime run on :8001) to exit"
  local quiet=0
  until [ $quiet -ge 2 ]; do
    sleep 300
    if grep -q "\[lane7w\] lane7w done" $L && ! pgrep -u $USER -f "coalrag.experiments.e[45]" >/dev/null; then quiet=$((quiet+1)); else quiet=0; fi
  done
  [ "$(served 8001)" = "qwen2.5-32b-gptq-int8" ] && e5 8001 qwen2.5-32b-gptq-int8 allreg --conds AR --queries adopted --conc 16
  llama
}
case $LANE in
  stop|diag) $LANE ;;
  lane9|lane7|tp2|lane7w|tp2b|lane6) trap 'say "$LANE done"' EXIT; say "started"; $LANE ;;
  *) echo "unknown lane $LANE"; exit 2 ;;
esac
