#!/usr/bin/env bash
# D87: rerun E7 for Mistral under the D73 serving standard on GPU 8, then finish E7 for Qwen-32B there (resumes).
set -u
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate crag
export PROJ=$HOME/coalrag PYTHONPATH=$HOME/coalrag/src CUDA_DEVICE_ORDER=PCI_BUS_ID
L=$PROJ/logs/gpu8_e7.log; say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
serve8() {  # $1 checkpoint, $2 served name; D73 standard + 8-prompt preflight
  pkill -u $USER -f "vllm serve.*--port 8008( |$)"; sleep 20
  conda activate crag-vllm
  VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=8 nohup vllm serve $1 --served-model-name $2 --host 127.0.0.1 --port 8008 --seed 0 \
    --generation-config vllm --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 16 --gpu-memory-utilization 0.85 \
    > $PROJ/logs/vllm_$2_8008.log 2>&1 &
  local pid=$! i; conda activate crag
  for i in $(seq 1 120); do
    curl -s -m 5 127.0.0.1:8008/v1/models | grep -q "\"$2\"" && break
    kill -0 $pid 2>/dev/null || { say "SERVE FAILED $2: $(grep -hE 'Error' $PROJ/logs/vllm_$2_8008.log | tail -1 | cut -c1-200)"; return 1; }
    sleep 10
  done
  local pf; pf=$(python -W ignore -m coalrag.tools.preflight $2 http://127.0.0.1:8008/v1 8192 2>&1 | tail -1); say "$2 on GPU 8 | $pf"
  echo "$pf" | grep -q " 8/8 "
}
say "started"
Q=$PROJ/results/quarantine/superseded_D87; mkdir -p $Q
for f in e4_mistral-small-24b-2501-fp8_select.jsonl e7_mistral-small-24b-2501-fp8_selector.jsonl; do
  [ -f $PROJ/results/store/$f ] && mv $PROJ/results/store/$f $Q/ && say "quarantined $f (V0 engine, model generation config)"
done
grep -q "^- D87" $PROJ/configs/DECISIONS.md || echo "- D87 The first Mistral E7 run (GPU 8, 04:56-05:21 UTC) used a V0 engine without --generation-config vllm, i.e. not the D73 standard. Its outputs are quarantined (results/quarantine/superseded_D87) and E7 Mistral is rerun under D73 on GPU 8 with the same committed code, still after pre-registration commit cf135a5; the first-run numbers (H13 -0.011, H14 +0.138) are superseded. Qwen-32B E7 moves from the shared GPU 7 server to a dedicated GPU 8 server under the same standard and resumes." >> $PROJ/configs/DECISIONS.md
if serve8 $HOME/models/Mistral-Small-24B-Instruct-2501-FP8-dynamic mistral-small-24b-2501-fp8; then
  for pass in 1 2; do python -W ignore -m coalrag.experiments.e7_select --served mistral-small-24b-2501-fp8 --base http://127.0.0.1:8008/v1 --conc 16 >> $PROJ/logs/e7_mistral_gpu8.log 2>&1; done
  say "Mistral E7: $(tail -1 $PROJ/logs/e7_mistral_gpu8.log)"
fi
if serve8 $HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8 qwen2.5-32b-gptq-int8; then
  pkill -u $USER -f "e7_select --served qwen2.5-32b-gptq-int8 --base http://127.0.0.1:8007"; sleep 5
  for pass in 1 2; do python -W ignore -m coalrag.experiments.e7_select --served qwen2.5-32b-gptq-int8 --base http://127.0.0.1:8008/v1 --conc 12 >> $PROJ/logs/e7_qwen32b_gpu8.log 2>&1; done
  say "32B E7: $(tail -1 $PROJ/logs/e7_qwen32b_gpu8.log)"
fi
pkill -u $USER -f "vllm serve.*--port 8008( |$)"; say "GPU 8 released"
cd $PROJ && git add -A && git -c user.name=user -c user.email=user@lab commit -qm "D87: E7 rerun under D73 on GPU 8" && say "committed $(git log -1 --format=%h)"
