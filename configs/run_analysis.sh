#!/usr/bin/env bash
# CPU analysis of every completed LLM run (D76). Writes logs/analysis/*.log and one combined ALL_<time>.txt to send back.
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate crag
export PROJ=$HOME/coalrag PYTHONPATH=$HOME/coalrag/src
A=$PROJ/logs/analysis; mkdir -p $A
for m in qwen2.5-32b-gptq-int8 mistral-small-24b-2501-fp8 qwen2.5-7b llama-3.3-70b-fp8; do
  f=$PROJ/results/store/e4_${m}_main.jsonl; [ -f $f ] || continue
  echo "[$(date +%H:%M)] $m"
  python -W ignore -m coalrag.analysis.e4_metrics $f --cad --B 2000 --by qset > $A/metrics_$m.log 2>&1
  python -W ignore -m coalrag.analysis.e4_hda $f > $A/hda_$m.log 2>&1
  python -W ignore -m coalrag.analysis.e5_analysis $m --B 1000 > $A/e5_$m.log 2>&1
done
[ -f $PROJ/results/store/e4_qwen3-8b_think-on.jsonl ] && python -W ignore -m coalrag.analysis.e5_analysis qwen3-8b --B 1000 > $A/e5_qwen3-8b.log 2>&1
python -W ignore -m coalrag.analysis.summary --B 2000 > $A/summary.log 2>&1
O=$A/ALL_$(date +%m%d_%H%M).txt
for f in $A/summary.log $A/metrics_*.log $A/e5_*.log $A/hda_*.log; do echo "########## $(basename $f)"; cat $f; done > $O
echo "done: $O ($(wc -l < $O) lines)"
