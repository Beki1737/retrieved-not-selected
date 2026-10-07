set -eo pipefail
PROJ=$HOME/coalrag
if ! command -v conda >/dev/null 2>&1 && [ ! -x $HOME/miniforge3/bin/conda ]; then
  cd /tmp && curl -fsSL -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3.sh -b -p $HOME/miniforge3
fi
if [ -x $HOME/miniforge3/bin/conda ]; then CONDA_BASE=$HOME/miniforge3; else CONDA_BASE=$(conda info --base); fi
source $CONDA_BASE/etc/profile.d/conda.sh
IDX=https://download.pytorch.org/whl/cu124

echo "### [1/2] crag (analysis)"
conda create -y -n crag -c conda-forge --override-channels python=3.11
conda activate crag
pip install torch==2.6.0 --index-url $IDX
pip install numpy==2.1.3 pandas==2.2.3 transformers==4.51.1 sentence-transformers==3.4.1 \
  "pyarrow>=17" "scipy>=1.13" "scikit-learn>=1.5,<1.7" "statsmodels>=0.14.4" "lightgbm>=4.5" \
  rank-bm25 "openai>=1.60" tqdm pyyaml "matplotlib>=3.9" ipykernel pyreadr
pip freeze > $PROJ/configs/lock-crag.txt
python -c "import torch,numpy,pandas,transformers,sentence_transformers,lightgbm,sklearn;print('CRAG_OK',torch.__version__,torch.cuda.is_available(),numpy.__version__,pandas.__version__,transformers.__version__,sentence_transformers.__version__)"
conda deactivate

echo "### [2/2] crag-vllm (serving only)"
conda create -y -n crag-vllm -c conda-forge --override-channels python=3.11
conda activate crag-vllm
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url $IDX
pip install vllm==0.8.5
pip install transformers==4.51.1
pip freeze > $PROJ/configs/lock-vllm.txt
python -c "import torch,vllm,transformers;print('VLLM_OK',torch.__version__,vllm.__version__,transformers.__version__)"
echo "### ALL DONE"
