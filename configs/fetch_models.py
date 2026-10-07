import os, sys
from huggingface_hub import HfApi, snapshot_download
api = HfApi(); DEST = os.path.expanduser("~/models")
FAMILIES = {
  "qwen2.5-32b": ["Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", "Qwen/Qwen2.5-32B-Instruct-AWQ"],
  "mistral-small-24b": ["RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic", "neuralmagic/Mistral-Small-24B-Instruct-2501-FP8-dynamic",
                        "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic"],
}
for fam, cands in FAMILIES.items():
    for repo in cands:
        try:
            info = api.model_info(repo, files_metadata=True)
            gb = sum((s.size or 0) for s in info.siblings) / 1e9
            print(f"[{fam}] FOUND {repo} ({gb:.1f} GB, gated={getattr(info, 'gated', None)})", flush=True)
            if "--download" in sys.argv:
                p = snapshot_download(repo, local_dir=os.path.join(DEST, repo.split("/")[1]),
                                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.jinja"])
                print(f"[{fam}] downloaded -> {p}", flush=True)
            break
        except Exception as e:
            print(f"[{fam}] not available: {repo} ({type(e).__name__})", flush=True)
