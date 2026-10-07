import os, re, json
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from coalrag.retrieval.lexical import doc_text
from coalrag.legacy_io import load_pickle

PROJ = Path(os.environ["PROJ"]); PROC = PROJ / "data" / "processed"; DS = PROJ / "data" / "raw" / "Nation-Level_Bias" / "Dataset"
MODELS = {"bge-small": "BAAI/bge-small-en-v1.5",
          "mxbai-large": "$HF_MODELS/MixedBread-AI/mxbai-embed-large-v1"}
ckg = pd.read_parquet(PROC / "ckg_v1_1.parquet")
sub = ckg[ckg.in_choi_test | ckg.in_choi_adopted].reset_index(drop=True)
texts = [doc_text(r) for r in sub.to_dict("records")]
print("docs:", len(texts), "| empty texts:", sum(not t for t in texts), "| mean chars:", int(np.mean([len(t) for t in texts])))
for name, path in MODELS.items():
    m = SentenceTransformer(path, device="cuda")
    E = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(PROC / f"emb_configA_{name}.npy", E); print(f"{name}: {E.shape}")
sub[["res_id"]].to_parquet(PROC / "emb_configA_index.parquet", index=False)

# fidelity check: our bge-small vs the legacy Colab embeddings (same text, same model?)
L = load_pickle(next((PROJ / "legacy").rglob("dense_embeddings_v1.pkl")))
E = np.load(PROC / "emb_configA_bge-small.npy"); pos = {r: i for i, r in enumerate(sub.res_id)}
ca = json.load(open(DS / "adopted_resolutions_dataset.json", encoding="utf-8"))
cn = json.load(open(DS / "not_adopted_resolutions_dataset.json", encoding="utf-8"))
rn = lambda s: int((re.search(r"RES/\s*(\d+)", str(s)) or re.search(r"(\d+)", str(s))).group(1))
num2rid = dict(zip(ckg[ckg.adopted].resnum, ckg[ckg.adopted].res_id)); sym2rid = dict(zip(ckg.symbol_norm, ckg.res_id))
unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
ca_cos = [float(unit(L["adopted_emb"][i]) @ E[pos[num2rid[rn(r["res_no"])]]]) for i, r in enumerate(ca)]
cn_cos = [float(unit(L["nonadopted_emb"][i]) @ E[pos[sym2rid[re.sub(r"\s+", "", r["res_no"]).upper()]]]) for i, r in enumerate(cn)]
print(f"legacy-vs-new bge-small cosine: adopted mean={np.mean(ca_cos):.4f} min={np.min(ca_cos):.4f} | "
      f"non-adopted mean={np.mean(cn_cos):.4f} min={np.min(cn_cos):.4f}")
