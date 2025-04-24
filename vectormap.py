#!/usr/bin/env python3
"""
vectormap.py – folder-level document embedding & 2-D interactive map

author: you
license: MIT
"""
import argparse, json, os, sys, textwrap, pathlib, re, logging
from typing import List, Tuple

import joblib, numpy as np, requests, tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
try:
    import umap  # optional but preferred
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

try:
    from pypdf import PdfReader            # PyPDF2 >= 3.x
except ImportError:
    from PyPDF2 import PdfReader           # fallback for PyPDF2 < 3.x

# ---------- CONFIG ---------- #
BATCH = 16                                 # #docs per embed call
TIMEOUT = 120                              # seconds
HEADERS = {"Content-Type": "application/json"}
# ---------------------------- #

def iter_files(root: pathlib.Path) -> List[pathlib.Path]:
    for p in root.rglob("*"):
        if p.suffix.lower() in {".md", ".pdf"} and p.is_file():
            yield p

def read_markdown(p: pathlib.Path) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    # strip YAML front-matter
    return re.sub(r"^---.*?---\s*", "", txt, flags=re.S)

def read_pdf(p: pathlib.Path) -> str:
    reader = PdfReader(str(p))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_text(p: pathlib.Path) -> str:
    return read_markdown(p) if p.suffix.lower() == ".md" else read_pdf(p)

def post_embeddings(host: str, model: str, texts: List[str]) -> List[List[float]]:
    url = host.rstrip("/") + "/api/embed"
    payload = {"model": model, "input": texts}
    r = requests.post(url, data=json.dumps(payload), headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    key = "embeddings" if "embeddings" in r.json() else "embedding"
    return r.json()[key]

def embed_folder(path: pathlib.Path, host: str, model: str, out_file: pathlib.Path):
    files, texts = [], []
    logging.info("Scanning files…")
    for p in iter_files(path):
        files.append(str(p))
        texts.append(load_text(p))
    logging.info("Found %d documents", len(files))

    # batch-wise embedding
    vectors = []
    for i in tqdm.tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch = texts[i : i + BATCH]
        vectors.extend(post_embeddings(host, model, batch))

    arr = np.vstack(vectors).astype(np.float32)
    joblib.dump({"files": files, "embeddings": arr}, out_file)
    print(f"Saved vectors → {out_file}")

def map_embeddings(cache: pathlib.Path, min_samples=3, eps=0.3):
    data = joblib.load(cache)
    X, paths = data["embeddings"], data["files"]

    reducer_name = "UMAP" if HAVE_UMAP else "t-SNE"
    red = (umap.UMAP(n_components=2, metric="cosine")
           if HAVE_UMAP else
           TSNE(n_components=2, metric="cosine", init="random", perplexity=30, n_iter=1000))
    coords = red.fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(coords)

    import plotly.express as px
    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1],
        color=labels.astype(str),
        hover_name=[pathlib.Path(p).name for p in paths],
        hover_data={"path": paths, "cluster": labels}
    )
    fig.update_layout(title=f"Document map ({reducer_name})", legend_title="cluster")
    fig.show()

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Modes:
              embed <folder>          generate embeddings & save to --out
              map   <embeddings.joblib>  open interactive 2-D map
            """)
    )
    sub = p.add_subparsers(dest="mode", required=True)

    e = sub.add_parser("embed")
    e.add_argument("folder", type=pathlib.Path)
    e.add_argument("--host", default="http://localhost:11434", help="Ollama base URL")
    e.add_argument("--model", default="mxbai-embed-large")
    e.add_argument("--out", type=pathlib.Path, default="embeddings.joblib")

    m = sub.add_parser("map")
    m.add_argument("cache", type=pathlib.Path)
    m.add_argument("--eps", type=float, default=0.3)
    m.add_argument("--min_samples", type=int, default=3)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode == "embed":
        embed_folder(args.folder, args.host, args.model, args.out)
    else:
        map_embeddings(args.cache, args.min_samples, args.eps)

if __name__ == "__main__":
    main()
