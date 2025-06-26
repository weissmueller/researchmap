#!/usr/bin/env python3
"""
vectormap_table.py – embed paper titles and Abstracts from a CSV/XLSX file, cluster, name clusters, and visualise.

Usage
-----
1. **embed  <table>**              → write embeddings.joblib (vectors only)
2. **label  embeddings.joblib**     → run DBSCAN + LLM naming, cache labels
3. **map    embeddings.joblib**     → open interactive map (uses cached labels)

This version reads from a table (CSV/XLSX) with columns 'Paper Title' and 'Abstract'.
"""

from __future__ import annotations

import argparse, json, logging, os, pathlib, textwrap, tempfile, warnings, webbrowser
from typing import List, Dict

import joblib, numpy as np, requests, tqdm, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", message=".*force_all_finite.*renamed.*", category=FutureWarning)

try:
    import umap  # preferred reducer
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import hdbscan
from sklearn.preprocessing import normalize

# ── constants ───────────────────────────────────────────────────────────────
BATCH = 16
TIMEOUT = 120
HEADERS = {"Content-Type": "application/json"}

MIN_CLUSTER_SIZE = 2
MIN_SAMPLES      = 1

##############################################################################
# 1) TABLE INGESTION                                                         #
##############################################################################

def read_table(input_file: pathlib.Path, skip_rows: int = 0) -> List[Dict[str, str]]:
    ext = input_file.suffix.lower()
    # Try to find the header row automatically if skip_rows is 0
    if skip_rows == 0:
        for n_skip in range(5):  # Try first 5 rows
            if ext == ".csv":
                df = pd.read_csv(input_file, skiprows=n_skip)
            elif ext in {".xlsx", ".xls"}:
                df = pd.read_excel(input_file, skiprows=n_skip)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            if set(["Paper Title", "Abstract"]).issubset(df.columns):
                skip_rows = n_skip
                break
        else:
            raise ValueError("Could not find columns 'Paper Title' and 'Abstract' in the first 5 rows.")
    # Now read with the correct skip_rows
    if ext == ".csv":
        df = pd.read_csv(input_file, skiprows=skip_rows)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(input_file, skiprows=skip_rows)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    if not set(["Paper Title", "Abstract"]).issubset(df.columns):
        raise ValueError("Input file must have columns 'Paper Title' and 'Abstract'")
    records = df[["Paper Title", "Abstract"]].fillna("").to_dict(orient="records")
    return records

##############################################################################
# 2) EMBEDDING                                                               #
##############################################################################

def post_embeddings(host: str, model: str, texts: List[str]):
    r = requests.post(host.rstrip("/") + "/api/embed",
                      json={"model": model, "input": texts},
                      headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("embeddings", r.json().get("embedding"))

def embed_table(input_file: pathlib.Path, host: str, model: str, out_file: pathlib.Path = None, skip_rows: int = 0):
    records = read_table(input_file, skip_rows)
    if not records:
        print("No records found in table."); return
    logging.info("Found %d records", len(records))
    files, texts = [], []
    for i, rec in enumerate(records):
        title = rec["Paper Title"].strip()
        Abstract = rec["Abstract"].strip()
        if not title and not Abstract:
            continue
        files.append(f"row_{i+1}")
        texts.append(f"{title}\n\n{Abstract}")
    if not files:
        print("No valid records in table."); return
    vecs = []
    for i in tqdm.tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        vecs.extend(post_embeddings(host, model, texts[i:i+BATCH]))
    # If out_file is None, use input_file with .joblib extension
    if out_file is None:
        out_file = input_file.with_suffix('.joblib')
    joblib.dump({"files": files, "embeddings": np.vstack(vecs, dtype=np.float32), "records": records}, out_file)
    print("✅ embeddings saved →", out_file)

##############################################################################
# 3) CLUSTERING + NAMING (offline step)                                      #
##############################################################################
P_TEMPLATE = (
    "You are an expert research librarian.\n"
    "Below are titles and Abstracts of papers grouped by semantic similarity.\n"
    "Provide ONE short (≤5‑word) theme capturing the common topic.\n"
    "Respond with ONLY that theme.\n"
    "Papers:\n{digest}\nTheme:")

def cluster_digest(files, labels, cid, records):
    sample = [i for i, l in enumerate(labels) if l == cid][:10]
    parts = []
    for idx in sample:
        rec = records[idx]
        title = rec["Paper Title"][:150]
        Abstract = rec["Abstract"][:500]
        parts.append(f"- **{title}**: {Abstract}…")
    return "\n".join(parts)

def name_cluster(digest, host, model):
    pay = {
        "model": model,
        "messages": [{"role": "user", "content": P_TEMPLATE.format(digest=digest)}],
        "stream": False,
        "temperature": 0.2,
    }
    print("LLM request: ", P_TEMPLATE.format(digest=digest))
    r = requests.post(host.rstrip("/") + "/api/chat", json=pay, timeout=TIMEOUT)
    r.raise_for_status()
    label = r.json()["message"]["content"].strip()
    print("LLM label: ", label)
    return label

def label_cache(
    cache: pathlib.Path,
    min_cluster_size: int,
    *,
    llm_host: str,
    llm_model: str,
    metric: str = "euclidean",
    min_samples: int | None = None,
):
    data = joblib.load(cache)
    X, files, records = data["embeddings"], data["files"], data["records"]
    if metric == "cosine":
        X_clust = normalize(X)
        metric_used = "euclidean"
    else:
        X_clust = X
        metric_used = metric
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric_used,
        cluster_selection_method="eom",
        prediction_data=False,
    ).fit(X_clust)
    labels = clusterer.labels_
    names: dict[int, str] = {}
    if llm_host:
        for cid in set(labels):
            if cid == -1:
                names[cid] = "Noise"
                continue
            digest = cluster_digest(files, labels, cid, records)
            try:
                names[cid] = name_cluster(digest, llm_host, llm_model)
            except Exception as e:
                logging.warning("LLM naming failed for cluster %s – %s", cid, e)
                names[cid] = f"Cluster {cid}"
    data["labels"] = labels
    data["cluster_names"] = names
    joblib.dump(data, cache)
    n_clusters = len([cid for cid in set(labels) if cid != -1])
    n_noise = list(labels).count(-1)
    print(f"✅ HDBSCAN labels & names stored → {cache}")
    print(f"Clusters found: {n_clusters} (noise points: {n_noise})")

##############################################################################
# 4) VISUALISATION (fast)                                                    #
##############################################################################
def map_embeddings(cache: pathlib.Path, *, viewer="default",
                   eps: float = 0.3, min_samples: int = 3):
    data = joblib.load(cache)
    X, files = data["embeddings"], data["files"]
    records = data.get("records")
    if records:
        paper_titles = [rec["Paper Title"] for rec in records]
    else:
        paper_titles = files
    if "labels" in data:
        labels = np.asarray(data["labels"], dtype=int)
        name_map: dict[int, str] = data.get("cluster_names", {})
        color_vals = [name_map.get(c, str(c)) for c in labels]
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric="euclidean").fit_predict(X)
        color_vals = labels.astype(str)
    reducer = (
        umap.UMAP(n_components=2, metric="cosine")
        if HAVE_UMAP else
        TSNE(n_components=2, metric="cosine", init="random", perplexity=30)
    )
    coords = reducer.fit_transform(X)
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=color_vals,
        hover_name=paper_titles,
        hover_data={"cluster": color_vals, "title": paper_titles},
    )
    fig.update_traces(customdata=files)
    js = f"""
    const viewer = '{viewer}';
    const toURI = p => viewer === 'obsidian'
        ? 'obsidian://open?path=' + encodeURIComponent(p)
        : 'file:///' + encodeURIComponent(p);
    document.getElementById('{{plot_id}}')
      .on('plotly_click', ev => window.location.href = toURI(ev.points[0].customdata));
    """
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as tmp:
        tmp.write(
            pio.to_html(
                fig,
                include_plotlyjs="cdn",
                full_html=True,
                post_script=[js],
            )
        )
    webbrowser.open(f"file://{tmp.name}")

##############################################################################
# 5) COMMAND-LINE INTERFACE                                                  #
##############################################################################
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Commands:
  embed  <table>             → generate embeddings & cache vectors
  label  <embeddings.joblib> → compute DBSCAN + name clusters once
  map    <embeddings.joblib> → open interactive map (uses cached names)"""
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # embed
    p_embed = sub.add_parser("embed")
    p_embed.add_argument("table", type=pathlib.Path)
    p_embed.add_argument("--host", default="http://localhost:11434")
    p_embed.add_argument("--model", default="mxbai-embed-large")
    p_embed.add_argument("--out", type=pathlib.Path, default=None)
    p_embed.add_argument("--skip-rows", type=int, default=0, help="Skip the first n rows before processing the table")

    # label
    p_lab = sub.add_parser("label")
    p_lab.add_argument("cache", type=pathlib.Path)
    p_lab.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    p_lab.add_argument("--llm-host", default="http://localhost:11434", help="Ollama host (default: http://localhost:11434)")
    p_lab.add_argument("--llm-model", default="gemma3:4b")
    p_lab.add_argument("--metric", choices=["euclidean", "cosine"], default="cosine", help="Distance metric for DBSCAN (default: cosine)")

    # map
    p_map = sub.add_parser("map")
    p_map.add_argument("cache", type=pathlib.Path)
    p_map.add_argument("--viewer", choices=["default", "obsidian"], default="default")
    p_map.add_argument("--eps", type=float, default=0.3)
    p_map.add_argument("--min-samples", type=int, default=3)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "embed":
        embed_table(args.table, args.host, args.model, args.out, args.skip_rows)
    elif args.cmd == "label":
        label_cache(
            args.cache,
            min_cluster_size=args.min_cluster_size,
            llm_host=args.llm_host,
            llm_model=args.llm_model,
            metric=args.metric,
        )
    else:  # map
        map_embeddings(
            args.cache,
            viewer=args.viewer,
            eps=args.eps,
            min_samples=args.min_samples,
        )

if __name__ == "__main__":
    main() 