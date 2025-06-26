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
import igraph as ig
import leidenalg
import community as community_louvain

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

def embed_table(input_file: pathlib.Path, host: str, model: str, out_file: pathlib.Path = None, skip_rows: int = 0, keywords_only: bool = False):
    records = read_table(input_file, skip_rows)
    if not records:
        print("No records found in table."); return
    logging.info("Found %d records", len(records))
    files, texts = [], []
    if keywords_only:
        for i, rec in enumerate(records):
            kw = rec.get("Keywords", "")
            if isinstance(kw, float):  # handle NaN
                kw = ""
            kw = kw.strip()
            if kw:
                text = kw
            else:
                text = rec.get("Paper Title", "").strip()
            if not text:
                continue
            files.append(f"row_{i+1}")
            texts.append(text)
    else:
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
    print(f"✅ embeddings saved → {out_file}")
    print(f"\nNext step: Run community detection:\n  python vectormap_table.py community '{out_file}' --algorithm leiden --k 5\n")

##############################################################################
# 3) CLUSTERING + NAMING (offline step)                                      #
##############################################################################
P_TEMPLATE = (
    "You are an expert research librarian.\n"
    "Below are titles and Abstracts of papers grouped by semantic similarity.\n"
    "Provide ONE short (≤5‑word) theme capturing the common topic.\n"
    "Respond with ONLY that theme.\n"
    "Papers:\n{digest}\nTheme:")

def cluster_digest(files, labels, cid, records, titles_only=False):
    sample = [i for i, l in enumerate(labels) if l == cid][:10]
    parts = []
    for idx in sample:
        rec = records[idx]
        title = rec["Paper Title"][:150]
        if titles_only:
            parts.append(f"- **{title}**")
        else:
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
    export_xlsx: bool = False,
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
    if export_xlsx:
        df = pd.DataFrame(records)
        cluster_names = [names.get(l, "Noise") for l in labels]
        df["Cluster Name"] = cluster_names
        out_xlsx = cache.with_suffix("").as_posix() + "_clusters.xlsx"
        df.to_excel(out_xlsx, index=False)
        print(f"Cluster assignments exported to {out_xlsx}")

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
    print(f"\nWorkflow complete! If you want to try different clustering parameters, rerun the community step. If you want to re-embed, rerun the embed step.\n")

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
  map    <embeddings.joblib> → open interactive map (uses cached names)
  community <cache>          → compute community detection labels & names
  refine <cache>            → refine cluster assignments"""
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
    p_embed.add_argument("--keywords-only", action="store_true", help="Use only keywords for embedding (fall back to title if empty)")

    # label
    p_lab = sub.add_parser("label")
    p_lab.add_argument("cache", type=pathlib.Path)
    p_lab.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    p_lab.add_argument("--llm-host", default="http://localhost:11434", help="Ollama host (default: http://localhost:11434)")
    p_lab.add_argument("--llm-model", default="gemma3:4b")
    p_lab.add_argument("--metric", choices=["euclidean", "cosine"], default="cosine", help="Distance metric for DBSCAN (default: cosine)")
    p_lab.add_argument("--export-xlsx", action="store_true", help="Export an XLSX file with cluster names for each paper")

    # map
    p_map = sub.add_parser("map")
    p_map.add_argument("cache", type=pathlib.Path)
    p_map.add_argument("--viewer", choices=["default", "obsidian"], default="default")
    p_map.add_argument("--eps", type=float, default=0.3)
    p_map.add_argument("--min-samples", type=int, default=3)

    # community
    p_comm = sub.add_parser("community")
    p_comm.add_argument("cache", type=pathlib.Path)
    p_comm.add_argument("--algorithm", choices=["leiden", "louvain"], default="leiden")
    p_comm.add_argument("--k", type=int, default=15, help="Number of neighbors for k-NN graph")
    p_comm.add_argument("--titles-only", action="store_true", help="Use only paper titles for clustering (not abstracts)")
    p_comm.add_argument("--keywords-only", action="store_true", help="Use only keywords for clustering (fall back to title if empty)")
    p_comm.add_argument("--export-xlsx", action="store_true", help="Export an XLSX file with cluster names for each paper (community workflow)")

    # refine
    p_refine = sub.add_parser("refine")
    p_refine.add_argument("cache", type=pathlib.Path)
    p_refine.add_argument("--llm-host", default="http://localhost:11434", help="Ollama host (default: http://localhost:11434)")
    p_refine.add_argument("--llm-model", default="gemma3:4b")
    p_refine.add_argument("--export-xlsx", action="store_true", help="Export an XLSX file with refined cluster names for each paper")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "embed":
        embed_table(args.table, args.host, args.model, args.out, args.skip_rows, keywords_only=getattr(args, 'keywords_only', False))
    elif args.cmd == "label":
        label_cache(
            args.cache,
            min_cluster_size=args.min_cluster_size,
            llm_host=args.llm_host,
            llm_model=args.llm_model,
            metric=args.metric,
            min_samples=getattr(args, 'min_samples', None),
            export_xlsx=getattr(args, 'export_xlsx', False),
        )
    elif args.cmd == "map":
        map_embeddings(
            args.cache,
            viewer=args.viewer,
            eps=args.eps,
            min_samples=args.min_samples,
        )
    elif args.cmd == "community":
        community_detection(
            args.cache,
            algorithm=args.algorithm,
            k=args.k,
            titles_only=getattr(args, 'titles_only', False),
            keywords_only=getattr(args, 'keywords_only', False),
            export_xlsx=getattr(args, 'export_xlsx', False),
        )
    elif args.cmd == "refine":
        refine_clusters(
            args.cache,
            llm_host=args.llm_host,
            llm_model=args.llm_model,
            export_xlsx=getattr(args, 'export_xlsx', False),
        )

def community_detection(cache: pathlib.Path, algorithm: str = 'leiden', k: int = 15, titles_only: bool = False, keywords_only: bool = False, export_xlsx: bool = False):
    data = joblib.load(cache)
    X, files, records = data["embeddings"], data["files"], data["records"]
    if keywords_only:
        # Re-embed only the keywords (fall back to title if empty)
        from sklearn.preprocessing import normalize
        import requests
        texts = []
        for rec in records:
            kw = rec.get("Keywords", "")
            if isinstance(kw, float):  # handle NaN
                kw = ""
            kw = kw.strip()
            if kw:
                texts.append(kw)
            else:
                title = rec.get("Paper Title", "").strip()
                texts.append(title)
        # Use the same embedding host/model as before if possible
        host = 'http://localhost:11434'
        model = 'mxbai-embed-large'
        try:
            host = data.get('embedding_host', host)
            model = data.get('embedding_model', model)
        except Exception:
            pass
        def post_embeddings(host, model, texts):
            r = requests.post(host.rstrip("/") + "/api/embed",
                              json={"model": model, "input": texts},
                              headers={"Content-Type": "application/json"}, timeout=120)
            r.raise_for_status()
            return r.json().get("embeddings", r.json().get("embedding"))
        BATCH = 16
        vecs = []
        for i in range(0, len(texts), BATCH):
            vecs.extend(post_embeddings(host, model, texts[i:i+BATCH]))
        X = normalize(np.vstack(vecs, dtype=np.float32))
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
    knn_graph = nbrs.kneighbors_graph(X, mode='connectivity')
    sources, targets = knn_graph.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(edges=edges, directed=False)
    g.vs['name'] = list(range(len(files)))
    # Community detection
    if algorithm == 'leiden':
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    elif algorithm == 'louvain':
        partition = g.community_multilevel()
    else:
        raise ValueError('Unknown algorithm: ' + algorithm)
    labels = [-1] * len(files)
    for cid, cluster in enumerate(partition):
        for idx in cluster:
            labels[idx] = cid
    # Name clusters using LLM if available
    names: dict[int, str] = {}
    llm_host = 'http://localhost:11434'  # default, can be made configurable
    llm_model = 'gemma3:4b'  # default, can be made configurable
    for cid in set(labels):
        if cid == -1:
            names[cid] = 'Noise'
            continue
        digest = cluster_digest(files, labels, cid, records, titles_only=titles_only)
        try:
            names[cid] = name_cluster(digest, llm_host, llm_model)
        except Exception as e:
            logging.warning("LLM naming failed for cluster %s – %s", cid, e)
            names[cid] = f"Cluster {cid}"
    data["labels"] = labels
    data["cluster_names"] = names
    joblib.dump(data, cache)
    n_clusters = len([cid for cid in set(labels) if cid != -1])
    n_noise = labels.count(-1)
    print(f"✅ Community detection labels & names stored → {cache}")
    print(f"Clusters found: {n_clusters} (noise points: {n_noise})")
    if export_xlsx:
        df = pd.DataFrame(records)
        cluster_names = [names.get(l, "Noise") for l in labels]
        df["Cluster Name"] = cluster_names
        out_xlsx = cache.with_suffix("").as_posix() + "_clusters.xlsx"
        df.to_excel(out_xlsx, index=False)
        print(f"Cluster assignments exported to {out_xlsx}")
    print(f"\nNext step: Visualize the clusters:\n  python vectormap_table.py map '{cache}'\n")

def refine_clusters(cache: pathlib.Path, llm_host: str, llm_model: str, export_xlsx: bool = False):
    import pandas as pd
    from tqdm import tqdm
    data = joblib.load(cache)
    records = data["records"]
    cluster_names = data.get("cluster_names", {})
    labels = data.get("labels", [])
    # Build a list of unique cluster names (excluding noise)
    unique_clusters = [cid for cid in set(labels) if cid != -1]
    name_map = {cid: cluster_names[cid] for cid in unique_clusters}
    name_list = [name_map[cid] for cid in unique_clusters]
    # For each paper, ask the LLM which cluster name fits best
    refined_labels = []
    refined_cluster_names = []
    print(f"Refining cluster assignments for {len(records)} papers...")
    for i, rec in enumerate(tqdm(records, desc="Refining")):
        title = rec.get("Paper Title", "").strip()
        abstract = rec.get("Abstract", "").strip()
        keywords = rec.get("Keywords", "").strip() if "Keywords" in rec else ""
        prompt = (
            f"You are an expert research librarian.\n"
            f"Given the following paper metadata, choose the best fitting cluster name from the list below.\n"
            f"Paper Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Keywords: {keywords}\n"
            f"Cluster names: {', '.join(name_list)}\n"
            f"Respond with ONLY the best fitting cluster name from the list above."
        )
        pay = {
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
        }
        import requests
        r = requests.post(llm_host.rstrip("/") + "/api/chat", json=pay, timeout=120)
        r.raise_for_status()
        label = r.json()["message"]["content"].strip()
        # Find the cluster id for the chosen name
        cid = None
        for k, v in name_map.items():
            if v.lower() == label.lower():
                cid = k
                break
        if cid is None:
            cid = -1  # If not found, assign as noise
        refined_labels.append(cid)
        refined_cluster_names.append(label)
        print(f"Paper {i+1}/{len(records)}: '{title[:60]}' → {label}")
    data["refined_labels"] = refined_labels
    data["refined_cluster_names"] = {cid: name_map.get(cid, "Noise") for cid in set(refined_labels) if cid != -1}
    joblib.dump(data, cache)
    print(f"✅ Refined cluster assignments stored in {cache}")
    if export_xlsx:
        df = pd.DataFrame(records)
        cluster_names_out = [name_map.get(cid, "Noise") if cid != -1 else "Noise" for cid in refined_labels]
        df["Refined Cluster Name"] = cluster_names_out
        out_xlsx = cache.with_suffix("").as_posix() + "_refined_clusters.xlsx"
        df.to_excel(out_xlsx, index=False)
        print(f"Refined cluster assignments exported to {out_xlsx}")
    print(f"\nNext step: Visualize the refined clusters:\n  python vectormap_table.py map '{cache}'\n")

if __name__ == "__main__":
    main() 