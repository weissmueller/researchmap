#!/usr/bin/env python3
"""
vectormap.py – embed Markdown/PDF files, cluster once, name clusters once, then visualise quickly.

Usage
-----
1. **embed  <folder>**              → write embeddings.joblib (vectors only)
2. **label  embeddings.joblib**     → run DBSCAN + LLM naming, cache labels
3. **map    embeddings.joblib**     → open interactive map (uses cached labels)

That keeps expensive steps (GROBID + Ollama chat) out of the frequently used
`map` command.
"""

from __future__ import annotations

import argparse, json, logging, os, pathlib, re, tempfile, textwrap, warnings, webbrowser, xml.etree.ElementTree as ET
from typing import List, Dict

import joblib, numpy as np, requests, tqdm
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

# ── housekeeping ────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*force_all_finite.*renamed.*", category=FutureWarning)

try:
    import umap  # preferred reducer
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

try:
    from pypdf import PdfReader  # PyPDF2 ≥3.x
except ImportError:
    from PyPDF2 import PdfReader

import plotly.express as px
import plotly.io as pio

# ── constants ───────────────────────────────────────────────────────────────
BATCH = 16
TIMEOUT = 120
HEADERS = {"Content-Type": "application/json"}
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api/processHeaderDocument")

##############################################################################
# 1) FILE INGESTION                                                          #
##############################################################################

def iter_files(root: pathlib.Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in {".md", ".pdf"} and p.is_file():
            yield p

def read_markdown(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_pdf_first_page(p: pathlib.Path) -> str:
    try:
        return PdfReader(str(p)).pages[0].extract_text() or ""
    except Exception:
        return ""

def load_text(path: pathlib.Path) -> str:
    body = read_pdf_first_page(path) if path.suffix.lower() == ".pdf" else read_markdown(path)
    return f"{path.stem}\n\n{body}"


##############################################################################
# 2) METADATA EXTRACTION                                                     #
##############################################################################

def pdf_meta_grobid(path: pathlib.Path) -> dict[str, str]:
    with open(path, "rb") as fh:
        r = requests.post(
            GROBID_URL,
            files={"input": fh},
            data={"consolidateHeader": "1"},
            headers={"Accept": "application/xml"},
            timeout=TIMEOUT,
        )
    if r.status_code != 200 or not r.content:
        raise ValueError(f"GROBID {r.status_code}: {r.text[:120]}")
    root = ET.fromstring(r.content)
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    title = (root.findtext(".//tei:titleStmt/tei:title", namespaces=ns) or path.stem).strip()
    abs_el = root.find(".//tei:profileDesc/tei:abstract", ns)
    abstract = "\n".join(p.text.strip() for p in abs_el.findall("tei:p", ns) if p.text) if abs_el is not None else ""
    doi = (root.findtext(".//tei:idno[@type='DOI']", namespaces=ns) or "").strip()
    return {"title": title[:150], "abstract": abstract, "doi": doi or None}

def pdf_meta_simple(path: pathlib.Path) -> Dict[str, str]:
    txt = read_pdf_first_page(path)
    lines = [l.strip() for l in txt.splitlines() if l.strip()][:10]
    title = lines[0] if lines else path.stem
    abstract = " ".join(lines[1:])[:600]
    return {"title": title, "abstract": abstract}

def md_meta(path: pathlib.Path) -> Dict[str, str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header = next((l[2:].strip() for l in lines if l.startswith("# ")), path.stem)
    abstract = " ".join(lines[1:12])
    return {"title": header, "abstract": abstract}

def extract_metadata(path: pathlib.Path, method: str = "simple") -> Dict[str, str]:
    if path.suffix.lower() == ".pdf":
        if method == "grobid":
            try:
                return pdf_meta_grobid(path)
            except Exception as e:
                logging.warning("GROBID failed for %s – %s; using simple fallback", path, e)
        return pdf_meta_simple(path)
    return md_meta(path)

##############################################################################
# 3) OLLAMA EMBEDDINGS                                                      #
##############################################################################

def post_embeddings(host: str, model: str, texts: List[str]):
    r = requests.post(host.rstrip("/") + "/api/embed",
                      json={"model": model, "input": texts},
                      headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("embeddings", r.json().get("embedding"))

def embed_folder(folder: pathlib.Path, host: str, model: str, out_file: pathlib.Path):
    root = pathlib.Path(str(folder).strip("'\" ")).expanduser()
    if not root.exists():
        print("❌ folder not found:", root); return
    logging.info("Scanning %s …", root)
    files, texts = [], []
    for p in iter_files(root):
        txt = load_text(p)
        if txt.strip():
            files.append(str(p))
            texts.append(txt)
    if not files:
        print("No .md/.pdf files in", root); return
    logging.info("Found %d documents", len(files))
    vecs = []
    for i in tqdm.tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        vecs.extend(post_embeddings(host, model, texts[i:i+BATCH]))
    joblib.dump({"files": files, "embeddings": np.vstack(vecs, dtype=np.float32)}, out_file)
    print("✅ embeddings saved →", out_file)

##############################################################################
# 4) CLUSTERING + NAMING (offline step)                                      #
##############################################################################
P_TEMPLATE = (
    "You are an expert research librarian.\n"
    "Below are titles and abstracts of papers grouped by semantic similarity.\n"
    "Provide ONE short (≤5‑word) theme capturing the common topic.\n"
    "Respond with ONLY that theme.\n"
    "Papers:\n{digest}\nTheme:")

def cluster_digest(paths, labels, cid, meta_source):
    sample = [p for p, l in zip(paths, labels) if l == cid][:10]
    parts = []
    for fp in sample:
        meta = extract_metadata(pathlib.Path(fp), meta_source)
        parts.append(f"- **{meta['title']}**: {meta['abstract'][:500]}…")
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
    eps: float,
    min_samples: int,
    *,
    meta_source: str,
    llm_host: str,
    llm_model: str,
    metric: str,
):
    data = joblib.load(cache)
    X, paths = data["embeddings"], data["files"]

    # cluster with the chosen distance metric
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric               # <-- use it here
    ).fit_predict(X)

    names: dict[int, str] = {}
    if llm_host:
        for cid in set(labels):
            if cid == -1:
                names[cid] = "Noise"
                continue
            digest = cluster_digest(paths, labels, cid, meta_source)
            try:
                names[cid] = name_cluster(digest, llm_host, llm_model)
            except Exception as e:
                logging.warning("LLM naming failed for cluster %s – %s", cid, e)
                names[cid] = f"Cluster {cid}"

    data["labels"] = labels
    data["cluster_names"] = names
    joblib.dump(data, cache)
    print("✅ labels & names stored →", cache)


##############################################################################
# 5) VISUALISATION (fast)                                                    #
##############################################################################

def map_embeddings(cache: pathlib.Path, *, viewer="default",
                   eps: float = 0.3, min_samples: int = 3):
    data = joblib.load(cache)
    X, paths = data["embeddings"], data["files"]

    # ── colours / labels ────────────────────────────────────────────────────
    if "labels" in data:
        labels = np.asarray(data["labels"], dtype=int)
        name_map: dict[int, str] = data.get("cluster_names", {})
        color_vals = [name_map.get(c, str(c)) for c in labels]
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric="euclidean").fit_predict(X)
        color_vals = labels.astype(str)

    # ── dimensionality reduction ───────────────────────────────────────────
    reducer = (
        umap.UMAP(n_components=2, metric="cosine")
        if HAVE_UMAP else
        TSNE(n_components=2, metric="cosine", init="random", perplexity=30)
    )
    coords = reducer.fit_transform(X)
    paths_posix = [pathlib.Path(p).as_posix() for p in paths]

    # ── scatter plot ───────────────────────────────────────────────────────
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=color_vals,
        hover_name=[pathlib.Path(p).name for p in paths],
        hover_data={"cluster": color_vals},
    )
    fig.update_traces(customdata=paths_posix)

    # click-handler JS
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
# 6) COMMAND-LINE INTERFACE                                                  #
##############################################################################

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Commands:
  embed  <folder>             → generate embeddings & cache vectors
  label  <embeddings.joblib>  → compute DBSCAN + name clusters once
  map    <embeddings.joblib>  → open interactive map (uses cached names)"""
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # embed
    p_embed = sub.add_parser("embed")
    p_embed.add_argument("folder", type=pathlib.Path)
    p_embed.add_argument("--host", default="http://localhost:11434")
    p_embed.add_argument("--model", default="mxbai-embed-large")
    p_embed.add_argument("--out", type=pathlib.Path, default="embeddings.joblib")

    # label
    p_lab = sub.add_parser("label")
    p_lab.add_argument("cache", type=pathlib.Path)
    p_lab.add_argument("--eps", type=float, default=0.3)
    p_lab.add_argument("--min-samples", type=int, default=3)
    p_lab.add_argument("--meta-source", choices=["simple", "grobid"], default="simple")
    p_lab.add_argument("--llm-host", required=True)
    p_lab.add_argument("--llm-model", default="gemma3:4b")
    p_lab.add_argument(
    "--metric",
    choices=["euclidean", "cosine"],
    default="cosine",
    help="Distance metric for DBSCAN (default: cosine)",
)


    # map
    p_map = sub.add_parser("map")
    p_map.add_argument("cache", type=pathlib.Path)
    p_map.add_argument("--viewer", choices=["default", "obsidian"], default="default")
    p_map.add_argument("--eps", type=float, default=0.3)
    p_map.add_argument("--min-samples", type=int, default=3)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "embed":
        embed_folder(args.folder, args.host, args.model, args.out)

    elif args.cmd == "label":
        label_cache(
            args.cache, args.eps, args.min_samples,
            meta_source=args.meta_source,
            llm_host=args.llm_host, llm_model=args.llm_model,
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
