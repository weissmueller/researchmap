## VectorMap – visualise document similarity

VectorMap scans a folder (including sub-folders), embeds every **Markdown** and **PDF** file with the `mxbai-embed-large` model served by your local **Ollama** instance, caches the vectors, and opens an interactive 2-D map showing clusters of similar documents.

### Key features
- Embedding via the Ollama `/api/embed` endpoint handled with **Requests** :contentReference[oaicite:0]{index=0}
- Progress bars during long operations thanks to **tqdm** :contentReference[oaicite:1]{index=1}
- Robust PDF text extraction using **pypdf** (preferred) or **PyPDF2** (fallback) :contentReference[oaicite:2]{index=2}
- Fast on-disk caching with **Joblib** :contentReference[oaicite:3]{index=3}
- Vector maths via **NumPy** and machine-learning utilities from **scikit-learn** :contentReference[oaicite:4]{index=4}
- Dimensionality reduction through **UMAP** (or t-SNE) :contentReference[oaicite:5]{index=5}
- Interactive Plotly scatter plot; hover reveals file names and cluster IDs :contentReference[oaicite:6]{index=6}
- Density-based clustering with **DBSCAN** (automatic number of clusters) :contentReference[oaicite:7]{index=7}

### Installation
- Clone the repo or copy `vectormap.py`.
- Ensure Python ≥ 3.9 is available.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
Start or confirm your Ollama server, e.g.:
ollama run mxbai-embed-large
Usage
Step 1 – embed and cache
python vectormap.py embed /path/to/folder \
    --host http://192.168.188.159:11434 \
    --model mxbai-embed-large \
    --out embeddings.joblib
Step 2 – open the map
python vectormap.py map embeddings.joblib
The Plotly viewer launches in your default browser. Use zoom, pan, and lasso selection to explore clusters.
Command-line reference
embed <folder> – generate embeddings, save to --out (default: embeddings.joblib)
map <joblib> – load cached vectors and display 2-D map
Shared options
--host Base URL of the Ollama server (default http://localhost:11434)
--model Embedding model name (default mxbai-embed-large)
Map-only options
--eps DBSCAN neighborhood radius (default 0.3)
--min_samples Minimum points per cluster (default 3)
Extending
Swap DBSCAN for HDBSCAN for larger corpora.
Call fig.write_html("map.html") to save a standalone, shareable HTML file.
Integrate the embeddings with vector databases such as Chroma or LanceDB via LangChain.
License
MIT