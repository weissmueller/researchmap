## ResearchMap – Visualise Document Similarity

VectorMap scans a folder (including sub-folders), embeds every **Markdown** and **PDF** file with the `mxbai-embed-large` model served by your local **Ollama** instance, caches the vectors, and opens an interactive 2-D map showing clusters of similar documents.

### Key Features
- Embedding via the Ollama `/api/embed` endpoint handled with **Requests**
- Progress bars during long operations thanks to **tqdm**
- Robust PDF text extraction using **pypdf**
- Fast on-disk caching with **Joblib**
- Vector maths via **NumPy** and machine-learning utilities from **scikit-learn**
- Dimensionality reduction through **UMAP** (or t-SNE)
- Interactive Plotly scatter plot; hover reveals file names and cluster IDs
- Density-based clustering with **DBSCAN** (automatic number of clusters)

### Installation
1. Clone the repo or copy `vectormap.py`.
2. Ensure Python ≥ 3.9 is available.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start or confirm your Ollama server, e.g.:
   ```bash
   ollama run mxbai-embed-large
   ```

### Usage

#### Step 1 – Embed and Cache