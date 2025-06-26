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
Run the `embed` command to process a folder of Markdown and PDF files, generate embeddings, and save them to a cache file:
```markdown
vectormap embed /path/to/folder
```

#### Step 2 – Visualise
Run the `visualise` command to load the cached embeddings and open the interactive map:
```markdown
vectormap visualise /path/to/folder
```

#### Step 3 – Cluster
Run the `cluster` command to apply DBSCAN clustering to the cached embeddings:
```markdown
vectormap cluster /path/to/folder
```

#### Additional Details
- The `embed` command processes all Markdown and PDF files in the specified folder and its subfolders.
- The `visualise` command uses dimensionality reduction (UMAP or t-SNE) to create a 2-D map and opens it in a browser.
- The `cluster` command applies DBSCAN clustering to identify document groups based on similarity.
- All commands rely on the cache file created during the embedding step.