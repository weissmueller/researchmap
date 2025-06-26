# vectormap_table.py

**vectormap_table.py** is a tool for clustering and visualizing research papers based on their metadata (titles, abstracts, keywords) from a CSV or XLSX file. It uses modern embedding models and advanced clustering algorithms to help you explore the thematic structure of your document collection.

---

## Features

- **Flexible Input:** Works with CSV or XLSX files containing columns like "Paper Title", "Abstract", and "Keywords".
- **Custom Embedding:** Choose to embed using the full text, only keywords, or other fields.
- **State-of-the-Art Clustering:** Supports community detection (Leiden/Louvain) on k-NN graphs for robust, modern clustering.
- **Automatic Cluster Naming:** Uses an LLM to generate short, descriptive names for each cluster.
- **Interactive Visualization:** Explore your clusters in an interactive 2D map.
- **Export Results:** Optionally export an XLSX file with cluster assignments for each paper (works for both HDBSCAN and community workflows).

---

## Typical Workflow

### 1. **Embed your data**

Extracts embeddings from your table, using only the "Keywords" column (falls back to title if keywords are missing):

```bash
python vectormap_table.py embed '/path/to/yourfile.xlsx' --keywords-only
```

- This creates a `.joblib` file with the same base name as your input file.

---

### 2. **Cluster using community detection**

Runs Leiden community detection on a k-nearest neighbor graph (with k=5):

```bash
python vectormap_table.py community '/path/to/yourfile.joblib' --algorithm leiden --k 5
```

- You can also use `--algorithm louvain` for Louvain clustering.
- The script will automatically name clusters using the LLM.
- **To export cluster assignments to XLSX:**

```bash
python vectormap_table.py community '/path/to/yourfile.joblib' --algorithm leiden --k 5 --export-xlsx
```

- This will create a file like `yourfile_clusters.xlsx` with all original metadata and a new column for the cluster name.

---

### 3. **(Optional) Cluster using HDBSCAN and export results**

You can also use the `label` command to cluster with HDBSCAN and export the results to an XLSX file:

```bash
python vectormap_table.py label '/path/to/yourfile.joblib' --llm-host http://localhost:11434 --export-xlsx
```

- This will create a file like `yourfile_clusters.xlsx` with all original metadata and a new column for the cluster name.

---

### 4. **Visualize the clusters**

Opens an interactive map in your browser:

```bash
python vectormap_table.py map '/path/to/yourfile.joblib'
```

- Hover over points to see the paper title and cluster name.

---

## Command Reference

### Embedding

- `--keywords-only`  
  Use only the "Keywords" column for embedding (falls back to title if empty).

### Community Detection

- `--algorithm leiden|louvain`  
  Choose the community detection algorithm (default: leiden).
- `--k N`  
  Number of neighbors for the k-NN graph (default: 15).
- `--titles-only`  
  Use only paper titles (not abstracts) when sending cluster digests to the LLM for naming.
- `--export-xlsx`  
  Export an XLSX file with the original paper metadata and the cluster name for each paper (community workflow).

### HDBSCAN Labeling

- `--export-xlsx`  
  Export an XLSX file with the original paper metadata and the cluster name for each paper (HDBSCAN workflow).

### Visualization

- No special arguments needed; just provide the `.joblib` file.

---

## Example

```bash
python vectormap_table.py embed '/path/to/yourfile.xlsx' --keywords-only
python vectormap_table.py community '/path/to/yourfile.joblib' --algorithm leiden --k 5 --export-xlsx
python vectormap_table.py label '/path/to/yourfile.joblib' --llm-host http://localhost:11434 --export-xlsx
python vectormap_table.py map '/path/to/yourfile.joblib'
```

---

## Requirements

- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## Notes

- Your input file must have at least the columns "Paper Title" and "Abstract". For keyword-based embedding, a "Keywords" column is recommended.
- The script will automatically skip header rows if needed.
- The LLM for cluster naming is expected to be available at `http://localhost:11434` (Ollama or compatible).
- The `--export-xlsx` option is available for both the `label` (HDBSCAN) and `community` workflows.

--- 