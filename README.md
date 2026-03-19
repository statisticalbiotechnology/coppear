# coppear

**Co-appearance scoring for protein/marker panels across experimental runs.**

`coppear` reads a CSV of assay selections grouped by run ID, builds a co-occurrence matrix, and scores every marker pair using two complementary metrics:

1. **Eigenvector centrality** — how central a pair is within the overall co-selection network (Perron–Frobenius leading eigenvector).
2. **Log Stability-Weighted Score** — log-ratio of observed co-selection frequency to the frequency expected by chance (analogous to pointwise mutual information).

Pairs that score high on both metrics tend to be consistently co-selected across runs and structurally central in the selection network — good candidates for stable, informative marker combinations.

---

## Input format

A CSV file with at least two columns:

| Column  | Description                                      |
|---------|--------------------------------------------------|
| `runID` | Identifier for each experimental run or sample   |
| `Assay` | Name of the marker/protein selected in that run  |

Each row is one marker selected in one run. Multiple rows with the same `runID` form a panel.

If the columns are not named `runID` and `Assay`, the script falls back to positional columns (index 1 and 2).

---

## Installation

```bash
pip install numpy pandas matplotlib scipy
```

---

## Usage

```bash
python coppear/coppear.py <input.csv> [--out-csv results.csv] [--out-plot scatter.png]
```

| Argument      | Default                    | Description                        |
|---------------|----------------------------|------------------------------------|
| `csv`         | *(required)*               | Path to the input CSV file         |
| `--out-csv`   | `marker_pair_scores.csv`   | Output CSV with all pair scores    |
| `--out-plot`  | `marker_scores_scatter.png`| Output scatter plot (PNG)          |

### Example

```bash
python coppear/coppear.py data/selected_features_cancer_classification.csv \
    --out-csv /tmp/results.csv \
    --out-plot ~/Downloads/scatter.png
```

---

## Output

### CSV (`--out-csv`)

One row per co-appearing marker pair:

| Column                  | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `marker_A`              | First marker in the pair                                     |
| `marker_B`              | Second marker in the pair                                    |
| `cooccurrence_count`    | Number of runs in which both markers were selected together  |
| `eig_centrality_pair`   | Geometric mean of both markers' eigenvector centralities     |
| `log_stability_score`   | Log(observed co-selection / expected by chance)              |

Pairs that never co-appeared, or with undefined expected probability, are excluded.

### Scatter plot (`--out-plot`)

A scatter of all co-appearing pairs with:

- **X-axis**: Log Stability-Weighted Score (positive = more co-selected than chance)
- **Y-axis**: Eigenvector Centrality (pair geometric mean)
- **Colour**: Normalised co-occurrence count (plasma colormap)
- **Annotations**: Top 5 pairs by combined rank labelled directly on the plot

---

## Methods

### Co-occurrence matrix

For each run, all pairs of markers in the same panel contribute +1 to the symmetric co-occurrence matrix **C**. Marginal selection probabilities **π** are estimated as the fraction of runs each marker appears in.

### Eigenvector centrality

The leading eigenvector of **C** is computed via sparse power iteration (`scipy.sparse.linalg.eigs`). Entries are normalised to [0, 1]. The pair score is the geometric mean of the two marker centralities.

### Log Stability-Weighted Score

```text
log_stab(i, j) = log( P(i,j)_observed / (π_i × π_j) )
```

where `P(i,j)_observed = C[i,j] / sum(C)`. Positive values mean the pair co-appears more often than independence would predict.
