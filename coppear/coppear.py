"""
Marker Co-appearance Analysis
==============================
Scores protein pairs using:
  1. Eigenvector centrality (from the co-occurrence adjacency matrix)
  2. Log Stability-Weighted Score (observed co-selection / expected by chance)

Input CSV: must contain columns 'runID' and 'Assay'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import combinations
from scipy.sparse.linalg import eigs
import argparse
import sys


# ── 1. Load & parse ──────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Use positional columns (col index 1 = runID, col index 2 = Assay)
    # but prefer named columns if they exist
    cols = df.columns.tolist()
    if "runID" not in cols or "Assay" not in cols:
        print(f"  Columns found: {cols}")
        print("  'runID' or 'Assay' not found by name — using column positions 1 and 2.")
        df = df.rename(columns={cols[1]: "runID", cols[2]: "Assay"})
    return df[["runID", "Assay"]]


# ── 2. Build co-occurrence matrix ────────────────────────────────────────────

def build_cooccurrence(df: pd.DataFrame):
    """
    Returns:
        markers  : sorted list of unique marker names
        C        : (n x n) symmetric co-occurrence count matrix  (np.ndarray)
        pi       : (n,) marginal selection probability per marker
        m        : total number of runs
    """
    df = df.dropna(subset=["Assay"])
    runs = df.groupby("runID")["Assay"].apply(set)
    m = len(runs)
    markers = sorted(df["Assay"].unique())
    idx = {mk: i for i, mk in enumerate(markers)}
    n = len(markers)

    C = np.zeros((n, n), dtype=float)
    pi = np.zeros(n, dtype=float)

    for run_markers in runs:
        for mk in run_markers:
            pi[idx[mk]] += 1
        for a, b in combinations(sorted(run_markers), 2):
            i, j = idx[a], idx[b]
            C[i, j] += 1
            C[j, i] += 1

    pi /= m  # convert to probability
    return markers, C, pi, m


# ── 3. Eigenvector centrality ────────────────────────────────────────────────

def eigenvector_centrality(C: np.ndarray) -> np.ndarray:
    """
    Leading eigenvector of C (the co-occurrence adjacency matrix).
    All entries are non-negative by Perron–Frobenius, so we take absolute value
    to be safe with numerical sign flips.
    """
    # Use sparse eigs for the single largest eigenvalue
    vals, vecs = eigs(C.astype(float), k=1, which="LM")
    ev = np.abs(vecs[:, 0].real)
    ev /= ev.max()          # normalise to [0, 1]
    return ev


# ── 4. Pairwise scores ────────────────────────────────────────────────────────

def _mi_term(p_joint, p_i, p_j):
    """p_joint * log2(p_joint / (p_i * p_j)), safely handling zeros."""
    if p_joint <= 0 or p_i <= 0 or p_j <= 0:
        return 0.0
    return p_joint * np.log2(p_joint / (p_i * p_j))


def pairwise_scores(markers, C, pi, ev, m):
    """
    For every marker pair (i < j) compute:
      - eig_pair  : geometric mean of individual eigenvector centralities
      - log_stab  : log( P(i,j)_obs / (pi[i] * pi[j]) )   [PMI, nats]
      - mutual_info : full binary mutual information in bits

    P(i,j)_obs = C[i,j] / m  (fraction of runs where both markers appear).
    PMI uses only the (1,1) cell; MI integrates all four cells of the
    2×2 co-occurrence table.
    """
    n = len(markers)
    records = []

    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] == 0:
                continue  # skip pairs that never co-appeared

            eig_pair = np.sqrt(ev[i] * ev[j])

            p11 = C[i, j] / m
            p10 = pi[i] - p11
            p01 = pi[j] - p11
            p00 = 1.0 - pi[i] - pi[j] + p11

            expected = pi[i] * pi[j]
            log_stab = np.log(p11 / expected) if expected > 0 else np.nan

            mi = (
                _mi_term(p11, pi[i], pi[j])
                + _mi_term(p10, pi[i], 1 - pi[j])
                + _mi_term(p01, 1 - pi[i], pi[j])
                + _mi_term(p00, 1 - pi[i], 1 - pi[j])
            )

            records.append({
                "marker_A": markers[i],
                "marker_B": markers[j],
                "cooccurrence_count": int(C[i, j]),
                "eig_centrality_pair": eig_pair,
                "log_stability_score": log_stab,
                "mutual_info_bits": mi,
            })

    return pd.DataFrame(records).dropna()


# ── 5. Scatter plot ───────────────────────────────────────────────────────────

def scatter_plot(df_pairs: pd.DataFrame, out_path: str = "marker_scores_scatter.png"):
    # ── colour by co-occurrence count ──
    counts = df_pairs["cooccurrence_count"].values
    norm_counts = (counts - counts.min()) / (counts.max() - counts.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sc = ax.scatter(
        df_pairs["log_stability_score"],
        df_pairs["eig_centrality_pair"],
        c=norm_counts,
        cmap="YlOrRd",
        alpha=0.75,
        s=18,
        linewidths=0.3,
        edgecolors="#aaaaaa",
        zorder=3,
    )

    # ── square-root Y scale: expands low centrality, compresses high ──
    ax.set_yscale("function", functions=(
        lambda x: np.sqrt(np.maximum(x, 0)),
        np.square,
    ))
    # Place ticks at interpretable centrality values
    yticks = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks])

    # ── reference lines ──
    ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=2)
    ax.axhline(df_pairs["eig_centrality_pair"].mean(),
               color="#aaaaaa", lw=0.8, ls="--", zorder=2)

    # ── colour bar ──
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Co-occurrence count (normalised)", color="#333333", fontsize=10)
    cbar.outline.set_edgecolor("#cccccc")

    # ── labels & aesthetics ──
    ax.set_xlabel("Log Stability-Weighted Score\n(log observed / expected)",
                  color="#333333", fontsize=12, labelpad=10)
    ax.set_ylabel("Eigenvector Centrality (pair, geometric mean) — √ scale",
                  color="#333333", fontsize=12, labelpad=10)
    ax.set_title("Protein Pair Co-appearance Scores",
                 color="#111111", fontsize=15, fontweight="bold", pad=16)

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(colors="#333333")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(color="#eeeeee", linewidth=0.5, zorder=1)

    # ── annotate top 5 pairs by combined rank ──
    df_pairs = df_pairs.copy()
    df_pairs["rank"] = (
        df_pairs["log_stability_score"].rank() +
        df_pairs["eig_centrality_pair"].rank()
    )
    top = df_pairs.nlargest(5, "rank")
    for _, row in top.iterrows():
        label = f"{row['marker_A']}–{row['marker_B']}"
        ax.annotate(
            label,
            (row["log_stability_score"], row["eig_centrality_pair"]),
            fontsize=7, color="#c0392b",
            xytext=(6, 4), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#c0392b88", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Scatter plot saved → {out_path}")
    plt.close()


# ── 6. Stability vs MI plot ──────────────────────────────────────────────────

def stability_mi_plot(df_pairs: pd.DataFrame, out_path: str = "marker_stability_mi.png"):
    centrality = df_pairs["eig_centrality_pair"].values

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sc = ax.scatter(
        df_pairs["log_stability_score"],
        df_pairs["mutual_info_bits"],
        c=centrality,
        cmap="Blues",
        alpha=0.75,
        s=18,
        linewidths=0.3,
        edgecolors="#aaaaaa",
        zorder=3,
    )

    ax.set_yscale("function", functions=(
        lambda x: np.sqrt(np.maximum(x, 0)),
        np.square,
    ))

    ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=2)
    ax.axhline(df_pairs["mutual_info_bits"].mean(),
               color="#aaaaaa", lw=0.8, ls="--", zorder=2)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Eigenvector Centrality (pair)", color="#333333", fontsize=10)
    cbar.outline.set_edgecolor("#cccccc")

    ax.set_xlabel("Log Stability-Weighted Score\n(log observed / expected)",
                  color="#333333", fontsize=12, labelpad=10)
    ax.set_ylabel("Mutual Information (bits) — √ scale",
                  color="#333333", fontsize=12, labelpad=10)
    ax.set_title("Protein Pair: Log Stability vs Mutual Information",
                 color="#111111", fontsize=15, fontweight="bold", pad=16)

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(colors="#333333")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(color="#eeeeee", linewidth=0.5, zorder=1)

    # annotate top 5 pairs by combined rank
    df_pairs = df_pairs.copy()
    df_pairs["rank"] = (
        df_pairs["log_stability_score"].rank() +
        df_pairs["mutual_info_bits"].rank()
    )
    top = df_pairs.nlargest(5, "rank")
    for _, row in top.iterrows():
        label = f"{row['marker_A']}–{row['marker_B']}"
        ax.annotate(
            label,
            (row["log_stability_score"], row["mutual_info_bits"]),
            fontsize=7, color="#c0392b",
            xytext=(6, 4), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#c0392b88", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Stability vs MI plot saved → {out_path}")
    plt.close()


# ── 7. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Marker co-appearance scoring")
    parser.add_argument("csv", help="Path to input CSV file")
    parser.add_argument("--out-csv",  default="marker_pair_scores.csv",
                        help="Output CSV with pair scores (default: marker_pair_scores.csv)")
    parser.add_argument("--out-plot", default="marker_scores_scatter.png",
                        help="Output scatter plot (default: marker_scores_scatter.png)")
    parser.add_argument("--out-plot-mi", default="marker_stability_mi.png",
                        help="Output stability vs MI plot (default: marker_stability_mi.png)")
    args = parser.parse_args()

    print(f"\n── Loading data from: {args.csv}")
    df = load_data(args.csv)
    print(f"   {len(df)} rows | {df['runID'].nunique()} runs | {df['Assay'].nunique()} unique markers")

    print("── Building co-occurrence matrix …")
    markers, C, pi, m = build_cooccurrence(df)
    print(f"   {len(markers)} markers | {m} runs")

    print("── Computing eigenvector centrality …")
    ev = eigenvector_centrality(C)

    print("── Computing pairwise scores …")
    df_pairs = pairwise_scores(markers, C, pi, ev, m)
    print(f"   {len(df_pairs)} co-appearing pairs")

    df_pairs.to_csv(args.out_csv, index=False)
    print(f"   Pair scores saved → {args.out_csv}")

    print("── Generating scatter plot …")
    scatter_plot(df_pairs, out_path=args.out_plot)

    print("── Generating stability vs MI plot …")
    stability_mi_plot(df_pairs, out_path=args.out_plot_mi)

    print("\n── Top 10 pairs by log stability score:")
    print(df_pairs.nlargest(10, "log_stability_score")
                  .to_string(index=False))

    print("\n── Top 10 pairs by eigenvector centrality:")
    print(df_pairs.nlargest(10, "eig_centrality_pair")
                  .to_string(index=False))

    print("\nDone.\n")


if __name__ == "__main__":
    main()
