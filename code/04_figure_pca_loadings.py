"""Render the PCA factor loading matrix heatmap (Appendix Figure D2).

Purpose:
    Visualizes absolute factor loadings across the 16 varimax-rotated principal factors:
    1. Loads the fitted 50 word clusters x 16 factor loadings matrix.
    2. Sorts word cluster rows by their primary assigned factor (factor blocks).
    3. Renders a publication-quality portrait heatmap (50 word clusters as rows,
       16 factors as columns, 14x16 inches, 600 dpi) with annotated loadings.
    4. Inserts horizontal demarcation lines between distinct factor blocks.

Input Files:
    - output/intermediate/tag_factor_analysis/pca_loadings_fitted.csv: 50x16 factor loadings matrix with assigned factor column.

Output Files:
    - output/figures/figure_pca_factor_loadings.png: Appendix Figure D2 heatmap visualization (600 dpi).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import DATA, OUT, out_path, require, save




def main():
    fitted = pd.read_csv(out_path('pca_loadings_fitted.csv'))
    loadings = fitted.set_index("group").drop(columns="assigned_factor")
    assignment = fitted[["group", "assigned_factor"]].rename(
        columns={"group": "groupid", "assigned_factor": "factor"})
    require(loadings.shape == (50, 16), "Expected 50 word clusters by 16 factors")
    require(set(assignment.groupid) == set(loadings.index),
            "Factor assignment covers different word clusters than the loadings")
    factor_of = dict(zip(assignment.groupid, assignment.factor.astype(int)))
    require(sorted(set(factor_of.values())) == list(range(16)),
            "Factor assignment must cover factors 0-15")
    # Use the original 16-factor interpretations to name the data-driven factors.
    names = pd.read_csv(DATA / "factor_scene_names.csv").sort_values("factor_id")
    require(names.factor_id.tolist() == list(range(16)), "Expected original factors 0-15")
    labels = names.factor_name.tolist()
    absolute = loadings.abs()

    order = sorted(loadings.index, key=lambda group: factor_of[group])
    shown = absolute.loc[order]

    fig, ax = plt.subplots(figsize=(14, 16))
    sns.heatmap(shown, annot=True, fmt=".2f", cmap="BuPu", ax=ax,
                vmin=0, vmax=.85, annot_kws={"size": 5},
                cbar_kws={"label": "Absolute factor loading"})
    ax.set_ylabel("Word cluster (group, sorted by assigned factor)")
    ax.set_xlabel("")
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(shown.index, fontsize=7)
    boundaries = [index for index in range(1, 50)
                  if factor_of[order[index]] != factor_of[order[index - 1]]]
    for boundary in boundaries:
        ax.axhline(boundary, color="0.25", linewidth=.55)
    fig.tight_layout()
    fig.savefig(out_path('figure_pca_factor_loadings.png'), dpi=600)
    plt.close(fig)

    print("Wrote figure_pca_factor_loadings.png (16 factors x 50 clusters).",
          flush=True)


if __name__ == "__main__":
    main()
