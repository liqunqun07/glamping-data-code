"""Scree plot and parallel analysis for the 16-factor PCA decision.

Purpose:
    Documents how the number of retained factors relates to common retention
    rules, using the eigenvalues fitted in step 1:
    1. Reads the eigenvalues of the standardized 50 word-cluster correlation matrix.
    2. Simulates parallel analysis: 1,000 random normal data sets with the same
       size as the fitting data (one row per fitting photograph, 50 columns);
       the 95th percentile of each simulated eigenvalue is the benchmark.
    3. Counts the components above the Kaiser threshold (eigenvalue > 1) and
       above the parallel-analysis benchmark, and compares both with the 16
       retained factors.
    4. Draws the scree plot with both benchmarks and the cumulative variance
       explained by the leading unrotated components.

Input Files:
    - output/intermediate/tag_factor_analysis/pca_eigenvalues.csv: Eigenvalues from step 1.
    - output/intermediate/tag_factor_analysis/photo_cluster_counts.csv: Fitting photographs by word cluster (sets the simulated data size).

Output Files:
    - output/intermediate/tag_factor_analysis/scree_parallel_analysis.csv: Eigenvalue, parallel-analysis benchmark, and cumulative variance by component.
    - output/figures/figure_pca_scree_plot.png: Scree plot and cumulative variance (300 dpi).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import out_path, require, save

RETAINED_FACTORS = 16
SIMULATIONS = 1000
SEED = 0


def main():
    eigen = pd.read_csv(out_path('pca_eigenvalues.csv'))
    photos = pd.read_csv(out_path('photo_cluster_counts.csv'))
    n_rows, n_cols = len(photos), photos.shape[1] - 1
    require(len(eigen) == n_cols == 50, 'Expected 50 components for 50 word clusters')

    rng = np.random.default_rng(SEED)
    simulated = np.empty((SIMULATIONS, n_cols))
    for i in range(SIMULATIONS):
        data = rng.standard_normal((n_rows, n_cols))
        simulated[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(data, rowvar=False)))[::-1]
    benchmark = np.percentile(simulated, 95, axis=0)

    table = pd.DataFrame({'component': eigen.component,
                          'eigenvalue': eigen.eigenvalue,
                          'pa_95th': benchmark,
                          'cum_var_pct': eigen.cumulative_percent})
    save(table, 'scree_parallel_analysis.csv')
    kaiser = int((table.eigenvalue > 1).sum())
    # Parallel analysis retains components until the first one that falls below the benchmark.
    below = np.flatnonzero(table.eigenvalue.to_numpy() <= benchmark)
    parallel = int(below[0]) if len(below) else n_cols
    print(f'Eigenvalues above 1: {kaiser}; parallel analysis supports {parallel}; '
          f'retained factors: {RETAINED_FACTORS}.', flush=True)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.plot(table.component, table.eigenvalue, 'o-', color='#336699', ms=3.5, lw=1.6,
              label='Observed eigenvalue')
    left.plot(table.component, table.pa_95th, 's--', color='#999999', ms=3.5, lw=1.6,
              label='Parallel analysis (95th pct)')
    left.axhline(1, color='red', lw=0.9, ls=':', label='Kaiser (λ = 1)')
    left.axvline(RETAINED_FACTORS, color='black', lw=0.9, ls='-.',
                 label=f'{RETAINED_FACTORS} factors retained')
    left.set_title('(a)')
    left.set_xlabel('Component')
    left.set_ylabel('Eigenvalue')
    left.legend(fontsize=8, loc='upper right')

    right.plot(table.component, table.cum_var_pct, 'o-', color='#336699', ms=3.5, lw=1.6)
    right.axvline(RETAINED_FACTORS, color='black', lw=0.9, ls='-.')
    cum16 = float(table.cum_var_pct.iloc[RETAINED_FACTORS - 1])
    right.annotate(f'{RETAINED_FACTORS}: {cum16:.2f}%', xy=(RETAINED_FACTORS, cum16),
                   xytext=(22, 40), arrowprops={'arrowstyle': '->', 'color': 'black'})
    right.set_title('(b)')
    right.set_xlabel('Number of components')
    right.set_ylabel('Cumulative variance (%, unrotated)')
    fig.tight_layout()
    fig.savefig(out_path('figure_pca_scree_plot.png'), dpi=300)
    plt.close(fig)
    print('Wrote figure_pca_scree_plot.png.', flush=True)


if __name__ == '__main__':
    main()
