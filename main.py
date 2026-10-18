"""
Main Execution Pipeline.

Purpose:
    Orchestrates the empirical analysis and visualization pipeline in sequential
    order, executing 13 standalone Python scripts that generate the manuscript
    tables, the CNN validation metrics, and the manuscript figures.

    The default run continues from the retained upstream artifacts: the 719-tag
    WMD distance matrix, the retained nine-topic LDA document assignments and
    model visualization, and the fitted ResNet18 checkpoint. The LDA model is
    not re-estimated, the CNN is used only to score the fixed 431-image
    validation split, and the 16-factor principal component analysis (PCA) is
    re-estimated in this run.

Workflow Steps:
    0. code/00_filter_reviews.py: Select If pic=Y source review IDs and
       require an exact match to the supplied 3,068 manual-coding records.
    1. code/01_analysis.py: Tag frequency filtering, WMD-profile clustering into
       50 word clusters, re-estimation of the 16-factor varimax-rotated PCA,
       data-driven scene membership, and scene-topic integration over the
       retained LDA assignments.
    2. code/02_scene_topic_tables.py: Appendix Tables 1-4 (scene frequencies,
       topic symbolism, scene-topic co-occurrence, and contribution matrices)
       plus the within-scene contribution heatmap.
    3. code/03_figure_intertopic_map.py: Appendix Figure D1 (intertopic distance
       map of the retained 9-topic LDA model).
    4. code/04_figure_pca_loadings.py: Appendix Figure D2 (PCA 16-factor loading
       heatmap of the loadings fitted in step 1).
    5. code/05_campsite_distribution.py: Table 5 (Appendix E; campsite-level
       distribution of the confirmed reviews and campsite-name group analysis).
    6. code/06_score_cnn_validation.py: CPU scoring of the fixed 431-image
       validation split with the retained ResNet18 checkpoint.
    7. code/07_cnn_validation_metrics.py: CNN validation metrics and confusion
       matrix (Appendix Figure F1), with the historical five-fold summary.
    8. code/08_gradcam.py: Grad-CAM visual explainability heatmaps
       (Appendix Figure F2).
    9. code/09_image_grid.py: 12 representative validation photographs with
       predicted labels and confidences (Appendix Figure F3).
   10. code/10_scene_figures.py: Figures 4, 6, and 7 (scene frequencies,
       scene-topic associations, and the scene-symbolism integration diagram).
   11. code/11_tag_group_mapping.py: Group-ID lookup tables linking the tag
       vocabulary, the 50 word clusters (group0..group49), the 16 PCA factors
       with names, the 15 ritual scenes, and the taxonomy subclasses.
   12. code/12_scree_parallel_analysis.py: Scree plot and parallel analysis
       comparing the 16 retained factors with the Kaiser and parallel-analysis
       benchmarks.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENV = os.environ.copy()
ENV['PYTHONDONTWRITEBYTECODE'] = '1'
ENV['OPENBLAS_NUM_THREADS'] = '1'
ENV['OMP_NUM_THREADS'] = '1'

def main():
    steps = [
        '00_filter_reviews.py',
        '01_analysis.py',
        '02_scene_topic_tables.py',
        '03_figure_intertopic_map.py',
        '04_figure_pca_loadings.py',
        '05_campsite_distribution.py',
        '06_score_cnn_validation.py',
        '07_cnn_validation_metrics.py',
        '08_gradcam.py',
        '09_image_grid.py',
        '10_scene_figures.py',
        '11_tag_group_mapping.py',
        '12_scree_parallel_analysis.py'
    ]
    for name in steps:
        print(f'Running {name}',flush=True)
        subprocess.run([sys.executable,str(ROOT/'code'/name)],cwd=ROOT,
                       env=ENV,check=True)
    print('Analysis completed. Tables and figures are in output/.',flush=True)

if __name__ == '__main__':
    main()
