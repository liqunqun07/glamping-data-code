# Glamping data and code — manuscript submission materials

> **Restricted use — contact the author before reuse.**
> These data and code accompany the manuscript **“Glamping as a Ritual Experience: Exploring Emotional Value in Outdoor Recreation through Visual and Textual Analysis”** and are provided for its submission and peer review. Public access or downloading does not grant permission for other uses. Any use beyond this manuscript's submission and peer review—including other research, model training, commercial use, redistribution, or repackaging—requires prior contact with **Qin Li at [kikili@zju.edu.cn](mailto:kikili@zju.edu.cn)** and explicit permission. Do not misuse these materials. Any third-party rights remain with their respective holders.

The original package directory structure is available directly in this repository. The platform username column has been removed from `input/raw_reviews.csv`, consistent with the existing release package. Repository presentation updated by Codex.

---

# Glamping scene and topic analysis

Use Python 3.10 or 3.11 on a CPU. The package versions pinned in `requirements.txt` are the ones used to generate the files in `output/` (Python 3.10.0rc1 on macOS, CPU only); a clean run with these versions regenerated every shipped output file byte for byte. From this directory, run:

```bash
python -m pip install -r requirements.txt
python main.py
```

Generated files are organized under `output/mappings/`, `output/tables/`, `output/figures/`, `output/cnn_validation/`, and `output/intermediate/`. These folders are created automatically.
The package uses relative paths and does not download models during this run.

## Contents

- `input/`: the 3,176 source reviews with image-eligibility flags, source image detections, the 3,068 manually verified reviews, the 1,559 machine-selected review texts, the 1,436 CNN training and validation images, campsite names, and the scene and factor taxonomies.
- `retained/`: the 719-tag WMD matrix, nine-topic document assignments and model visualization, the fitted ResNet18 checkpoint, the five-fold cross-validation accuracy summary.
- `code/`: the feature processing, clustering, factor extraction, aggregation, campsite analysis, CNN validation scoring, figure generation, and group-ID mapping routines (image-eligibility check 00 followed by analysis steps 01 to 12).
- `code_original/`: selected historical WMD and LDA scripts and text stopwords. These files are not called by `main.py`; see the folder README for the missing upstream resources.

The two main mapping files are the shortest route for inspecting the full chain: `tag_group_mapping.csv` has one row per keyword; `cluster_group_mapping.csv` has one row per group and both corpus counts. `tables/` contains the manuscript tables, `figures/` contains exportable images, `cnn_validation/` contains image-classifier validation details, and `intermediate/tag_factor_analysis/` contains tag and factor data, `intermediate/scene_topic_analysis/` contains scene and topic data, and `intermediate/campsite_analysis/` contains campsite analysis data. The diagnostic row-order CSV is omitted because the same order is encoded in the PCA heatmap and factor mapping. The stand-alone word-cluster table, duplicated scene taxonomy, and scene-topic summary are also omitted; their information is available from the mapping tables, `input/scene_taxonomy_v3.csv`, and the generated tables.

The default run executes all downstream statistical modeling, campsite comparisons, CNN validation scoring, and table/figure generation from the supplied detections and model artifacts.

## Data inventory

| File or directory | Records or coverage | Role |
| --- | --- | --- |
| `input/raw_reviews.csv` | 3,176 reviews | Source review fields plus `record_id`, `If pic` (Y/N), and `exclusion_reason`. |
| `input/human_coding.csv` | 3,068 reviews | Manual verification scores across five dimensions (0-3 scale) and dual-gate classification outcomes. |
| `input/candidate_posts.csv` | 1,559 reviews | Machine-selected review IDs, translated texts, and classification labels. |
| `input/raw_vision_tags.csv` | 9,572 image records; 1,476 binary tag columns | Google Cloud Vision entity detections across photographs. |
| `input/cnn_images/` | 1,005 training images and 431 validation images | Image dataset for CNN model training and validation evaluation. |
| `input/campsite_display_order.csv` and `input/campsite_names.json` | 190 campsites | Site identity, display order, and explicit name-group categorizations. |
| `input/scene_taxonomy_v3.csv` and `input/factor_scene_names.csv` | 15 scenes and 16 factor names | Original scene taxonomy and Appendix factor table that name the data-driven membership. All source image records are used. |
| `code_original/lda_custom_stopwords.csv` | 58 stopwords | Historical text preprocessing stopword list. |
| `retained/tag_wmd_distance.csv` | 719 by 719 | Word Mover's Distance matrix calculated from Google News word embeddings. |
| `retained/lda_topic_assignments.csv` and `retained/lda_model_visualization.html` | Nine topics over 1,559 reviews | The CSV contains only `record_id` and `topic_id` (one dominant topic per review); no posterior probability vectors are stored. Keywords and topic-map coordinates are in the HTML. |
| `retained/best_glamping.pt` and `retained/cnn_five_fold_accuracy.csv` | Fitted checkpoint and five-fold accuracies | Saved ResNet18 model checkpoint and five-fold cross-validation performance metrics. |
| `workflow_documentation.json` | Current workflow | English description of the package entry point, script order, inputs, and artifact limitations. |

The source files remain unchanged when running the package. The final 1,409-review dataset, fitted loadings, scene tables, co-occurrence matrices, manuscript figures, and group-ID mapping tables are generated from these inputs in `output/`.

### Identifier mapping chain

`output/mappings/tag_group_mapping.csv` and `output/mappings/cluster_group_mapping.csv` resolve every identifier used across the analysis in one code-generated lookup. The chain has three levels:

1. **keyword -> word cluster (group)**: each of the 719 vector-covered keywords belongs to exactly one Ward word cluster (`cluster_id`, written `group0`..`group49`; consistent with `tag_factor_membership.csv` and the heatmap rows of `figure_pca_factor_loadings.png`).
2. **cluster -> factor -> scene**: each cluster inherits the PCA factor it loads most strongly on (`factor_id`, `factor_name` from `factor_scene_names.csv`) and that factor's ritual scene.
3. **cluster -> Appendix subclass**: each cluster maps to exactly one taxonomy subclass name, drawn verbatim from `scene_taxonomy_v3.csv` so it matches Appendix Table 1. The keyword-level table carries one row per keyword (no subclass expansion, so no keyword repeats), and the cluster-level table carries one row per cluster. Per-factor subclass occurrence counts sum to the factor total.

The cluster -> subclass binding is semantic rather than estimated: the clusters and factors are fitted, but no fitted quantity assigns a cluster to a specific subclass. It is supplied as an explicit codebook, `input/group_subclass_assignment_v1.csv` (50 rows), which assigns each of the 50 clusters to one of the 50 taxonomy subclass names based on keyword semantics. `code/11_tag_group_mapping.py` joins this codebook to the fitted membership and requires every assigned subclass to be a valid subclass of its scene; no subclass name is invented.

### Basis: 1,409 vs 1,559 records

The candidate-count columns in the mapping tables, tag counts, and PCA loadings are computed on the **full candidate corpus** (1,559 machine-positive reviews; 8,965 photographs; 87,972 tag occurrences). The reported scene frequencies in the manuscript (`output/tables/table1_scene_frequencies.csv`) use the **final 1,409-review glamping-confirmed subset** (8,330 image records; 81,764 occurrences). `cluster_group_mapping.csv` also includes `final_1409_occurrences`, computed on the final 8,330 photographs. Those 50 group counts sum to 81,764, and their 15 scene sums equal Table 1 exactly. `candidate_1559_occurrences` sums to 87,972. The keyword-to-group-to-factor identifiers are the same on both bases; only the counts differ.

### Declared uncovered vocabulary

Of the 732 high-frequency tags (occurrence strictly above the median, 4.0), 719 have semantic vector coverage in the WMD matrix and enter the analysis. The remaining **13 tags are excluded** because they lack vector coverage and cannot be placed in a cluster: Arecales, Arrosticini, Canidae, Cervelat, Close-up, Diot, Felidae, Grey, Jewellery, Longaniza, Subshrub, T-shirt, Windsports. These tags do not appear in any mapping table or figure.

## Methodology and analytical pipeline

### Initial image-eligibility filter

The source collection contains 3,176 reviews. In `input/raw_reviews.csv`, `If pic` indicates whether photographs meet the study inclusion criteria: Y for 3,068 eligible reviews and N for 108 reviews excluded for unclear photographs. Exclusion reasons follow the authors' supplied screening description. `record_id` combines `spot_id` and `num` with an underscore. Step `code/00_filter_reviews.py` selects the Y IDs and requires them to match `input/human_coding.csv` exactly before the existing analysis runs. The supplied manual scores are retained unchanged; this filter does not generate scores. In the campsite distribution table, `eligible_reviews` counts these eligible reviews per campsite.

The analysis evaluates 3,068 manually verified social media reviews across five experiential dimensions on a 0-3 scale. Reviews meeting the core accommodation/hospitality thresholds and experiential criteria are confirmed as glamping. The final analytical corpus comprises the 1,409 reviews confirmed as glamping by both preliminary machine screening and the manual verification protocol.

Photographs associated with the 1,559 machine-positive reviews enter tag frequency filtering and factor modeling. Tags with occurrences above the median (4.0) are retained (732 tags, of which 719 have semantic vector coverage in the WMD matrix). Ward hierarchical clustering on WMD profiles yields 50 word clusters. Standardized photo-level cluster counts enter a 16-factor principal factor extraction with varimax rotation, accounting for 54.05% of total standardized-variable variance. Scene membership is derived from the fitted model itself: every tag is assigned to the factor holding its largest absolute loading, and each stable factor id takes the scene name of the original Appendix factor table in `input/factor_scene_names.csv`, where factors 1 and 15 both merge into Food and cooking. Fitted PCA loadings are reported in `output/intermediate/tag_factor_analysis/pca_loadings_fitted.csv`. Sixteen factors are retained as a fixed specification, not by the Kaiser rule. The fitted correlation matrix has 19 eigenvalues above one and a parallel analysis (1,000 simulated data sets, 95th percentile) supports 17 components, but the eigenvalues beyond the sixteenth are barely above one (1.00 to 1.05) and differ little from one another; to avoid an unnecessarily large number of factors, 16 are retained. `code/12_scree_parallel_analysis.py` writes the eigenvalues with the parallel-analysis benchmark to `output/intermediate/tag_factor_analysis/scree_parallel_analysis.csv` and the scree plot to `output/figures/figure_pca_scree_plot.png`. The first 16 unrotated components explain 54.22% of the variance; the fitted rotated 16-factor solution explains 54.05%.

The nine text topics use the established document assignments. Each review has one assigned topic; scene-tag counts from all of its photographs are summed into that topic column. Matrix cells count tag occurrences, not distinct reviews, photographs, or same-image scene pairs. Reviews with more photographed tags can therefore contribute more weight. Topic keywords and map coordinates are extracted from the retained LDA model visualization. The scene-by-topic contribution table follows standard two-stage normalization: cell counts divided by each topic column total, followed by within-scene row percentage normalization (RP).

For campsite names, the premium group matches `luxury|boutique|glamping` in English names. The plain landscape group matches U+5C71 or the U+666F/U+533A sequence in the original names and excludes the premium group. Inclusive landscape matching and English-only `mountain|scenic` calculations are also generated to distinguish these operational definitions. Group membership and summary statistics are output to `output/tables/table5_campsite_distribution.csv`.

The full 719 by 719 retained WMD matrix enters clustering. All 9,572 source image rows are used. The 8,965 photographs from the 1,559 candidate reviews enter PCA before manual eligibility filtering. The final 1,409 reviews contain 8,330 image records and 81,764 scene-tag occurrences. The 16-factor extraction uses the same candidate corpus regardless of the final review filter.

## Retained artifacts and upstream procedures

The historical WMD and LDA code is included in `code_original/` to document the earlier analysis procedures. It is not executable as supplied because the original upstream resources and environment are incomplete in this package. The saved outputs in `retained/` provide the inputs for the current downstream analysis; inability to rerun an upstream procedure from this package does not imply that it was omitted from the original study.

The WMD matrix is retained because generating it anew requires the external Google News word-vector resource and the original preprocessing environment. The selected historical routine is in `code_original/04rerundistance.py`; it is provided as a methodological trace and is excluded from `main.py`. The 719 by 719 retained matrix is the fixed input used by the current Ward clustering stage.

The nine-topic LDA document assignments and visualization are retained as upstream inputs. The selected historical topic-modeling routine is in `code_original/05_lda_topic_model.py`; it records preprocessing and model selection steps but is excluded from `main.py`. Rebuilding the exact early topic fit requires the original translated review preparation and model environment, and the nine-topic, alpha 0.91 selection included interpretation. Neither historical script is represented as an exact from-scratch rebuild of the retained outputs.

The CNN checkpoint is used to score all 431 validation images on CPU, generating validation accuracy (80.74%), weighted F1 (80.5%), and the confusion matrix figure (`output/figures/figure_cnn_confusion_matrix.png`). Grad-CAM visualizations (`output/figures/gradcam_examples.png`) are generated from the model weights and validation photographs, and representative prediction grids (`output/figures/plot2_v2_representative_grid.png`) from the saved validation predictions and photographs. Reference training, cross-validation, and scoring procedures are retained with the local source materials outside this submission package; they are excluded from the default run. Optional review-photo scoring requires `input/02_review_photo_manifest.csv` and `input/review_images/`, which are not supplied; validation scoring uses the supplied images. The original model-selection note is retained with the local source materials outside this submission package.

## Scope of retained evidence

The manual-coding file contains 3,068 reviews, including the 1,559 CNN-positive candidates. The source review file now identifies all 108 image-eligibility exclusions through `If pic=N` and `exclusion_reason`. The review-level rule for aggregating CNN photo predictions is not supplied and cannot be reconstructed from these files. The single score set per review does not establish the number of coders or whether their assessments were independent.

The historical custom stopword list and preprocessing routine do not establish that every travel-related or place-related term was removed from the retained LDA fit: its saved topic keywords still include such terms. Retained topic assignments and map coordinates remain fixed; the current pipeline does not refit LDA.

Scene names are interpretations of data-driven clusters and factors. Heterogeneous subclasses remain in their fitted factors, including geographical features in Furnishings and plant/family terms in Music performance. No semantic reassignment or alternative-factor sensitivity analysis is performed by the default workflow.

