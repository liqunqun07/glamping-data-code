"""Extract image tag features, fit 16 principal factors, and aggregate 15 ritual scenes.

Purpose:
    Performs visual feature processing on photographs attached to the 1,559
    machine-positive candidate reviews:
    1. Validates the 3,068 human-coded reviews and checks the dual-gate eligibility rule.
    2. Filters high-frequency visual tags (> median) and identifies semantic vector coverage.
    3. Performs Ward hierarchical clustering on WMD profiles to obtain 50 word clusters.
    4. Fits a 16-factor principal factor analysis (PCA) with varimax rotation.
    5. Aggregates tag occurrences into the 15 ritual scenes through the data-driven
       membership derived from the fitted argmax factor assignments.
    6. Identifies the final 1,409 confirmed glamping reviews and merges scene counts with LDA topics.

Input Files:
    - input/human_coding.csv: Five-dimension manual verification scores (0-3 scale) for 3,068 reviews.
    - input/raw_vision_tags.csv: Google Cloud Vision API entity detections for 9,572 photographs.
    - input/candidate_posts.csv: 1,559 machine-positive (CNN-selected) review records.
    - retained/tag_wmd_distance.csv: 719x719 Word Mover's Distance semantic distance matrix.
    - input/scene_taxonomy_v3.csv: Taxonomy definitions for the 15 ritual scenes.
    - input/factor_scene_names.csv: Original names of the 16 factors and their
    merged 15-scene identities.
    - retained/lda_topic_assignments.csv: Retained 9-topic assignments for 1,559 reviews.

Output Files:
    - output/intermediate/tag_factor_analysis/tag_counts.csv: Occurrence counts for all 1,476 detected tags across fitting photos.
    - output/intermediate/tag_factor_analysis/tag_vocabulary.csv: 732 tags with occurrence count strictly above the median (4.0).
    - output/intermediate/tag_factor_analysis/photo_cluster_counts.csv: Photograph-level counts across the 50 word clusters.
    - output/intermediate/tag_factor_analysis/pca_loadings_fitted.csv: Fitted factor loading matrix (50 clusters x 16 factors).
    - output/intermediate/tag_factor_analysis/pca_eigenvalues.csv: Eigenvalues and variance explained for all 50 principal components.
    - output/mappings/tag_factor_membership.csv: Keyword mapping to clusters, assigned factors, and occurrences.
    - output/intermediate/tag_factor_analysis/photo_factor_counts.csv: Photograph-level counts across the 16 rotated factors.
    - output/intermediate/scene_topic_analysis/photo_scenes.csv: Photograph-level occurrence counts for the 15 ritual scenes.
    - output/intermediate/scene_topic_analysis/review_scenes.csv: Review-level aggregated scene counts for 1,559 candidate reviews.
    - output/intermediate/scene_topic_analysis/review_scene_topics.csv: Merged scene counts and LDA topics for 1,409 confirmed reviews.
    - output/intermediate/tag_factor_analysis/analysis_summary.json: Summary statistics of dataset, vocabulary, factors, and occurrences.
"""
import json
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from common import DATA, RETAINED, OUT, out_path, require, save

def main():
    coding = pd.read_csv(DATA / 'human_coding.csv')
    scores = ['accommodation_score','hospitality_score','interior_score',
              'visual_score','theme_score']
    require(coding.record_id.is_unique and len(coding) == 3068,
            'Coding must contain 3,068 unique records')
    for col in scores:
        require(coding[col].isin(range(4)).all(), f'Scores must be on the 0-3 scale: {col}')
    core = (coding.accommodation_score >= 2) & (coding.hospitality_score >= 2)
    experience = (coding[scores[2:]] >= 2).sum(axis=1) >= 2
    computed = np.where(~core,'Traditional Camping',np.where(experience,'Glamping','Mixed'))
    require((computed == coding.gold_verdict).all(), 'Review eligibility criteria verification failed')

    raw = pd.read_csv(DATA / 'raw_vision_tags.csv').fillna(0)
    raw['record_id'] = raw.spot_id.astype(str) + '_' + raw.num.astype(str)
    require(raw.filename.is_unique, 'Source tag filenames must be unique')
    # Identify all tag columns explicitly. The terminal Chignon tag has one
    # detection, falls below the frequency cutoff, and cannot affect the fit.
    tags = [c for c in raw if c not in ('Unnamed: 0','filename','spot_id','num','record_id')]
    require(len(tags) == 1476, 'Unexpected source tag columns')
    require(np.isin(raw[tags].to_numpy(), [0,1]).all(), 'Tag detections must be binary')
    # Retain all source image records, including identical image copies.
    # Source image rows are the observation units.
    require(len(raw) == 9572, 'Expected all 9,572 source image records')
    candidates = pd.read_csv(DATA / 'candidate_posts.csv')
    require(candidates.record_id.is_unique and len(candidates) == 1559,
            'Expected 1,559 unique machine-positive reviews')
    photos = raw[raw.record_id.isin(candidates.record_id)].copy()
    require(set(photos.record_id) == set(candidates.record_id),
            'Tag data must cover every machine-positive review')
    require(len(photos) == 8965, 'Expected 8,965 fitting image records')
    counts = photos[tags].sum().rename_axis('word').rename('count').to_frame()
    counts = counts.sort_values('count', ascending=False)
    save(counts.reset_index(), 'tag_counts.csv')
    median = float(counts['count'].median())
    vocabulary = counts[counts['count'] > median]
    save(vocabulary.reset_index(), 'tag_vocabulary.csv')
    distance = pd.read_csv(RETAINED / 'tag_wmd_distance.csv', index_col=0)
    require(distance.index.equals(pd.Index(distance.columns)), 'Distance order differs')
    covered = [t for t in distance.index if t in vocabulary.index]
    missing = vocabulary.index.difference(covered)
    print(f'{len(covered)} of {len(vocabulary)} vocabulary tags have WMD vector '
          f'coverage; {len(missing)} excluded: {", ".join(map(str, missing))}',
          flush=True)
    require(len(vocabulary) == 732 and len(covered) == 719,
            'Expected vocabulary must contain 732 tags, including 719 with WMD coverage')
    profiles = distance.loc[covered,covered].to_numpy()
    # Preserve the historical procedure: Euclidean Ward on WMD row profiles.
    # This is not Ward linkage on a precomputed distance metric.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*symmetric non-negative hollow.*')
        groups = AgglomerativeClustering(n_clusters=50, metric='euclidean',
                                         linkage='ward').fit_predict(profiles)
    clusters = pd.DataFrame({'keyword':covered,'cluster_id':groups})
    present = photos[covered].to_numpy() != 0
    first_seen = []
    seen = set()
    for row, column in np.argwhere(present):
        group = int(groups[column])
        if group not in seen:
            seen.add(group)
            first_seen.append(group)
        if len(first_seen) == 50:
            break
    require(len(first_seen) == 50, 'All word clusters must occur')
    cluster_counts = pd.DataFrame({'filename':photos.filename.to_numpy()})
    for group in first_seen:
        cluster_counts[f'group{group}'] = present[:,groups == group].sum(axis=1)
    save(cluster_counts,'photo_cluster_counts.csv')
    x = StandardScaler().fit_transform(cluster_counts.iloc[:,1:])
    np.random.seed(0)
    model = FactorAnalyzer(rotation='varimax', n_factors=16, method='principal',
                           svd_method='randomized')
    model.fit(x)
    assigned = np.abs(model.loadings_).argmax(axis=1)
    fitted = pd.DataFrame(model.loadings_, index=cluster_counts.columns[1:],
                          columns=[f'factor_{k:02d}' for k in range(16)])
    fitted.index.name = 'group'
    save(fitted.assign(assigned_factor=assigned).reset_index(),
         'pca_loadings_fitted.csv')
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(x.T))[::-1]
    save(pd.DataFrame({'component':range(1,51),'eigenvalue':eigenvalues,
                      'variance_percent':eigenvalues / 50 * 100,
                      'cumulative_percent':eigenvalues.cumsum() / 50 * 100}),
         'pca_eigenvalues.csv')
    cluster_to_factor = dict(zip(first_seen, assigned))
    clusters['factor_id'] = clusters.cluster_id.map(cluster_to_factor)
    clusters['occurrences_in_fit'] = clusters.keyword.map(counts['count'])
    save(clusters,'tag_factor_membership.csv')
    factors = pd.DataFrame({'filename':photos.filename.to_numpy(),
                            'record_id':photos.record_id.to_numpy()})
    for k in range(16):
        cols = cluster_counts.columns[1:][assigned == k]
        factors[f'factor_{k:02d}'] = cluster_counts[cols].sum(axis=1).to_numpy()
    save(factors,'photo_factor_counts.csv')
    taxonomy = pd.read_csv(DATA / 'scene_taxonomy_v3.csv')
    require(len(taxonomy) == 15 and taxonomy.scene_column.is_unique,
            'Expected 15 unique V3 scene columns')
    factor_names = pd.read_csv(DATA / 'factor_scene_names.csv')
    require(len(factor_names) == 16 and factor_names.factor_id.is_unique
            and set(factor_names.factor_id) == set(range(16)),
            'Expected the 16 stable factor ids 0-15')
    # Scene membership is data-driven: every covered tag inherits the scene of the
    # factor holding its largest absolute loading (argmax). Factor identities are the
    # stable component ids 0-15; factor-to-scene names follow the original Appendix
    # 16-factor table, with factors 1 and 15 both merging into Food and cooking.
    derived = clusters[['keyword','cluster_id','factor_id']].merge(
        factor_names[['factor_id','scene_name']],on='factor_id',validate='many_to_one')
    require(set(derived.scene_name) == set(taxonomy.display_name),
            'Assigned-factor scenes must cover the 15-scene taxonomy')
    derived = derived.merge(taxonomy[['scene_column','display_name']],
                            left_on='scene_name',right_on='display_name',
                            validate='many_to_one')
    used = derived.sort_values('keyword').reset_index(drop=True)
    used = used[['keyword','scene_column','cluster_id','factor_id']]
    # The 16-factor PCA model is fitted above and scene membership is fully
    # data-driven: every tag inherits the scene of the factor holding its largest
    # absolute loading. No predeclared tag-to-scene codebook is read or consulted;
    # the derived `used` mapping is the sole basis for the scene aggregation below.
    photo_scenes = photos[['filename','record_id']].copy()
    for row in taxonomy.itertuples(index=False):
        columns = used.loc[used.scene_column == row.scene_column,'keyword'].tolist()
        photo_scenes[row.scene_column] = photos[columns].sum(axis=1).to_numpy()
    photo_scenes = photo_scenes[['filename','record_id'] + taxonomy.scene_column.tolist()]
    save(photo_scenes,'photo_scenes.csv')
    reviews = photo_scenes.groupby('record_id')[taxonomy.scene_column.tolist()].sum()
    reviews = reviews.reindex(candidates.record_id).reset_index()
    save(reviews,'review_scenes.csv')
    final_ids = set(candidates.record_id) & set(coding.loc[coding.gold_verdict == 'Glamping','record_id'])
    require(len(final_ids) == 1409, 'Unexpected final review count')
    topics = pd.read_csv(RETAINED / 'lda_topic_assignments.csv')
    require(topics.record_id.is_unique and set(topics.record_id) == set(candidates.record_id),
            'Historical topic assignments do not cover the fitting corpus')
    final = reviews[reviews.record_id.isin(final_ids)].merge(topics, on='record_id',validate='one_to_one')
    save(final,'review_scene_topics.csv')
    final_photo_count = int(photos.record_id.isin(final_ids).sum())
    summary = {'source_tag_rows':len(raw),
               'fitting_reviews':len(candidates),'fitting_photo_rows':len(photos),
               'final_reviews':len(final),'final_photo_rows':final_photo_count,
               'frequency_median':median,'retained_vocabulary':len(vocabulary),
               'vector_covered_tags':len(covered),'word_clusters':50,'retained_factors':16,
               'eigenvalues_above_one':int((eigenvalues > 1).sum()),
               'first_16_variance_percent':float(eigenvalues[:16].sum()/50*100),
               'fitted_factor_variance_percent':float((model.loadings_**2).sum()/50*100),
               'final_scene_occurrences':int(final[taxonomy.scene_column].sum().sum())}
    (out_path('analysis_summary.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
