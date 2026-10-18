"""Build the group-ID mapping tables linking tag vocabulary, word clusters,
factor (PCA) IDs and names, ritual scenes, and taxonomy subclasses.

Purpose:
    Provides a single, code-generated lookup that resolves every identifier used
    across the analysis outputs:
    1. keyword-level mapping: each of the 719 vector-covered tags to its
       frequency count, word cluster (group), assigned PCA factor, factor name,
       ritual scene, and the taxonomy subclass that its group belongs to.
    2. cluster-level mapping: the 50 word clusters (group0..group49) to their
       assigned factor, factor name, scene, taxonomy subclass, keyword count,
       and total occurrences in the fitting and final photo sets.

    The group -> subclass binding is NOT data-driven: the Ward word clusters and
    the varimax factors are fitted from the WMD/PCA artifacts, but the taxonomy
    subclasses are a stable manuscript codebook, and no fitted quantity assigns a
    cluster to a specific subclass. The semantic cluster-to-subclass assignment
    is therefore supplied as an explicit, human-reviewed codebook
    (input/group_subclass_assignment_v1.csv) that maps each of the 50 clusters to
    exactly one of the 50 taxonomy subclass names. Every subclass name is drawn
    verbatim from input/scene_taxonomy_v3.csv, so the output names match the
    manuscript Appendix Table 1 exactly.

Input Files:
    - output/intermediate/tag_factor_analysis/tag_counts.csv: Occurrence counts for all 1,476 detected tags.
    - output/mappings/tag_factor_membership.csv: Keyword to cluster and assigned factor.
    - input/factor_scene_names.csv: Stable factor ids 0-15 and their names.
    - input/scene_taxonomy_v3.csv: Scene taxonomy with subclasses.
    - input/group_subclass_assignment_v1.csv: Human-reviewed cluster -> subclass
      codebook (50 rows keyed by group, with a scene, factor, and subclass).

Output Files:
    - output/mappings/tag_group_mapping.csv: Keyword-level mapping (one row per keyword;
      719 unique keywords, no subclass expansion).
    - output/mappings/cluster_group_mapping.csv: Cluster-level mapping (50 rows).
"""

import pandas as pd

from common import DATA, require, save, out_path


def main():
    counts = pd.read_csv(out_path('tag_counts.csv'))
    require(counts.word.is_unique and len(counts) == 1476,
            'Expected 1,476 unique source tags')
    count_of = dict(zip(counts.word, counts['count']))

    membership = pd.read_csv(out_path('tag_factor_membership.csv'))
    require(membership.keyword.is_unique and len(membership) == 719,
            'Expected 719 unique vector-covered keywords')
    membership['group'] = 'group' + membership.cluster_id.astype(str)

    factor_names = pd.read_csv(DATA / 'factor_scene_names.csv')
    require(len(factor_names) == 16 and factor_names.factor_id.is_unique
            and set(factor_names.factor_id) == set(range(16)),
            'Expected the 16 stable factor ids 0-15')

    taxonomy = pd.read_csv(DATA / 'scene_taxonomy_v3.csv')
    require(len(taxonomy) == 15 and taxonomy.scene_column.is_unique,
            'Expected 15 unique V3 scene columns')
    # Valid subclass names per scene, taken verbatim from the taxonomy codebook.
    subclasses_of = dict(zip(taxonomy.display_name, taxonomy.subclasses))
    valid_subclass = {
        sc for subs in subclasses_of.values() for sc in str(subs).split('; ')
    }

    assignment = pd.read_csv(DATA / 'group_subclass_assignment_v1.csv')
    # group_subclass_assignment_v1.csv is keyed by group; derive the matching
    # cluster_id so the join with the membership frame stays explicit.
    assignment['cluster_id'] = assignment.group.str.replace('group', '', regex=False).astype(int)
    require(len(assignment) == 50 and assignment.group.is_unique,
            'Expected 50 unique cluster assignments')
    require(set(assignment.group) == {f'group{i}' for i in range(50)},
            'Cluster assignment must cover groups group0..group49')
    # Every assigned subclass must be a real taxonomy subclass of its scene.
    for row in assignment.itertuples(index=False):
        scene = subclasses_of.get(row.scene)
        require(scene is not None and row.subclass in str(scene).split('; '),
                f'{row.group}: subclass "{row.subclass}" is not a taxonomy '
                f'subclass of scene "{row.scene}"')
    subclass_by_group = dict(zip(assignment.group, assignment.subclass))

    derived = membership.merge(factor_names, on='factor_id', validate='many_to_one')
    require(set(derived.scene_name) == set(taxonomy.display_name),
            'Assigned-factor scenes must cover the 15-scene taxonomy')
    # Carry the single subclass for each keyword via its group.
    derived['subclass'] = derived.group.map(subclass_by_group)
    require(derived.subclass.notna().all(),
            'Every keyword group must resolve to one taxonomy subclass')

    # The codebook must describe the fitted group, not merely use valid labels.
    for row in assignment.itertuples(index=False):
        members = derived.loc[derived.group == row.group]
        require(len(members) == int(row.n_examples),
                f'{row.group}: keyword count differs from the codebook')
        require(set(members.keyword) == set(str(row.examples).split('; ')),
                f'{row.group}: keyword list differs from the codebook')
        require(int(members.occurrences_in_fit.sum()) == int(row.subclass_count),
                f'{row.group}: occurrence count differs from the codebook')
        require(members.factor_id.nunique() == 1
                and int(members.factor_id.iloc[0]) == int(row.factor_id)
                and members.scene_name.nunique() == 1
                and members.scene_name.iloc[0] == row.scene,
                f'{row.group}: factor or scene differs from the fitted assignment')

    # Keyword level: one row per keyword (no subclass expansion), so no keyword
    # is repeated. The 719 keywords stay 719 rows.
    keyword_map = derived[['keyword', 'cluster_id', 'group', 'factor_id',
                           'factor_name', 'scene_name', 'subclass']].copy()
    keyword_map.insert(1, 'candidate_1559_tag_count', keyword_map.keyword.map(count_of))
    keyword_map = keyword_map.rename(columns={'scene_name': 'scene'})
    keyword_map = keyword_map.sort_values(
        ['factor_id', 'subclass', 'keyword']).reset_index(drop=True)
    require(len(keyword_map) == 719 and keyword_map.keyword.is_unique,
            'Keyword mapping must be 719 unique rows, one per keyword')
    save(keyword_map, 'tag_group_mapping.csv')

    # Cluster level: one row per word cluster (group0..group49).
    cluster_rows = []
    for cluster_id, group_rows in derived.groupby('cluster_id'):
        head = group_rows.iloc[0]
        cluster_rows.append({
            'cluster_id': cluster_id,
            'group': f'group{cluster_id}',
            'n_keywords': len(group_rows),
            'factor_id': head.factor_id,
            'factor_name': head.factor_name,
            'scene_name': head.scene_name,
            'subclass': head.subclass,
            'candidate_1559_occurrences': int(group_rows.occurrences_in_fit.sum()),
        })
    cluster_map = pd.DataFrame(cluster_rows).sort_values('cluster_id').reset_index(drop=True)
    require(len(cluster_map) == 50 and set(cluster_map.group) ==
            {f'group{i}' for i in range(50)},
            'Cluster mapping must cover the 50 fitted word clusters')
    require(cluster_map.factor_id.nunique() == 16,
            'Cluster mapping must cover all 16 factors')
    # Subclass counts must sum to each factor's total occurrences.
    fac_total = derived.groupby('factor_id').occurrences_in_fit.sum()
    sub_by_fac = cluster_map.groupby('factor_id').candidate_1559_occurrences.sum()
    require(all(int(sub_by_fac[k]) == int(fac_total[k]) for k in fac_total.index),
            'Per-factor subclass counts must sum to the factor total')
    # Appendix Table 1 uses the final 1,409 reviews; expose that basis beside
    # the 1,559-review fitting counts so the two cannot be confused.
    photo_groups = pd.read_csv(out_path('photo_cluster_counts.csv'))
    final_reviews = pd.read_csv(out_path('review_scene_topics.csv'))
    photo_record_ids = photo_groups.filename.str.rsplit('_pic', n=1).str[0]
    final_photos = photo_groups.loc[photo_record_ids.isin(final_reviews.record_id)]
    require(len(final_reviews) == 1409 and len(final_photos) == 8330,
            'Expected 1,409 final reviews and 8,330 matching photographs')
    group_counts = final_photos[[f'group{i}' for i in range(50)]].sum()
    cluster_map['final_1409_occurrences'] = cluster_map.group.map(group_counts).astype(int)
    require(int(cluster_map.final_1409_occurrences.sum()) == 81764,
            'Expected 81,764 final-review tag occurrences')
    appendix_scenes = pd.read_csv(out_path('table1_scene_frequencies.csv'))
    scene_counts = cluster_map.groupby('scene_name').final_1409_occurrences.sum()
    require(all(int(scene_counts[row.scene_name]) == int(row.frequency)
                for row in appendix_scenes.itertuples(index=False)),
            'Final group totals must equal all 15 Appendix scene frequencies')
    save(cluster_map, 'cluster_group_mapping.csv')


if __name__ == '__main__':
    main()
