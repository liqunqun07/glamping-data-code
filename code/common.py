"""
Common utilities and filesystem configuration for the analysis workflow.

Purpose:
    Defines root project directories (input, retained, output), topic enumerations,
    and reusable helper functions (assertions, table persistence, slug formatting,
    and 15-scene column identification) used across all analysis scripts.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'input'
RETAINED = ROOT / 'retained'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
MAPPING_FILES = {'tag_group_mapping.csv', 'cluster_group_mapping.csv',
                 'tag_factor_membership.csv'}
TABLE_FILES = {'scene_topic_column_percentage.csv', 'dominant_topic_by_scene.csv',
               'top3_scenes_for_topics_without_dominant_scene.csv'}
CNN_FILES = {'cnn_validation_predictions.csv', 'cnn_validation_metrics.csv',
             'cnn_validation_metrics.json'}
TAG_FACTOR_FILES = {'photo_factor_counts.csv', 'tag_vocabulary.csv', 'pca_loadings_fitted.csv', 'photo_cluster_counts.csv', 'pca_eigenvalues.csv', 'tag_counts.csv', 'analysis_summary.json', 'scree_parallel_analysis.csv'}
SCENE_TOPIC_FILES = {'photo_scenes.csv', 'review_scenes.csv', 'review_scene_topics.csv'}
CAMPSITE_FILES = {'campsite_name_group_membership.csv', 'campsite_name_group_summary.csv'}
TOPICS = list(range(1, 10))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def out_path(name):
    # Keep the output tree predictable for both generated files and downstream reads.
    if Path(name).name != name:
        raise ValueError('Output name must be a filename without directories')
    if name in MAPPING_FILES:
        folder = 'mappings'
    elif name.startswith('table') and name.endswith('.csv') or name in TABLE_FILES:
        folder = 'tables'
    elif name.endswith('.png'):
        folder = 'figures'
    elif name in CNN_FILES:
        folder = 'cnn_validation'
    elif name in TAG_FACTOR_FILES:
        folder = 'intermediate/tag_factor_analysis'
    elif name in SCENE_TOPIC_FILES:
        folder = 'intermediate/scene_topic_analysis'
    elif name in CAMPSITE_FILES:
        folder = 'intermediate/campsite_analysis'
    else:
        folder = 'intermediate/other'
    target = OUT / folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

def save(table, name):
    path = out_path(name)
    table.to_csv(path, index=False)
    print(f'Wrote {path.relative_to(ROOT)}: {len(table):,} rows', flush=True)

def slug(label):
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')

def scene_columns(table):
    columns = [c for c in table if re.match(r'^fac\d+_', c)]
    require(len(columns) == 15, 'Expected 15 interpreted scene columns')
    return columns
