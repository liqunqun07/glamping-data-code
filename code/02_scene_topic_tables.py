"""Build ritual scene and topic co-occurrence tables for confirmed glamping reviews.

Purpose:
    Processes the 1,409 jointly confirmed glamping reviews to generate:
    1. Table 1: Ritual scene frequencies, percentages, and category groupings.
    2. Table 2: Text topic distribution and top-20 representative keywords.
    3. Table 3: 15x9 scene-topic raw co-occurrence frequency matrix.
    4. Table 4: Normalized contribution percentages (within-scene RP and within-topic CP).
    5. Dominant topic mapping per scene and top-3 scenes for unclaimed topics (Topics 1 and 9).
    6. Heatmap visualization of within-scene percentage contributions (RP).

Input Files:
    - output/intermediate/scene_topic_analysis/review_scene_topics.csv: Review-level scene counts and assigned LDA topics.
    - input/scene_taxonomy_v3.csv: 15 ritual scenes taxonomy and category assignments.
    - retained/lda_model_visualization.html: pyLDAvis payload containing keyword log-probabilities.

Output Files:
    - output/tables/table1_scene_frequencies.csv: Table 1 (scene occurrences and percentage of total tags).
    - output/tables/table2_topic_frequencies_keywords.csv: Table 2 (topic frequencies and top keywords).
    - output/tables/table3_scene_topic_frequency.csv: Table 3 (15 scenes x 9 topics co-occurrence counts).
    - output/tables/scene_topic_column_percentage.csv: Column percentage (CP) matrix.
    - output/tables/table4_scene_contribution_rp.csv: Table 4 (Within-scene contribution RP matrix).
    - output/tables/dominant_topic_by_scene.csv: Primary topic holding the highest RP for each scene.
    - output/tables/top3_scenes_for_topics_without_dominant_scene.csv: Top 3 scenes for Topics 1 and 9.
    - output/figures/scene_topic_contribution_heatmap.png: Heatmap of ritual scene contribution by topic.
"""

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import DATA, RETAINED, OUT, out_path, TOPICS, require, save, scene_columns

TOPIC_NAMES = {
    1: "Enjoyment and Convenience", 2: "Exciting and special experience",
    3: "Recreation and relaxation", 4: "Mountaineering and adventure",
    5: "Tasting and feasting", 6: "Refresh and escape",
    7: "Event and celebration", 8: "Photo shoots and showcase",
    9: "Family and friends",
}
SYMBOLISM = {
    1: "Self-renewal", 2: "Self-participation", 3: "Self-renewal",
    4: "Self-participation", 5: "Social connections", 6: "Self-renewal",
    7: "Social connections", 8: "Self-display", 9: "Social connections",
}


def topic_keywords(path, topn=20):
    """Top terms per topic from the saved model visualization (lambda = 1)."""
    html = path.read_text(encoding="utf-8")
    found = re.search(r"var\s+\w+_data\s*=\s*(\{.*?\});", html, re.S)
    require(found is not None, f"Model visualization payload not found in {path.name}")
    columns = json.loads(found.group(1))["tinfo"]
    rows = [dict(zip(columns, values)) for values in zip(*columns.values())]
    keywords = {}
    for topic in TOPICS:
        block = [row for row in rows if row["Category"] == f"Topic{topic}"]
        block.sort(key=lambda row: row["logprob"], reverse=True)
        require(len(block) >= topn, f"Topic{topic} has fewer than {topn} terms")
        keywords[topic] = [row["Term"] for row in block[:topn]]
    return keywords


def main():
    source = out_path("review_scene_topics.csv").parent
    reviews = pd.read_csv(source / "review_scene_topics.csv")
    columns = scene_columns(reviews)
    require(len(reviews) == 1409, "Expected the 1,409 confirmed reviews")
    taxonomy = pd.read_csv(DATA / 'scene_taxonomy_v3.csv')
    require(set(taxonomy.scene_column) == set(columns),
            "Taxonomy scene columns disagree with the review scene columns")

    # ---- Table 1: ritual-scene frequencies ----
    total = int(reviews[columns].to_numpy().sum())
    table1 = pd.DataFrame([{
        "scene_name": row.display_name,
        "main_category": row.main_category,
        "subclasses": row.subclasses,
        "frequency": int(reviews[row.scene_column].sum()),
        "percentage": round(100 * int(reviews[row.scene_column].sum()) / total, 2),
    } for row in taxonomy.itertuples(index=False)])
    save(table1, "table1_scene_frequencies.csv")

    # ---- Table 2: topic frequencies and keywords ----
    keywords = topic_keywords(RETAINED / "lda_model_visualization.html")
    counts = reviews.topic_id.value_counts().reindex(TOPICS, fill_value=0)
    require(int(counts.sum()) == 1409, "Topic counts must cover all reviews")
    table2 = pd.DataFrame([{
        "symbolism": SYMBOLISM[topic],
        "topic": f"Topic{topic}: {TOPIC_NAMES[topic]}",
        "frequency": int(counts[topic]),
        "percentage": round(100 * counts[topic] / 1409, 2),
        "keywords": "; ".join(keywords[topic]),
    } for topic in TOPICS])
    save(table2, "table2_topic_frequencies_keywords.csv")

    # ---- Tables 3 and 4: scene-topic co-occurrence ----
    ordered_columns = taxonomy.scene_column.tolist()
    frequency = reviews.groupby("topic_id")[ordered_columns].sum().reindex(TOPICS, fill_value=0).T
    frequency.index = taxonomy.display_name.tolist()
    frequency.columns = [f"topic_{topic}" for topic in TOPICS]
    frequency.index.name = "scene_name"
    cp = frequency.div(frequency.sum(axis=0), axis=1)
    rp = cp.div(cp.sum(axis=1), axis=0)
    save(frequency.reset_index(), "table3_scene_topic_frequency.csv")
    save((cp * 100).round(3).reset_index(), "scene_topic_column_percentage.csv")
    save((rp * 100).round(3).reset_index(), "table4_scene_contribution_rp.csv")

    # Dominant scene per topic: the topic holding each scene's maximum RP
    # claims that scene; topics with no claiming scene receive a top-three list.
    dominant = pd.DataFrame({
        "scene_name": frequency.index,
        "dominant_topic": rp.idxmax(axis=1).str.replace("topic_", "").astype(int),
        "max_rp_percent": (rp.max(axis=1) * 100).round(3),
    })
    save(dominant, "dominant_topic_by_scene.csv")
    claimed = set(dominant.dominant_topic)
    unclaimed = [topic for topic in TOPICS if topic not in claimed]
    top_rows = []
    for topic in TOPICS:
        column = f"topic_{topic}"
        ranked = rp[column].sort_values(ascending=False)
        if topic in unclaimed:
            for rank in range(3):
                top_rows.append({"topic_id": topic,
                                 "rank": rank + 1,
                                 "scene_name": ranked.index[rank],
                                 "rp_percent": round(100 * ranked.iloc[rank], 3)})
    save(pd.DataFrame(top_rows), "top3_scenes_for_topics_without_dominant_scene.csv")

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(rp * 100, annot=True, fmt=".1f", cmap="BuPu", ax=ax)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Ritual scene")
    fig.tight_layout()
    fig.savefig(out_path('scene_topic_contribution_heatmap.png'), dpi=600)
    plt.close(fig)

    print(f"Scene-topic tables complete: {total:,} scene occurrences; "
          f"topics without a dominant scene: {unclaimed}", flush=True)


if __name__ == "__main__":
    main()
