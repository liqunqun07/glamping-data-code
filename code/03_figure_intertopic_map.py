"""Render the intertopic distance map (Appendix Figure D1) of the retained 9-topic LDA model.

Purpose:
    Renders the two-dimensional multidimensional scaling (MDS) map of the 9 LDA topics:
    1. Extracts topic coordinate positions (PC1, PC2) from the pyLDAvis visualization payload.
    2. Scales circle areas proportionally to each topic's frequency among the 1,409 confirmed reviews.
    3. Colors circles by the four overarching ritual symbolism dimensions:
       - Self-renewal (Topics 1, 3, 6)
       - Self-participation (Topics 2, 4)
       - Social connections (Topics 5, 7, 9)
       - Self-display (Topic 8)

Input Files:
    - retained/lda_model_visualization.html: pyLDAvis payload with topic MDS coordinates (x, y).
    - output/intermediate/scene_topic_analysis/review_scene_topics.csv: Document topic assignments for 1,409 confirmed reviews.

Output Files:
    - output/figures/figure_intertopic_distance_map.png: High-resolution (600 dpi) intertopic distance map figure.
"""

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import DATA, RETAINED, OUT, out_path, require

TOPIC_NAMES = {
    1: "Enjoyment and Convenience", 2: "Exciting and special experience",
    3: "Recreation and relaxation", 4: "Mountaineering and adventure",
    5: "Tasting and feasting", 6: "Refresh and escape",
    7: "Event and celebration", 8: "Photo shoots and showcase",
    9: "Family and friends",
}


def positions_from_html(path):
    html = path.read_text(encoding="utf-8")
    found = re.search(r"var\s+\w+_data\s*=\s*(\{.*?\});", html, re.S)
    require(found is not None, f"Topic coordinates are absent from {path.name}")
    positions = json.loads(found.group(1))["mdsDat"]
    require(positions["topics"] == list(range(1, 10)), "Expected nine ordered topics")
    return positions


def main():
    source = out_path("review_scene_topics.csv").parent
    reviews = pd.read_csv(source / "review_scene_topics.csv",
                          usecols=["record_id", "topic_id"])
    require(len(reviews) == 1409, "Expected the 1,409 confirmed reviews")
    counts = reviews.topic_id.value_counts().reindex(range(1, 10), fill_value=0)
    require(int(counts.sum()) == 1409, "Circles must use all confirmed reviews")
    positions = positions_from_html(RETAINED / "lda_model_visualization.html")

    fig = plt.figure(figsize=(11, 10.3))
    ax = fig.add_axes([.09, .29, .83, .59])
    for index, topic in enumerate(positions["topics"]):
        x, y = positions["x"][index], positions["y"][index]
        ax.scatter(x, y, s=int(counts[topic]) * 65, facecolor="#c5dff7",
                   edgecolor="#376b9d", linewidth=1.3, zorder=3)
        ax.text(x, y, str(topic), ha="center", va="center", fontsize=13,
                fontweight="bold", zorder=4)
    ax.axhline(0, color="0.75", lw=.7, zorder=1)
    ax.axvline(0, color="0.75", lw=.7, zorder=1)
    ax.set_xlim(min(positions["x"]) - .08, max(positions["x"]) + .08)
    ax.set_ylim(min(positions["y"]) - .08, max(positions["y"]) + .08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    for index, topic in enumerate(range(1, 10)):
        column = 0 if index < 5 else 1
        row = index if index < 5 else index - 5
        x = .09 if column == 0 else .53
        y = .245 - row * .038
        fig.text(x, y, f"{topic}  {TOPIC_NAMES[topic]} "
                       f"({counts[topic] / 1409 * 100:.1f}%)", fontsize=9)
    fig.savefig(out_path('figure_intertopic_distance_map.png'), dpi=600,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure_intertopic_distance_map.png with topic counts "
          f"{counts.tolist()}", flush=True)


if __name__ == "__main__":
    main()
