"""Build campsite-level distribution of confirmed reviews and name analysis (Table 5).

Purpose:
    Processes all 3,068 eligible reviews across 190 campsites:
    1. Cross-tabulates machine screening (1,559 CNN-positive) and manual verification (GSHVP).
    2. Constructs Table 5: Per-campsite counts and percentages for Glamping (1,409),
       Traditional Camping (1,629), and Mixed / Insufficient Evidence (30).
    3. Evaluates campsite naming patterns:
       - Premium group: Names containing "luxury", "boutique", or "glamping" in English.
       - Mutually exclusive landscape group: Names containing natural mountain/scenic characters.
    4. Computes review-weighted glamping conversion rates across naming groups.

Input Files:
    - input/human_coding.csv: Manual coding decisions for all 3,068 reviews.
    - input/candidate_posts.csv: Record IDs of the 1,559 CNN-positive reviews.
    - input/campsite_display_order.csv: Canonical display order and English campsite names.
    - input/campsite_names.json: English-to-Chinese original campsite names mapping.

Output Files:
    - output/tables/table5_campsite_distribution.csv: Table 5 (190 campsites with review breakdown).
    - output/intermediate/campsite_analysis/campsite_name_group_membership.csv: Binary indicator flags for campsite naming groups.
    - output/intermediate/campsite_analysis/campsite_name_group_summary.csv: Summary counts, glamping reviews, and percentages.
"""

from collections import Counter, defaultdict
import json
import re

import pandas as pd

from common import DATA, RETAINED, OUT, out_path, require, save


def main():
    coding = pd.read_csv(DATA / "human_coding.csv")
    candidates = pd.read_csv(DATA / "candidate_posts.csv", usecols=["record_id"])
    order = pd.read_csv(DATA / "campsite_display_order.csv")
    require(len(coding) == 3068, "Expected all 3,068 human-coded reviews")
    require(len(order) == 190, "Expected the 190-campsite display order")
    cnn_positive = set(candidates.record_id)
    require(len(cnn_positive) == 1559, "Expected the 1,559 CNN-positive reviews")

    counts = defaultdict(Counter)
    for row in coding.itertuples(index=False):
        if row.record_id in cnn_positive and row.gold_verdict == "Glamping":
            category = "Glamping"
        elif row.gold_verdict == "Mixed":
            category = "Mixed"
        else:
            category = "Traditional"
        counts[row.campsite_name_en][category] += 1
    require(len(counts) == 190, f"Expected 190 campsites, found {len(counts)}")

    rows = []
    for entry in order.itertuples(index=False):
        counter = counts[entry.campsite_name_en]
        total = sum(counter.values())
        rows.append({"no": entry.no, "campsite_name_en": entry.campsite_name_en,
                     "eligible_reviews": total,
                     **{f"{key.lower()}_n_pct":
                        f"{counter[key]} ({counter[key] / total * 100:.2f}%)"
                        for key in ("Glamping", "Traditional", "Mixed")}})
    table = pd.DataFrame(rows)
    totals = {key: sum(int(row[f"{key.lower()}_n_pct"].split(" (")[0])
                       for row in rows)
              for key in ("Glamping", "Traditional", "Mixed")}
    require(sum(totals.values()) == 3068,
            "Every coded review must fall in exactly one campsite column")
    save(table, "table5_campsite_distribution.csv")
    names = pd.DataFrame(json.loads((DATA / 'campsite_names.json').read_text()))
    require(names.campsite_name_en.is_unique and set(names.campsite_name_en) == set(counts),
            'Original-name map must cover every campsite')
    names['premium_name'] = names.campsite_name_en.str.contains(
        'luxury|boutique|glamping',case=False,regex=True)
    names['landscape_original_name'] = names.original_name.str.contains(
        '\u5c71|\u666f\u533a',regex=True)
    names['landscape_english_name'] = names.campsite_name_en.str.contains(
        'mountain|scenic',case=False,regex=True)
    names['plain_landscape_original_name'] = names.landscape_original_name & ~names.premium_name
    names['plain_landscape_english_name'] = names.landscape_english_name & ~names.premium_name
    save(names.drop(columns='original_name'), 'campsite_name_group_membership.csv')
    summaries = []
    for group in ['premium_name','plain_landscape_original_name',
                  'landscape_original_name','plain_landscape_english_name','landscape_english_name']:
        selected = names.loc[names[group],'campsite_name_en']
        n = {key:sum(counts[name][key] for name in selected)
             for key in ('Glamping','Traditional','Mixed')}
        total = sum(n.values())
        summaries.append({'group':group,'campsites':len(selected),'reviews':total,
                          'glamping_n':n['Glamping'],'traditional_n':n['Traditional'],
                          'mixed_n':n['Mixed'],
                          'glamping_percent':round(100*n['Glamping']/total,2)})
    save(pd.DataFrame(summaries),'campsite_name_group_summary.csv')
    print(f"Campsite distribution complete: 190 campsites; "
          f"{totals['Glamping']:,} Glamping / {totals['Traditional']:,} Traditional "
          f"/ {totals['Mixed']} Mixed.", flush=True)


if __name__ == "__main__":
    main()
