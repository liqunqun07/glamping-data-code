"""
Scene frequency publication figure.

Purpose:
    Generates the high-resolution publication figure for ritual scene distributions:
    Figure 4: Frequency distribution bar chart of 15 ritual scenes grouped by
    environmental, design, and activity dimensions.

Input Files:
    - output/tables/table1_scene_frequencies.csv: Scene occurrences and category groupings.

Output Files:
    - output/figures/figure4_scene_frequencies.png: Grouped frequency bar chart (600 dpi).
"""
import textwrap
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import OUT, out_path

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif']})

def main():
    scenes = pd.read_csv(out_path('table1_scene_frequencies.csv'))
    grouped = []
    for category in ['Surrounding environment', 'Design factor', 'Activity']:
        grouped.extend(scenes[scenes.main_category == category].sort_values('frequency', ascending=False).to_dict('records'))
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, row in enumerate(grouped):
        kind = row['main_category']
        ax.bar(i, row['frequency'], color='black' if kind == 'Design factor' else 'white', edgecolor='black', hatch='//////' if kind == 'Surrounding environment' else '//' if kind == 'Activity' else None, width=.7)
        ax.text(i, row['frequency'] + 320, f"{row['frequency']:,}", ha='center', fontsize=10)
    ax.set_xticks(range(15), [textwrap.fill(row['scene_name'], 24) for row in grouped], rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 35000)
    ax.set_ylabel('Scene tag occurrences', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    from matplotlib.ticker import StrMethodFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.legend(handles=[Rectangle((0,0),1,1,facecolor='white',edgecolor='black',hatch='//////',label='Environmental'), Rectangle((0,0),1,1,facecolor='black',label='Design-related'), Rectangle((0,0),1,1,facecolor='white',edgecolor='black',hatch='//',label='Activity-related')], loc='upper center', bbox_to_anchor=(.5, -.42), ncol=3, frameon=False, fontsize=12)
    fig.subplots_adjust(bottom=.4, left=.08, right=.99, top=.97)
    fig.savefig(out_path('figure4_scene_frequencies.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('Wrote Figure 4 from the current scene frequency table.',flush=True)

if __name__ == '__main__':
    main()
