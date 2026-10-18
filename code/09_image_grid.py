"""Render representative validation photograph comparison grid (Appendix Figure F3).

Purpose:
    Renders a 3x4 multi-panel visual grid of representative validation photographs:
    1. Selects 6 representative glamping photographs and 6 representative camping photographs.
    2. Displays each photograph with its predicted class, true label, and confidence.
    3. Provides qualitative visual evidence of model classification behavior and hardware distinctions.

Input Files:
    - output/cnn_validation/cnn_validation_predictions.csv: Model predictions and true labels for validation photos.
    - input/cnn_images/valid/: Original validation photographs from glamping ('0') and camping ('1').

Output Files:
    - output/figures/plot2_v2_representative_grid.png: Publication-ready 12-image comparison grid figure (500 dpi).
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from common import DATA, RETAINED, OUT, out_path
BASE_DIR = str(DATA)

df = pd.read_csv(out_path('cnn_validation_predictions.csv'))
df_by_path = df.set_index("image_path")

glamping_paths = [
    "cnn_images/valid/0/209.jpg",
    "cnn_images/valid/0/132.jpg",
    "cnn_images/valid/0/99.jpg",
    "cnn_images/valid/0/505.jpg",
    "cnn_images/valid/0/632.jpg",
    "cnn_images/valid/0/463.jpg",
]
camping_paths = [
    "cnn_images/valid/1/54.jpg",
    "cnn_images/valid/1/507.jpg",
    "cnn_images/valid/1/223.jpg",
    "cnn_images/valid/1/60.jpg",
    "cnn_images/valid/1/500.jpg",
    "cnn_images/valid/1/669.jpg",
]

n_cols_per_side = 2
n_rows = len(glamping_paths) // n_cols_per_side  # 3

fig = plt.figure(figsize=(16, 12.5))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

def draw_image(ax, full_path):
    row = df_by_path.loc[str(Path(full_path).relative_to(DATA))]
    img = Image.open(full_path).convert("RGB")
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    title = f"{row['pred_label']} ({row['true_label']})"
    ax.set_title(title, color="green", fontsize=13)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

for r in range(n_rows):
    for c in range(n_cols_per_side):
        g_rel = glamping_paths[r * n_cols_per_side + c]
        pos = r * 4 + c + 1
        ax = fig.add_subplot(n_rows, 4, pos)
        draw_image(ax, os.path.join(BASE_DIR, g_rel))

        c_rel = camping_paths[r * n_cols_per_side + c]
        pos2 = r * 4 + n_cols_per_side + c + 1
        ax2 = fig.add_subplot(n_rows, 4, pos2)
        draw_image(ax2, os.path.join(BASE_DIR, c_rel))

fig.suptitle("", fontsize=1)  # placeholder to reserve top margin for group labels
plt.tight_layout(rect=[0, 0, 1, 0.94])

# Image visualization using the supplied model and validation data.
fig.text(0.27, 0.965, "Glamping", ha="center", va="center", fontsize=20, fontweight="bold")
fig.text(0.73, 0.965, "Camping", ha="center", va="center", fontsize=20, fontweight="bold")
# Image visualization using the supplied model and validation data.
fig.add_artist(plt.Line2D([0.5, 0.5], [0.02, 0.955], color="black", linewidth=1.2, transform=fig.transFigure))

fig.savefig(out_path("plot2_v2_representative_grid.png"), dpi=500, facecolor="white", bbox_inches="tight")
print("Saved plot2_v2_representative_grid.png")
