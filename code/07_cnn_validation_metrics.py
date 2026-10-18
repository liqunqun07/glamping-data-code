"""Summarize validation performance metrics of the CNN image classifier (Appendix Figure F1).

Purpose:
    Computes performance metrics and renders the confusion matrix for the ResNet18 classifier:
    1. Computes True Positives (TP), False Negatives (FN), False Positives (FP), True Negatives (TN).
    2. Calculates per-class precision, recall, and F1 scores, overall accuracy (80.74%), and weighted F1 (80.5%).
    3. Incorporates historical 5-fold cross-validation results (mean 77.61%, std 1.13%).
    4. Renders the publication-ready confusion matrix figure with cell counts and percentages.

Input Files:
    - output/cnn_validation/cnn_validation_predictions.csv: Scored predictions on 431 validation images.
    - retained/cnn_five_fold_accuracy.csv: 5-fold cross-validation accuracy records.

Output Files:
    - output/cnn_validation/cnn_validation_metrics.csv: Precision, recall, F1, and support for Glamping and Camping classes.
    - output/cnn_validation/cnn_validation_metrics.json: JSON metrics summary, confusion matrix counts, and 5-fold stats.
    - output/figures/figure_cnn_confusion_matrix.png: Visual confusion matrix figure (1100 dpi, 3762 px wide).
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import DATA, RETAINED, OUT, out_path, require, save

CELL_SHADES = [0.12, 0.95, 0.81, 0.36]
FIG_W, FIG_H = 3.42, 2.893
AXES_RECT = [0.2317, 0.1429, 0.6894, 0.8157]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"


def class_scores(tp, fn, fp):
    precision = 100 * tp / (tp + fp)
    recall = 100 * tp / (tp + fn)
    return precision, recall, 2 * precision * recall / (precision + recall)


def main():
    source = out_path('cnn_validation_predictions.csv')
    predictions = pd.read_csv(source)
    require(len(predictions) == 431, "Expected 431 validation predictions")
    require(set(predictions.true_label) == {"glamping", "camping"},
            "Unexpected validation labels")
    tp = int(((predictions.true_label == "glamping") &
              (predictions.pred_label == "glamping")).sum())
    fn = int(((predictions.true_label == "glamping") &
              (predictions.pred_label == "camping")).sum())
    fp = int(((predictions.true_label == "camping") &
              (predictions.pred_label == "glamping")).sum())
    tn = int(((predictions.true_label == "camping") &
              (predictions.pred_label == "camping")).sum())
    accuracy = 100 * (tp + tn) / len(predictions)
    g_prec, g_rec, g_f1 = class_scores(tp, fn, fp)
    c_prec, c_rec, c_f1 = class_scores(tn, fp, fn)
    n_glamping, n_camping = tp + fn, fp + tn
    weighted_f1 = (n_glamping * g_f1 + n_camping * c_f1) / 431

    folds = pd.read_csv(RETAINED / "cnn_five_fold_accuracy.csv")
    require(len(folds) == 5, "Expected five cross-validation folds")
    mean = 100 * folds.validation_accuracy.mean()
    std = 100 * folds.validation_accuracy.std(ddof=0)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AXES_RECT)
    cells = [[(tp, g_rec), (fn, 100 - g_rec)], [(fp, 100 - c_rec), (tn, c_rec)]]
    ax.imshow([CELL_SHADES[:2], CELL_SHADES[2:]], cmap="gray", vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            value, pct = cells[row][column]
            color = "white" if CELL_SHADES[2 * row + column] < .5 else "black"
            ax.text(column, row, f"{value}\n({pct:.1f}%)", ha="center",
                    va="center", fontsize=8.8, color=color, linespacing=1.2)
    ax.set_xticks([0, 1], ["Glamping", "Camping"], fontsize=8.8)
    ax.set_yticks([0, 1], ["Glamping", "Camping"], fontsize=8.8)
    ax.set_xlabel("Predicted label", fontsize=8.8)
    ax.set_ylabel("True label", fontsize=8.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.savefig(out_path('figure_cnn_confusion_matrix.png'), dpi=1100)
    plt.close(fig)

    save(pd.DataFrame([
        {"class": "Glamping", "precision_percent": round(g_prec, 1),
         "recall_percent": round(g_rec, 1), "f1_percent": round(g_f1, 1),
         "n": n_glamping},
        {"class": "Camping", "precision_percent": round(c_prec, 1),
         "recall_percent": round(c_rec, 1), "f1_percent": round(c_f1, 1),
         "n": n_camping},
    ]), "cnn_validation_metrics.csv")
    (out_path('cnn_validation_metrics.json')).write_text(json.dumps({
        "n_validation": 431, "confusion_TP_FN_FP_TN": [tp, fn, fp, tn],
        "overall_accuracy_percent": round(accuracy, 2),
        "weighted_f1_percent": round(weighted_f1, 1),
        "five_fold_accuracy_percent": [round(100 * value, 2)
                                       for value in folds.validation_accuracy],
        "five_fold_mean_percent": round(mean, 2),
        "five_fold_std_pp_population": round(std, 2)}, indent=2), encoding="utf-8")
    print(f"CNN validation metrics: confusion {tp}/{fn}/{fp}/{tn}, accuracy "
          f"{accuracy:.2f}%, weighted F1 {weighted_f1:.1f}%, five-fold mean "
          f"{mean:.2f}% (sd {std:.2f}pp).", flush=True)


if __name__ == "__main__":
    main()
