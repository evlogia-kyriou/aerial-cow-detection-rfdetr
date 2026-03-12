import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
LOG_PATH   = r"D:\Demo\RF-DETR\output\first\log.txt"
OUTPUT_IMG = r"D:\Demo\RF-DETR\output\first\training_progress.png"
# ─────────────────────────────────────────────────────────────────────────────

epochs, ap50, ap5095, f1, precision, recall, train_loss, val_loss = [], [], [], [], [], [], [], []

with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        ep = d.get("epoch", None)
        if ep is None:
            continue

        res = d.get("test_results_json", {})
        class_map = res.get("class_map", [{}])
        cow = next((c for c in class_map if c.get("class") == "cow"), class_map[0] if class_map else {})

        epochs.append(ep + 1)
        ap50.append(res.get("map", 0) * 100)
        ap5095.append(cow.get("map@50:95", 0) * 100)
        f1.append(res.get("f1_score", 0) * 100)
        precision.append(res.get("precision", 0) * 100)
        recall.append(res.get("recall", 0) * 100)
        train_loss.append(d.get("train_loss", None))
        val_loss.append(d.get("test_loss", None))

epochs     = np.array(epochs)
ap50       = np.array(ap50)
ap5095     = np.array(ap5095)
f1         = np.array(f1)
precision  = np.array(precision)
recall     = np.array(recall)
train_loss = np.array(train_loss, dtype=float)
val_loss   = np.array(val_loss,   dtype=float)

# ── PLOT ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(12, 22))
fig.suptitle("RF-DETR Training Progress — Aerial Cow Detection", fontsize=15, fontweight="bold", y=0.99)

BLUE   = "#2563EB"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
RED    = "#DC2626"
PURPLE = "#7C3AED"
TEAL   = "#0D9488"

def shade(ax, x, y, color, alpha=0.12):
    ax.fill_between(x, 0, y, color=color, alpha=alpha)

def annotate_best(ax, x, y, color, label="Best"):
    idx = np.argmax(y)
    ax.axhline(y[idx], color=color, linestyle="--", linewidth=0.8, alpha=0.5)
    offset_x = x[idx] + max(1, len(x) * 0.03)
    ax.annotate(f"{label}: {y[idx]:.1f}% @ ep {x[idx]}",
                xy=(x[idx], y[idx]), xytext=(offset_x, y[idx] - 5),
                fontsize=8.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

# Panel 1 — AP@0.50 + AP@0.50:0.95
ax = axes[0]
ax.plot(epochs, ap50,   color=BLUE,  linewidth=2, marker="o", markersize=3, label="AP@0.50")
ax.plot(epochs, ap5095, color=TEAL,  linewidth=2, marker="s", markersize=3, label="AP@0.50:0.95", linestyle="--")
shade(ax, epochs, ap50, BLUE)
annotate_best(ax, epochs, ap50, BLUE, "Best AP@0.50")
ax.set_ylabel("AP (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_title("Detection Accuracy (AP)", fontsize=11)

# Panel 2 — F1
ax = axes[1]
ax.plot(epochs, f1, color=PURPLE, linewidth=2, marker="o", markersize=3, label="F1 Score")
shade(ax, epochs, f1, PURPLE)
annotate_best(ax, epochs, f1, PURPLE, "Best F1")
ax.set_ylabel("F1 Score (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_title("F1 Score", fontsize=11)

# Panel 3 — Precision
ax = axes[2]
ax.plot(epochs, precision, color=GREEN, linewidth=2, marker="o", markersize=3, label="Precision")
shade(ax, epochs, precision, GREEN)
annotate_best(ax, epochs, precision, GREEN, "Best Prec")
ax.set_ylabel("Precision (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_title("Precision", fontsize=11)

# Panel 4 — Recall
ax = axes[3]
ax.plot(epochs, recall, color=ORANGE, linewidth=2, marker="o", markersize=3, label="Recall")
shade(ax, epochs, recall, ORANGE)
annotate_best(ax, epochs, recall, ORANGE, "Best Recall")
ax.set_ylabel("Recall (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_title("Recall", fontsize=11)

# Panel 5 — Loss
ax = axes[4]
ax.plot(epochs, train_loss, color=RED,  linewidth=2, marker="o", markersize=3, label="Train Loss")
ax.plot(epochs, val_loss,   color=BLUE, linewidth=2, marker="s", markersize=3, label="Val Loss", linestyle="--")
ax.set_ylabel("Loss", fontsize=11)
ax.set_xlabel("Epoch", fontsize=11)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_title("Train vs. Validation Loss", fontsize=11)

for ax in axes:
    ax.set_xlim(1, epochs.max())

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight")
print(f"Saved → {OUTPUT_IMG}")

# ── SUMMARY ─────────────────────────────────────────────────────────────────
print(f"\n{'Epoch':>6} {'AP@50':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
print("-" * 44)
for i in range(len(epochs)):
    print(f"{epochs[i]:>6} {ap50[i]:>8.1f} {f1[i]:>8.1f} {precision[i]:>8.1f} {recall[i]:>8.1f}")