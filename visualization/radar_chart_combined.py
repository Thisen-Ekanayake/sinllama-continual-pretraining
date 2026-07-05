import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Config: (csv path, title, domain column name) ----
DATASETS = [
    ("benchmark/EnglishMMLU_results_final/domain_accuracy.csv", "English-MMLU Accuracy by Domain"),
    ("benchmark/SinhalaMMLU_results_final/domain_accuracy.csv", "Sinhala-MMLU Accuracy by Domain"),
]

colors = ["#5B6EE1", "#2ECC71", "#F0932B", "#9B59B6", "#E74C3C", "#00CFC1"]

fig, axes = plt.subplots(1, 2, figsize=(16, 9.5), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")

handles, labels = None, None

for ax, (csv_path, title) in zip(axes, DATASETS):
    df = pd.read_csv(csv_path)
    domain_col = "Domain" if "Domain" in df.columns else "domain"
    categories = df[domain_col].tolist()
    models = [c for c in df.columns if c != domain_col]
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_facecolor("#E9ECF5")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color="#2c3e50")

    max_val = df[models].values.max()
    r_max = int(np.ceil(max_val / 20.0) * 20)
    ax.set_ylim(0, r_max)
    ax.set_yticks(range(0, r_max + 1, 20))
    ax.set_yticklabels(range(0, r_max + 1, 20), fontsize=9, color="#555")
    ax.set_rlabel_position(0)

    ax.grid(color="white", linewidth=1)
    ax.spines["polar"].set_visible(False)

    for i, model in enumerate(models):
        values = df[model].tolist()
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, color=color, linewidth=2, marker="o", markersize=5, label=model)

    ax.set_title(title, fontsize=16, color="#2c3e50", pad=30)

    if handles is None:
        handles, labels = ax.get_legend_handles_labels()

# ---- Overall figure title ----
fig.suptitle("MMLU Benchmark Results Across Languages", fontsize=22, color="#2c3e50", y=0.98)

# ---- Single shared legend, in one row (sequential, not stacked) ----
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=len(labels),
    frameon=False,
    fontsize=11,
)

plt.tight_layout(rect=[0, 0.08, 1, 0.88])
out_path = "visualization/MMLU_domain_accuracy_radar_charts.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.6, facecolor="white")
print(f"Saved chart to {out_path}")
