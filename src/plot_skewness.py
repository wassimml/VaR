import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
x = np.linspace(-5, 8, 1000)

normal      = stats.norm.pdf(x, 0, 1)
right_skew  = stats.lognorm.pdf(x, s=0.9, loc=-1.2, scale=1.5)
left_skew   = stats.lognorm.pdf(-x + 3, s=0.9, loc=-1.2, scale=1.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle("Effet du Skewness sur la distribution des rendements", fontsize=14, fontweight="bold")

configs = [
    (left_skew,  "Skew < 0\n(Asymétrie négative)",  "#e74c3c", ""),
    (normal,     "Skew = 0\n(Distribution normale)", "#2ecc71", ""),
    (right_skew, "Skew > 0\n(Asymétrie positive)",   "#3498db", ""),
]

for ax, (dist, title, color, note) in zip(axes, configs):
    ax.plot(x, dist, color=color, linewidth=2.5)
    ax.fill_between(x, dist, alpha=0.25, color=color)
    ax.fill_between(x, dist, where=(x < -1.5), alpha=0.5, color="#e74c3c", label="Zone pertes extrêmes")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(title, fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("Rendements")
    ax.set_ylabel("Densité")
    ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center", fontsize=9,
            style="italic", color="gray")
    ax.set_xlim(-5, 8)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=2)
plt.savefig("reports/skewness_illustration.png", dpi=150, bbox_inches="tight")
plt.show()

print("Skewness des distributions simulées :")
for name, sample in [("Skew négatif", -np.random.lognormal(0, 0.9, 10000) + 3),
                     ("Normal",        np.random.normal(0, 1, 10000)),
                     ("Skew positif",  np.random.lognormal(0, 0.9, 10000) - 1.2)]:
    print(f"  {name:15s} → skew = {stats.skew(sample):+.3f}")
