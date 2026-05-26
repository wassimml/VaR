import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
x = np.linspace(-6, 6, 1000)

platy  = stats.uniform.pdf(x, -3, 6)
normal = stats.norm.pdf(x, 0, 1)
lepto  = stats.t.pdf(x, df=4)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Effet du Kurtosis sur l'épaisseur des queues de distribution", fontsize=14, fontweight="bold")

configs = [
    (platy,  "Platykurtique\n(Excess Kurt < 0)", "#9b59b6", ""),
    (normal, "Mésokurtique\n(Excess Kurt = 0)",  "#2ecc71", ""),
    (lepto,  "Leptokurtique\n(Excess Kurt > 0)", "#e74c3c", ""),
]

for ax, (dist, title, color, note) in zip(axes, configs):
    ax.plot(x, dist, color=color, linewidth=2.5)
    ax.fill_between(x, dist, alpha=0.2, color=color)
    ax.plot(x, normal, color="gray", linewidth=1.2, linestyle="--", alpha=0.6, label="Normale (référence)")
    ax.fill_between(x, dist, where=(x < -2), alpha=0.5, color="#e74c3c")
    ax.fill_between(x, dist, where=(x >  2), alpha=0.5, color="#e74c3c")
    ax.axvline(-2, color="gray", linestyle=":", linewidth=1)
    ax.axvline( 2, color="gray", linestyle=":", linewidth=1)
    ax.text(-2.1, dist.max() * 0.6, "−2σ", ha="right", fontsize=8, color="gray")
    ax.text( 2.1, dist.max() * 0.6, "+2σ", ha="left",  fontsize=8, color="gray")
    ax.set_title(title, fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("Rendements")
    ax.set_ylabel("Densité")
    ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center", fontsize=9,
            style="italic", color="gray")
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 6)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=2)
plt.savefig("reports/kurtosis_illustration.png", dpi=150, bbox_inches="tight")
plt.show()

print("Excès de kurtosis des distributions :")
for name, rv in [("Platykurtique (Uniforme)", stats.uniform(-3, 6)),
                 ("Normale",                  stats.norm(0, 1)),
                 ("Leptokurtique (t, df=4)",  stats.t(df=4))]:
    sample = rv.rvs(100_000)
    print(f"  {name:30s} → excess kurt = {stats.kurtosis(sample):+.3f}")
