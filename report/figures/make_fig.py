import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
e = np.arange(1, 41)

def noise(scale=0.012):
    return np.random.normal(0, scale, len(e))

# Underfitting: both curves plateau high, gap tiny
u_tr = 1.05 * np.exp(-e / 9.0) + 0.78 + noise()
u_va = 1.05 * np.exp(-e / 9.0) + 0.82 + noise()

# Good fit: both fall, val flattens just above train
g_tr = 1.55 * np.exp(-e / 7.5) + 0.20 + noise()
g_va = 1.55 * np.exp(-e / 7.0) + 0.30 + noise()

# Overfitting: train -> 0, val bottoms out then climbs
o_tr = 1.70 * np.exp(-e / 5.0) + 0.03 + noise(0.008)
o_va = 1.60 * np.exp(-e / 5.5) + 0.34 + 0.017 * np.clip(e - 12, 0, None) + noise()
best = int(np.argmin(o_va))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
titles = ["(a) Underfitting", "(b) Good fit", "(c) Overfitting"]
pairs = [(u_tr, u_va), (g_tr, g_va), (o_tr, o_va)]

for a, t, (tr, va) in zip(ax, titles, pairs):
    a.plot(e, tr, color="#1f4e79", lw=1.6, label="Training loss")
    a.plot(e, va, color="#c0392b", lw=1.6, ls="--", label="Validation loss")
    a.set_title(t, fontsize=10)
    a.set_xlabel("Epoch", fontsize=9)
    a.set_ylim(0, 2.0)
    a.grid(alpha=0.25, lw=0.5)
    a.tick_params(labelsize=8)

ax[0].set_ylabel("Cross-entropy loss", fontsize=9)
ax[2].axvline(e[best], color="grey", lw=1.0, ls=":")
ax[2].annotate("early-stopping\npoint", xy=(e[best], o_va[best]),
               xytext=(e[best] + 4, 1.15), fontsize=7.5, color="grey",
               arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
ax[0].legend(fontsize=8, loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig("loss_curves.pdf", bbox_inches="tight")
print("saved, early stop epoch =", e[best])
