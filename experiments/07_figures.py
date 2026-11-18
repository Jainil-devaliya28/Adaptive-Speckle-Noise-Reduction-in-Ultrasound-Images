"""Step 7: every figure used in the report and slides (results/figures/*.png)."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from skimage import measure  # noqa: E402

from common import FIGS, RESULTS, F  # noqa: E402
from speckle import simulate as S  # noqa: E402

# palette: one highlight for the proposed method, recessive grey for the rest;
# fixed categorical order when a few named methods are compared
PROP = "AS-NLM (proposed)"
HL, GREY, INK, MUTED = "#2a78d6", "#b9bec4", "#1b1f23", "#5b6268"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
FOCUS = [PROP, "Perona-Malik", "TV", "OBNLM", "SRAD", "Median"]
SHORT = {PROP: "AS-NLM", "Perona-Malik": "P-M", "Bilateral": "Bilat."}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e8ea", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=170)
    plt.close(fig)
    print("wrote", name)


def short(n):
    return SHORT.get(n, n)


# ------------------------------------------------------------- 1. simulator
def fig_simulator():
    E = S.phantom("cyst", seed=1)
    b, ref, env = S.simulate_bmode(E, seed=1)
    psf = S.make_psf(dz_mm=0.05, dx_mm=0.05)
    fig, ax = plt.subplots(1, 5, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 0.55, 1, 1, 1.2]})
    ax[0].imshow(10 * np.log10(np.maximum(E, 1e-3)), cmap="gray", vmin=-15, vmax=10)
    ax[0].set_title("(a) Echogenicity map (dB)")
    ax[1].imshow(np.abs(psf), cmap="magma", extent=[-psf.shape[1] * .025, psf.shape[1] * .025, psf.shape[0] * .025, -psf.shape[0] * .025])
    ax[1].set_title("(b) |PSF| (mm)")
    ax[1].set_aspect("auto")
    ax[2].imshow(b, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("(c) Simulated B-mode")
    ax[3].imshow(ref, cmap="gray", vmin=0, vmax=1)
    ax[3].set_title("(d) Speckle-free reference")
    for a in ax[[0, 2, 3]]:
        a.axis("off")
    ax[1].grid(False)
    # envelope statistics in background tissue vs Rayleigh
    region = env[330:420, 20:110].ravel()  # homogeneous background in the cyst phantom
    region = region / np.sqrt(np.mean(region ** 2) / 2)
    x = np.linspace(0, region.max(), 200)
    ax[4].hist(region, bins=60, density=True, color=GREY, edgecolor="white", linewidth=0.5)
    ax[4].plot(x, stats.rayleigh.pdf(x), color=HL, lw=2, label="Rayleigh pdf")
    snr = region.mean() / region.std()
    ax[4].set_title(f"(e) Envelope histogram, SNR = {snr:.2f}")
    ax[4].text(0.97, 0.75, "theory: 1.91", transform=ax[4].transAxes, ha="right", color=MUTED)
    ax[4].legend(loc="upper right")
    ax[4].set_xlabel("normalised amplitude")
    save(fig, "fig1_simulator.png")


# ------------------------------------------------------------- 2. gallery
def fig_sim_gallery(kind="cyst"):
    d = np.load(os.path.join(RESULTS, f"sim_example_{kind}.npz"))
    order = ["Reference", "B-mode (input)", "Median", "Lee", "Kuan", "Frost", "Perona-Malik",
             "SRAD", "DPAD", "TV", "Bilateral", "OBNLM", PROP]
    fig, axs = plt.subplots(2, 7, figsize=(17, 6.2))
    for a in axs.ravel():
        a.axis("off")
    for a, k in zip(axs.ravel(), order):
        a.imshow(d[k], cmap="gray", vmin=0, vmax=1)
        a.set_title(short(k), fontsize=10, color=HL if k == PROP else INK, fontweight="bold" if k == PROP else "normal")
    save(fig, f"fig2_gallery_{kind}.png")


def fig_zoom():
    d = np.load(os.path.join(RESULTS, "sim_example_cyst.npz"))
    sl = (slice(90, 330), slice(40, 340))
    keys = ["Reference", "B-mode (input)", "Perona-Malik", "TV", "OBNLM", "SRAD", PROP]
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(2, 7, height_ratios=[0.8, 1], hspace=0.25)
    for i, k in enumerate(keys):
        a = fig.add_subplot(gs[0, i])
        a.imshow(d[k][sl], cmap="gray", vmin=0, vmax=1)
        a.axhline(154 - 90, color="#eda100", lw=0.8, ls="--")
        a.set_title(short(k), color=HL if k == PROP else INK)
        a.axis("off")
    a = fig.add_subplot(gs[1, :])
    row = 154
    x = np.arange(40, 340)
    a.plot(x, d["B-mode (input)"][row, 40:340], color=GREY, lw=1, label="Input")
    a.plot(x, d["Reference"][row, 40:340], color=INK, lw=2, ls="--", label="Reference")
    for k, c in zip(["Perona-Malik", "OBNLM", PROP], [CAT[1], CAT[2], CAT[0]]):
        a.plot(x, d[k][row, 40:340], color=c, lw=2, label=short(k))
    a.set_xlabel("lateral pixel (row through both anechoic cysts)")
    a.set_ylabel("display intensity")
    a.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig3_zoom_profile.png")


# ------------------------------------------------------------- 3. bars
def fig_sim_bars():
    df = pd.read_csv(os.path.join(RESULTS, "sim_per_image.csv"))
    methods = [m for m in df["method"].unique()]
    mets = [("PSNR", "PSNR (dB)"), ("SSIM", "SSIM"), ("FOM", "Pratt FOM"), ("EPI", "EPI")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
    for a, (m, lab) in zip(axs, mets):
        g = df.groupby("method", sort=False)[m]
        mu, sd = g.mean().reindex(methods), g.std().reindex(methods)
        order = mu.sort_values().index
        cols = [HL if k == PROP else (GREY if k != "Noisy" else "#e0e2e4") for k in order]
        a.barh([short(k) for k in order], mu[order], xerr=sd[order], color=cols, height=0.7,
               error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
        a.set_title(lab)
        a.grid(axis="y", visible=False)
        for i, k in enumerate(order):
            if k == PROP:
                a.text(mu[k] * 0.03 if m != "PSNR" else 0.6, i, f"{mu[k]:.3g}", va="center", color="white",
                       fontweight="bold", fontsize=9)
    fig.suptitle("Simulation benchmark, 12 held-out phantoms (mean ± s.d.)", y=1.02)
    save(fig, "fig4_sim_metrics.png")


def fig_cnr():
    df = pd.read_csv(os.path.join(RESULTS, "sim_per_image.csv"))
    df = df[df["phantom"] == "cyst"]
    tg = ["anechoic", "hyperechoic", "hypoechoic", "subtle -3dB"]
    methods = list(df["method"].unique())
    fig, ax = plt.subplots(figsize=(14, 4.2))
    w = 0.2
    x = np.arange(len(methods))
    for j, t in enumerate(tg):
        mu = df.groupby("method", sort=False)[f"gCNR {t}"].mean().reindex(methods)
        ax.bar(x + (j - 1.5) * w, mu, width=w * 0.92, color=CAT[j], label=t)
    ax.set_xticks(x, [short(m) for m in methods], rotation=0)
    ax.set_ylabel("gCNR (1 = perfectly separable)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_gcnr_targets.png")


def fig_tradeoff():
    t = pd.read_csv(os.path.join(RESULTS, "tuning_log.csv"))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    others = t[~t["method"].isin(FOCUS)]
    ax.scatter(others["SSIM"], others["FOM"], s=14, color=GREY, label="other filters", zorder=1)
    for m, c in zip(FOCUS, CAT):
        g = t[t["method"] == m]
        ax.scatter(g["SSIM"], g["FOM"], s=22 if m != PROP else 34, color=c, label=short(m),
                   edgecolor="white", linewidth=0.6, zorder=3 if m == PROP else 2)
    ax.set_xlabel("SSIM (noise suppression and structure)")
    ax.set_ylabel("Pratt FOM (edge fidelity)")
    ax.set_title("Every tuning configuration: the speckle / edge trade-off")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig6_tradeoff.png")


# ------------------------------------------------------------- 4. robustness
def fig_robustness():
    df = pd.read_csv(os.path.join(RESULTS, "robustness.csv"))
    factors = list(df["factor"].unique())
    fig, axs = plt.subplots(1, len(factors), figsize=(16, 4.4), sharey=True)
    tuned = {"f0 (MHz)": 5.0, "Dynamic range (dB)": 50, "SNR (dB)": 40}
    for a, f in zip(axs, factors):
        g = df[df["factor"] == f].groupby(["level", "method"])["SSIM"].mean().unstack()
        for m, c in zip(FOCUS, CAT):
            a.plot(g.index, g[m], marker="o", ms=5, lw=2.4 if m == PROP else 1.6, color=c, label=short(m))
        a.axvline(tuned[f], color=MUTED, lw=0.8, ls="--")
        a.text(tuned[f], 0.565, " tuned here", color=MUTED, fontsize=8)
        a.set_xlabel(f)
    for a, t in zip(axs, ["Transmit frequency", "Display dynamic range", "Electronic SNR"]):
        a.set_title(t)
    axs[0].set_ylabel("SSIM (parameters fixed at tuning point)")
    axs[0].set_ylim(0.55, 0.95)
    axs[1].legend(loc="lower right", fontsize=8.5, ncol=2)
    save(fig, "fig7_robustness.png")


def fig_ablation():
    df = pd.read_csv(os.path.join(RESULTS, "ablation.csv"))
    g = df.groupby("variant", sort=False)[["SSIM", "FOM", "PSNR"]].mean()
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
    labels = list(g.index)
    for a, m in zip(axs, ["SSIM", "FOM", "PSNR"]):
        cols = [HL if l.startswith("G.") else GREY for l in labels]
        a.barh(labels[::-1], g[m].values[::-1], color=cols[::-1], height=0.65)
        lo = g[m][g[m] > 0].min()
        a.set_xlim(max(0, lo - (g[m].max() - lo) * 0.6), g[m].max() + (g[m].max() - lo) * 0.25)
        for i, v in enumerate(g[m].values[::-1]):
            a.text(v, i, f" {v:.3f}" if m != "PSNR" else f" {v:.2f}", va="center", fontsize=8.5, color=INK)
        a.set_title(m)
        a.grid(axis="y", visible=False)
    save(fig, "fig8_ablation.png")


def fig_maps():
    b, ref, _ = S.simulate_bmode(S.phantom("cyst", seed=1), seed=1)
    tp = json.load(open(os.path.join(RESULTS, "tuned_params.json")))[PROP]
    out, mp = F.asnlm(b, return_maps=True, **tp)
    fig, axs = plt.subplots(1, 6, figsize=(18, 4.4))
    items = [(b, "Input", "gray", (0, 1)), (np.clip(mp["r"], 0, 5), "Speckle index r(x)", "magma", (0, 5)),
             (mp["w1"], "w1: raw-image test", "cividis", (0, 1)), (mp["w2"], "w2: re-test on NLM", "cividis", (0, 1)),
             (mp["beta"], "β(x) NLM strength", "viridis", None), (out, "AS-NLM output", "gray", (0, 1))]
    for a, (im, t, cm, lim) in zip(axs, items):
        h = a.imshow(im, cmap=cm, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
        a.set_title(t)
        a.axis("off")
        if cm != "gray":
            fig.colorbar(h, ax=a, fraction=0.04, pad=0.02)
    save(fig, "fig9_asnlm_maps.png")


# ------------------------------------------------------------- 5. BUSI
def fig_busi_gallery():
    files = sorted(glob.glob(os.path.join(RESULTS, "busi_example_*.npz")))[:4]
    if not files:
        raise FileNotFoundError("no BUSI examples yet")
    keys = ["Input", "Lee", "Perona-Malik", "SRAD", "OBNLM", PROP]
    fig, axs = plt.subplots(len(files), len(keys), figsize=(17, 2.25 * len(files)), gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    axs = np.atleast_2d(axs)
    for r, f in enumerate(files):
        d = np.load(f)
        cont = measure.find_contours(d["mask"].astype(float), 0.5)
        for c, k in enumerate(keys):
            a = axs[r, c]
            a.imshow(d[k], cmap="gray", vmin=0, vmax=1)
            for ct in cont:
                a.plot(ct[:, 1], ct[:, 0], color="#eda100", lw=0.8, alpha=0.8)
            a.axis("off")
            if r == 0:
                a.set_title(short(k), color=HL if k == PROP else INK)
        axs[r, 0].text(-0.04, 0.5, os.path.basename(f)[13:-4].replace("_", " "), transform=axs[r, 0].transAxes,
                       rotation=90, va="center", ha="right", color=MUTED)
    save(fig, "fig10_busi_gallery.png")


def fig_busi_box():
    df = pd.read_csv(os.path.join(RESULTS, "busi_per_image.csv"))
    methods = list(df["method"].unique())
    mets = [("gCNR", "gCNR lesion vs tissue"), ("BCR", "Boundary contrast ratio"), ("Dice", "Segmentation Dice"), ("SNR_bg", "Background SNR")]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    for a, (m, lab) in zip(axs, mets):
        data = [df[df["method"] == k][m].values for k in methods]
        bp = a.boxplot(data, vert=False, patch_artist=True, widths=0.6, showfliers=False,
                       medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for p, k in zip(bp["boxes"], methods):
            p.set_facecolor(HL if k == PROP else GREY)
            p.set_edgecolor("white")
        a.set_yticks(range(1, len(methods) + 1), [short(k) for k in methods] if a is axs[0] else [""] * len(methods))
        a.invert_yaxis()
        a.set_title(lab)
        a.grid(axis="y", visible=False)
    fig.suptitle(f"BUSI clinical images (n = {df['image'].nunique()}), parameters tuned only on simulation", y=1.02)
    save(fig, "fig11_busi_metrics.png")


def fig_runtime():
    df = pd.read_csv(os.path.join(RESULTS, "sim_per_image.csv"))
    t = df[df["method"] != "Noisy"].groupby("method")["time_s"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh([short(k) for k in t.index], t.values, color=[HL if k == PROP else GREY for k in t.index], height=0.65)
    ax.set_xscale("log")
    ax.set_xlabel("seconds per 512 × 384 image (Python/NumPy, 1 core, log scale)")
    for i, v in enumerate(t.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, "fig12_runtime.png")


def fig_method_noise():
    d = np.load(os.path.join(RESULTS, "sim_example_organ.npz"))
    keys = ["Median", "Perona-Malik", "TV", "OBNLM", "SRAD", PROP]
    fig, axs = plt.subplots(1, len(keys), figsize=(17, 4.2))
    for a, k in zip(axs, keys):
        a.imshow(d["B-mode (input)"] - d[k], cmap="gray", vmin=-0.4, vmax=0.4)
        a.set_title(short(k), color=HL if k == PROP else INK)
        a.axis("off")
    fig.suptitle("Method noise (input − output): ideally pure speckle with no anatomy visible", y=1.0)
    save(fig, "fig13_method_noise.png")


if __name__ == "__main__":
    fig_simulator()
    for k in ("cyst", "organ", "edges"):
        fig_sim_gallery(k)
    fig_zoom()
    fig_sim_bars()
    fig_cnr()
    fig_tradeoff()
    fig_maps()
    fig_runtime()
    fig_method_noise()
    for fn in (fig_robustness, fig_ablation, fig_busi_gallery, fig_busi_box):
        try:
            fn()
        except FileNotFoundError as e:
            print("skipped", fn.__name__, "-", e)
