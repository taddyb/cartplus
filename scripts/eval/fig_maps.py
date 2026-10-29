"""Spatial comparison maps for Section 5 (Figures fig:map_gb, fig:map_us, fig:map_aus; held-out 2011-2019, single models = per-catchment
mean of the metric over the three seeds): (a) NSE of CaRT+, (b) CaRT+ minus the stateful LSTM, (c) CaRT+ minus the EA-LSTM, (d) CaRT+ minus
CaRT; ungauged catchments outlined in black. Geometry from prep_map_geometry.py (<geom>/geom_{gb,us,aus}.npz, default <out>/data/maps).
Writes <out>/figs/map_gb.pdf, map_us.pdf and map_aus.pdf (+ .png).
usage: python scripts/eval/fig_maps.py [gb|us|aus|all] [--geom DIR] [--runs DIR] [--out DIR]   (all = gb and us)
"""
import os, sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalcommon as EC
_ap = EC.parser(__doc__); _ap.add_argument("what", nargs="?", default="all", choices=["gb", "us", "aus", "all"]); _ap.add_argument("--geom", default="")
A_ = _ap.parse_args()
import fig_unified as FU
PP = FU.PP
GEOM = A_.geom or PP + "data/maps"
TAGS = {ds: {m: FU.KEY[m] for m in ("CaRT", "CaRT+", "LSTM", "EA-LSTM")} for ds in ("gb", "us", "aus")}     # the models drawn (evalcommon keys)
SEQ = LinearSegmentedColormap.from_list("seq", ["#e9f1fb", "#9cc3ee", "#2a78d6", "#123e7a"])                  # one hue, light -> dark
DIV = LinearSegmentedColormap.from_list("div", ["#a24a08", "#f0a35e", "#e4e3df", "#7fb0ea", "#1c5aa6"])      # LSTM/CaRT better <- grey -> CaRT+ better
INK, MUTED, LAND = "#1f1f1e", "#6b6a66", "#f4f3ef"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
AUS_W, AUS_H = 8.6, 9.4                                                                                  # AUS: near-square continent -> 2x2 of square panels
AUS_ADJ = dict(left=0.01, right=0.99, top=0.935, bottom=0.10, hspace=0.17, wspace=0.02)
AUS_CB = ([0.05, 0.05, 0.36, 0.016], [0.49, 0.05, 0.46, 0.016])


def seedmean_nse(ds, m, D):
    S = [np.load(EC.sim_path(A_, ds, FU.KEY[m], k)).astype(float) for k in range(3)]
    return np.mean([FU.metrics(s, D["q"], D["per"])[:, 0] for s in S], 0)


def rings(g, pre):
    xy, off, key = g[pre + "_xy"], g[pre + "_off"], g[pre + "_key" if pre == "land" else pre + "_id"]
    return [(key[i], xy[off[i]:off[i + 1]]) for i in range(len(off) - 1)]


def area(r): x, y = r[:, 0], r[:, 1]; return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def draw(ax, cats, land, val, ids, ug, cmap, norm, scale, extent=None):
    ax.add_collection(PolyCollection([r / scale for _, r in land], facecolor=LAND, edgecolor="#b9b7b0", linewidth=0.4, zorder=0, rasterized=True))
    idx = {k: i for i, k in enumerate(ids)}
    rs = sorted([(area(r), k, r) for k, r in cats if k in idx], key=lambda t: -t[0])                              # large first: nested small on top
    col = []
    for _, k, r in rs:
        v = val[idx[k]]; col.append(cmap(norm(v)) if np.isfinite(v) else (1, 1, 1, 1))
    ax.add_collection(PolyCollection([r / scale for _, _, r in rs], facecolors=col, edgecolor="white", linewidth=0.15, zorder=1, rasterized=True))
    ugr = [r / scale for _, k, r in rs if ug[idx[k]]]
    ax.add_collection(PolyCollection(ugr, facecolor="none", edgecolor=INK, linewidth=0.5, zorder=2, rasterized=True))
    xy = np.concatenate([r for _, r in (cats if extent is None else extent)]) / scale                     # extent: rings that set the view (default: catchments)
    ax.set_xlim(xy[:, 0].min() - 0.02 * np.ptp(xy[:, 0]), xy[:, 0].max() + 0.02 * np.ptp(xy[:, 0]))
    ax.set_ylim(xy[:, 1].min() - 0.02 * np.ptp(xy[:, 1]), xy[:, 1].max() + 0.02 * np.ptp(xy[:, 1]))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]


def fig(ds):
    D = FU.load_ds(ds); g = np.load(GEOM + f"/geom_{ds}.npz", allow_pickle=False)
    ids = [int(i) for i in D["ids"]]; mask = D["mask"]; ug = D["ug"]                                     # GB: the 64 held-out catchments (not every catchment outside the training mask)
    N = {m: seedmean_nse(ds, m, D) for m in TAGS[ds]}
    cats, land = rings(g, "cat"), rings(g, "land")
    if ds == "us": land = [(k, r) for k, r in land]
    lo = 0.6 if ds == "gb" else 0.3
    if ds == "aus": print("aus: catchments with a polygon", len(set(ids) & {int(k) for k, _ in cats}), "of", len(ids), "; ungauged outlined", int(sum(ug[ids.index(int(k))] for k, _ in cats if int(k) in ids)))
    pan = [("(a) NSE of CaRT$^+$", N["CaRT+"], SEQ, Normalize(lo, 1.0), None),
           ("(b) CaRT$^+$ $-$ stateful LSTM", N["CaRT+"] - N["LSTM"], DIV, Normalize(-0.1, 0.1), "LSTM"),
           ("(c) CaRT$^+$ $-$ EA-LSTM", N["CaRT+"] - N["EA-LSTM"], DIV, Normalize(-0.1, 0.1), "EA-LSTM"),
           ("(d) CaRT$^+$ $-$ CaRT", N["CaRT+"] - N["CaRT"], DIV, Normalize(-0.1, 0.1), "CaRT")]
    if ds == "gb": f, axs = plt.subplots(1, 4, figsize=(11, 5.4)); scale = 1e3
    elif ds == "aus": f, axs = plt.subplots(2, 2, figsize=(AUS_W, AUS_H)); axs = axs.ravel(); scale = 1e3
    else: f, axs = plt.subplots(2, 2, figsize=(11, 6.6)); axs = axs.ravel(); scale = 1e3
    for ax, (ttl, v, cm, nm, ref) in zip(axs, pan):
        draw(ax, cats, land, v, ids, ug, cm, nm, scale, extent=land if ds == "aus" else None)
        if ref is None: sub = f"median {np.nanmedian(v[mask]):.3f} gauged\n{np.nanmedian(v[ug]):.3f} ungauged"
        else:
            d = v[mask & np.isfinite(v)]; du = v[ug & np.isfinite(v)]
            sub = f"median {np.median(d):+.3f}\nCaRT$^+$ better in {np.mean(d > 0) * 100:.0f}% gauged,{chr(10) if ds == 'gb' else ' '}{np.mean(du > 0) * 100:.0f}% ungauged"
        ax.set_title(ttl + "\n" + sub, loc="left", fontsize=8.5, color=INK)
    if ds == "gb": f.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.17, wspace=0.06)
    elif ds == "aus": f.subplots_adjust(**AUS_ADJ)
    else: f.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.13, hspace=0.22, wspace=0.03)
    c1 = f.add_axes([0.04, 0.08, 0.3, 0.025] if ds != "aus" else AUS_CB[0]); c2 = f.add_axes([0.45, 0.08, 0.45, 0.025] if ds != "aus" else AUS_CB[1])
    f.colorbar(plt.cm.ScalarMappable(Normalize(lo, 1.0), SEQ), cax=c1, orientation="horizontal", extend="min").set_label("NSE (single model, 2011–2019)", color=INK)
    cb = f.colorbar(plt.cm.ScalarMappable(Normalize(-0.1, 0.1), DIV), cax=c2, orientation="horizontal", extend="both")
    cb.set_label("$\\Delta$NSE (> 0: CaRT$^+$ better)", color=INK)
    if ds == "aus": cb.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])                                            # narrower bar than the US one: fewer ticks
    f.savefig(PP + f"figs/map_{ds}.pdf", dpi=300); f.savefig(PP + f"figs/map_{ds}.png", dpi=150); plt.close(f)
    print(ds, {k: round(float(np.nanmedian(v[mask])), 3) for k, v in N.items()})


if __name__ == "__main__":
    what = A_.what
    for ds in (("gb", "us") if what == "all" else (what,)): fig(ds)
