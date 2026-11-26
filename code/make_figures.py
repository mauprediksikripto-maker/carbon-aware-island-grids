"""Final figures for the JOCS manuscript, drawn from the v10_p2b result tables (and the hourly traces for Fig. 1).
Usage: python make_figures.py <results_dir> <traces.pkl> <out_dir>"""
import sys, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

R, TR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MM = 1 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.spines.top": False,
                     "axes.spines.right": False})
# Okabe-Ito
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7", sky="#56B4E9",
         yellow="#F0E442", black="#000000", grey="#999999")
NAME = {"Jawa_Bali": "Jawa-Bali", "Sumatera": "Sumatera", "Sulawesi": "South Sulawesi",
        "DE_real": "Germany 2024 (real)", "GB_real": "Great Britain 2024 (real)"}
t = lambda f: pd.read_csv(f"{R}/{f}.csv")


def save(fig, n):
    fig.savefig(f"{OUT}/Figure_{n}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{OUT}/Figure_{n}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def panel(ax, s, y=1.06):
    ax.text(-0.13, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


# ---------- Fig. 1: hourly ACI and MEF, one sample week ----------
tr = pickle.load(open(TR, "rb"))
w = slice(24 * 280, 24 * 287)   # 7-13 October 2024, Monday-Sunday
h = np.arange(168)
fig, (a, b) = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM), constrained_layout=True)
for g, c in [("Jawa_Bali", C["red"]), ("Sumatera", C["blue"]), ("Sulawesi", C["green"]),
             ("DE_real", C["orange"]), ("GB_real", C["purple"])]:
    a.plot(h, tr[g].avg_ci.to_numpy()[w], color=c, label=NAME[g], ls="-" if "real" not in g else "--")
a.set(xlabel="Hour of the week, 7–13 October (Mon–Sun)", ylabel="Average carbon intensity (gCO$_2$/kWh)", xlim=(0, 167), ylim=(0, None))
a.set_xticks(range(0, 169, 24)); a.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), handlelength=1.8)
panel(a, "(a)", 1.18)
for g, c in [("Jawa_Bali", C["red"]), ("Sumatera", C["blue"])]:
    v = tr[g]; hod = np.arange(len(v)) % 24
    b.plot(range(24), v.groupby(hod).avg_ci.mean(), color=c, label=f"{NAME[g]}, ACI")
    b.plot(range(24), v.groupby(hod).mef.mean(), color=c, ls="--", marker="o", ms=2.5, label=f"{NAME[g]}, mean MEF")
b.set(xlabel="Hour of day", ylabel="Annual mean by hour (gCO$_2$/kWh)", xlim=(0, 23), ylim=(0, 1050))
b.set_xticks([0, 6, 12, 18, 23]); b.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), handlelength=1.8)
panel(b, "(b)", 1.18)
save(fig, 1)

# ---------- Fig. 2: regime map ----------
rm = t("table10_regime_map")
fig, axs = plt.subplots(2, 2, figsize=(190 * MM, 120 * MM), constrained_layout=True)
for i, g in enumerate(["Sumatera", "Jawa_Bali"]):
    d = rm[rm.grid == g]
    lim = np.ceil(d.saving_avgCI.abs().max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    scales = sorted(d.solar_scale.unique()); kap = sorted(d.coal_min.unique())
    for j, gf in enumerate([0.0, 1.0]):
        ax = axs[i, j]; e = d[d.gas_first == gf]
        Z = e.pivot(index="coal_min", columns="solar_scale", values="saving_avgCI").loc[kap, scales].to_numpy()
        share = e[e.coal_min == kap[0]].set_index("solar_scale").loc[scales, "solar_share_pct"].to_numpy()
        im = ax.pcolormesh(np.arange(len(scales) + 1), np.arange(len(kap) + 1), Z, cmap="RdBu", norm=norm,
                           shading="flat")
        ax.contour(np.arange(len(scales)) + 0.5, np.arange(len(kap)) + 0.5, Z, levels=[0], colors="k",
                   linewidths=0.9)
        ax.set_xticks(np.arange(len(scales)) + 0.5, [f"{s:.1f}" for s in share])
        ax.set_yticks(np.arange(len(kap)) + 0.5, [f"{k:.1f}" for k in kap])
        ax.set(xlabel="Solar energy share (%)", ylabel="Coal minimum stable fraction κ")
        ax.set_title(f"{NAME[g]}, {'gas-first' if gf else 'coal-first'} merit order", loc="left")
        panel(ax, "(" + "abcd"[2 * i + j] + ")")
    cb = fig.colorbar(im, ax=axs[i, :], shrink=0.9, pad=0.01)
    cb.set_label("ACI scheduler saving (%)")
save(fig, 2)

# ---------- Fig. 3: harm valley over ten realizations ----------
rc = t("table17b_cells")
fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 65 * MM), constrained_layout=True)
cols = {0.2: C["blue"], 0.4: C["orange"], 0.6: C["green"], 0.8: C["red"]}
mk = {0.2: "o", 0.4: "s", 0.6: "^", 0.8: "D"}
for ax, g, p in zip(axs, ["Sumatera", "Jawa_Bali"], "ab"):
    d = rc[(rc.grid == g) & (rc.gas_first == 0)]
    for k, c in cols.items():
        e = d[np.isclose(d.coal_min, k)].sort_values("solar_share_pct")
        ax.errorbar(e.solar_share_pct, e.avgCI_mean, yerr=e.avgCI_sd, color=c, marker=mk[k], ms=3, capsize=1.5,
                    lw=1, elinewidth=0.7, label=f"κ = {k}")
    ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log")
    ax.set(xlabel="Solar energy share (%)", ylabel="ACI scheduler saving (%)")
    ax.set_title(f"{NAME[g]}, coal-first merit order", loc="left"); panel(ax, f"({p})")
axs[0].legend(frameon=False, loc="lower left")
save(fig, 3)

# ---------- Fig. 4: marginal unit class by hour ----------
me = t("table19_mechanism")
cls = [("pct_curtailed_RE", "Curtailed renewables", C["sky"]), ("pct_coal_at_min", "Coal at minimum", C["grey"]),
       ("pct_coal_above_min", "Coal above minimum", C["black"]), ("pct_ccgt", "CCGT", C["orange"]),
       ("pct_peaker", "Peaker", C["red"])]
fig, axs = plt.subplots(2, 3, figsize=(190 * MM, 105 * MM), constrained_layout=True)
k = 0
for i, g in enumerate(["Sumatera", "Jawa_Bali"]):
    d = me[me.grid == g]
    for j, sc in enumerate(sorted(d.solar_scale.unique())):
        ax = axs[i, j]; e = d[d.solar_scale == sc].sort_values("hour")
        ax.stackplot(e.hour, *[e[c] for c, _, _ in cls], colors=[c for _, _, c in cls], alpha=0.55,
                     labels=[l for _, l, _ in cls], step=None)
        ax.set(xlim=(0, 23), ylim=(0, 100), xlabel="Hour of day", ylabel="Hours with this marginal class (%)")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        ax2.plot(e.hour, e.aci_mean, color=C["blue"], lw=1.2, label="Mean ACI")
        ax2.set_ylabel("Mean ACI (gCO$_2$/kWh)", color=C["blue"]); ax2.tick_params(axis="y", colors=C["blue"])
        ax.set_title(f"{NAME[g]}, solar share {e.solar_share_pct.iloc[0]:.1f}%", loc="left")
        panel(ax, f"({'abcdef'[k]})"); k += 1
hs, ls = axs[0, 0].get_legend_handles_labels()
from matplotlib.lines import Line2D
fig.legend(hs + [Line2D([], [], color=C["blue"], lw=1.2)], ls + ["Mean ACI (right axis)"], loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0))
save(fig, 4)

# ---------- Fig. 5: Sobol indices, consequential ACI saving ----------
so = t("table8_sobol_consequential")
lab = {"coal_min": "Coal minimum κ", "coal_scale": "Coal capacity", "ccgt_scale": "CCGT capacity",
       "solar_scale": "Solar capacity", "gas_first": "Merit order", "forecast_sigma": "Forecast error"}
order = list(lab)
fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM), constrained_layout=True)
for ax, g, p in zip(axs, ["Sumatera", "Jawa_Bali"], "ab"):
    d = so[(so.grid == g) & (so.output == "c_avgCI")].set_index("input").loc[order]
    x = np.arange(len(order))
    ax.bar(x - 0.2, d.S1.clip(lower=0), 0.4, yerr=d.S1_conf, color=C["sky"], label="First-order $S_1$",
           error_kw=dict(lw=0.7, capsize=1.5))
    ax.bar(x + 0.2, d.ST, 0.4, yerr=d.ST_conf, color=C["blue"], label="Total-order $S_T$",
           error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_xticks(x, [lab[o] for o in order], rotation=30, ha="right")
    ax.set(ylabel="Sobol index", ylim=(0, 1.05))
    ax.set_title(f"{NAME[g]}: ACI scheduler saving", loc="left"); panel(ax, f"({p})")
axs[0].legend(frameon=False, loc="upper left")
save(fig, 5)

# ---------- Fig. 6: calibration age (ten realizations) ----------
ca = t("table17c_summary")
sig = [("MEF_recalibrated", "MEF, recalibrated each step", C["blue"], "-", "o"),
       ("MEF_lag1", "MEF, one step behind", C["orange"], "--", "s"),
       ("MEF_stale", "MEF, calibrated once (stale)", C["green"], ":", "^"),
       ("avgCI", "ACI", C["red"], "-", "x")]
fig, axs = plt.subplots(2, 2, figsize=(190 * MM, 115 * MM), constrained_layout=True)
k = 0
for i, g in enumerate(["Sumatera", "Jawa_Bali"]):
    for j, path in enumerate(["solar_only", "solar_then_gas_first"]):
        ax = axs[i, j]; d = ca[(ca.grid == g) & (ca.path == path)].sort_values("step")
        for s, l, c, lsty, m in sig:
            ax.errorbar(d.solar_share_pct, d[f"saving_{s}_mean"], yerr=d[f"saving_{s}_sd"], color=c, ls=lsty,
                        marker=m, ms=3, capsize=1.5, elinewidth=0.7, label=l)
        if path != "solar_only":
            sw = d[d.step == 4].solar_share_pct.iloc[0]
            ax.axvline(sw, color=C["grey"], lw=0.8, ls="-."); ax.text(sw, ax.get_ylim()[1], " switch to gas-first",
                                                                       fontsize=6, va="top", color="0.35")
        ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log")
        ax.set(xlabel="Solar energy share (%)", ylabel="Consequential saving (%)")
        ax.set_title(f"{NAME[g]}: {'solar growth only' if path == 'solar_only' else 'solar growth, then gas-first order'}",
                     loc="left")
        panel(ax, f"({'abcd'[k]})"); k += 1
axs[0, 0].legend(frameon=False, loc="lower left")
save(fig, 6)

# ---------- Fig. 7: load shape ----------
ls_ = t("table18_load_shape")
cq = ls_[ls_.kind == "conseq"]
fig, ax = plt.subplots(figsize=(140 * MM, 80 * MM), constrained_layout=True)
s = cq[(cq.grid == "Sumatera") & cq.variant.str.startswith("analytic_bf")].sort_values("daily_min_to_max")
ax.plot(s.daily_min_to_max, s.saving_MEF, color=C["blue"], marker="s", ms=3.5, ls="--",
        label="Sumatera, MEF (analytical shape sweep)")
ax.plot(s.daily_min_to_max, s.saving_avgCI, color=C["red"], marker="o", ms=3.5,
        label="Sumatera, ACI (analytical shape sweep)")
j = cq[(cq.grid == "Jawa_Bali") & cq.variant.isin(["base", "p2b_flat_season", "coal_cap_low", "coal_cap_high"])]
ax.scatter(j.daily_min_to_max, j.saving_MEF, marker="D", facecolor="none", edgecolor=C["blue"], s=18, zorder=3,
           label="Jawa-Bali, MEF (empirical-shape variants)")
ax.scatter(j.daily_min_to_max, j.saving_avgCI, marker="*", color=C["red"], s=30, zorder=3,
           label="Jawa-Bali, ACI (empirical-shape variants)")
ax.axvspan(0.74, 0.83, color="0.9", zorder=0, label="Jawa-Bali observed 2024 (0.74–0.83)")
ax.axhline(0, color="k", lw=0.6)
ax.set(xlabel="Daily minimum / daily peak load", ylabel="Consequential saving at s = 2% (%)", xlim=(0.47, 0.85))
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
save(fig, 7)
print("done")
