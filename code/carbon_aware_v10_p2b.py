# %% [md]
# # Carbon-aware scheduling on coal-dominated island grids: merit-order dispatch model and experiments (v10)
# Code release for the article *"Model-based marginal emission signals for carbon-aware scheduling on coal-dominated
# island grids: a computational regime map along Indonesia's transition pathway"* (D. Irawan, Sudarmaji, I. Samsudin;
# Faculty of Computer Science, Universitas Muhammadiyah Metro, Indonesia). Submitted to the *Journal of Computational Science*.
#
# **What this notebook does**
# 1. Builds hourly carbon-intensity traces (average CI and marginal emission factor, MEF) for Indonesian island grids
#    (Jawa-Bali, Sumatera, South Sulawesi) and a synthetic high-renewable reference with a transparent merit-order
#    dispatch model (coal minimum stable level, take-or-pay gas block, curtailment), and loads real European traces
#    (Electricity Maps 2024, DE and GB) for comparison.
# 2. Simulates a year of delay-tolerant batch jobs under carbon-agnostic, temporal, spatial and spatio-temporal
#    policies and evaluates them by consequential accounting: the data-centre load is added to demand and the grid is
#    re-dispatched.
# 3. Runs the analyses reported in the article: Sobol global sensitivity analysis, regime map, calibration age,
#    ablation of the marginal signal, PPO transfer, emission-factor envelope, 10-seed replicates, load-shape robustness,
#    dispatch mechanism and slack sensitivity.
#
# **Data**
# - Indonesian grids are calibrated to official 2024 energy mixes (Statistik PLN 2024, Tables 23 and 26; South
#   Sulawesi: ESDM press release); emission factors are derived from official fuel-use data (section 1).
# - The Jawa-Bali demand uses the 2024 daily load curves, weekly peak envelope and holiday loads of PLN UIP2B Jamali
#   (Statistik Sistem Jawa Madura Bali 2024), digitized from the report's figures (section 1; also in `data/`).
# - European traces: Electricity Maps (ODbL), retrieved at run time from a pinned commit (section 1).
# - `DATA_VERIFIED = True`. If any grid or emission-factor value is changed, set it to False: every figure is then
#   watermarked "PRELIMINARY".
#
# **Reproducibility**
# - All numbers in the results are produced by this code with fixed seeds (`SEED = 42`).
# - This is the exact code of run `v10_p2b` (Google Colab CPU runtime, 25 Nov 2026). Only the explanatory text cells
#   were edited for this release; every code cell is unchanged.
# - Section numbers follow the order in which the stages were added during development; numbers 9 and 15 (provenance)
#   are not consecutive.
#
# **Run safety (Colab)**
# - Outputs are written straight to Google Drive (`MyDrive/jocs_pilot/<RUN_TAG>/`) stage by stage; outside Colab they go
#   to `./jocs_pilot/<RUN_TAG>/`.
# - If the runtime disconnects, run all cells again: finished stages are skipped, partial stages resume
#   (row-level for tables, checkpoint-level for PPO).
# - Hard time budget per session: `MAX_RUNTIME_MIN` (default 120 min). Stages stop cleanly before the limit.
# - A complete run takes about 2-3 h of Colab CPU time, possibly over more than one session.
# - When every stage is done, a ZIP of all results with SHA-256 checksums is written automatically.

# %%
# gymnasium + stable-baselines3 are installed automatically in the PPO cell when RUN_PPO = True
import json, hashlib, os, time, zlib, gc, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
DATA_VERIFIED = True       # v8: targets and emission factors from official sources (see section 1)
RUN_PPO = True             # v8: recomputed on the re-calibrated grids
RUN_LEGACY = True          # v8: sections 4-5 recomputed on the re-calibrated grids
PREV_RUN = None            # v8: no reuse (grids changed); set a folder name to reuse finished Sobol rows
PPO_TIMESTEPS = 60_000
PPO_CKPT_EVERY = 10_000    # PPO checkpoint interval (timesteps)
HOURS = 24 * 365
RUN_TAG = "v10_p2b"       # v10: Jawa-Bali demand re-based on UIP2B 2024 -> every stage is recomputed
ZIP_TAG = "v10_p2b"       # name of the ZIP written when every stage is finished
MAX_RUNTIME_MIN = 120      # hard budget for this session
SAFETY_MIN = 8             # stop new work this many minutes before the budget ends
START = time.time()

try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/jocs_pilot"; IN_COLAB = True
except ImportError:
    BASE = "jocs_pilot"; IN_COLAB = False
OUT = f"{BASE}/{RUN_TAG}"
os.makedirs(OUT, exist_ok=True)
print("Saving to:", OUT)

def time_left():
    return (MAX_RUNTIME_MIN - SAFETY_MIN) * 60 - (time.time() - START)

def budget_ok(label=""):
    ok = time_left() > 0
    if not ok:
        print(f"[time budget] stopping before {label}. Re-run all later to resume.")
    return ok

def done(stage):
    return os.path.exists(f"{OUT}/.done_{stage}")

def mark(stage, **info):
    json.dump(dict(info, finished=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{OUT}/.done_{stage}", "w"))

def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

def read_rows(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def watermark(ax):
    if not DATA_VERIFIED:
        ax.text(0.5, 0.5, "PRELIMINARY", transform=ax.transAxes, fontsize=34, color="grey",
                alpha=0.18, ha="center", va="center", rotation=25)

def save_fig(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}", dpi=300)
    fig.savefig(f"{OUT}/{name.rsplit('.', 1)[0]}.pdf")              # vector copy for the journal
    plt.show(); plt.close(fig); gc.collect()

# %% [md]
# ## 1. Grid model (merit-order dispatch)
# Each hour: must-run renewables (geothermal, hydro, solar, wind) → coal (with a minimum stable level) →
# CCGT → peakers (OCGT/diesel). **Average CI** = total emissions / total generation.
# **Marginal emission factor (MEF)** = emission factor of the marginal (partly loaded) dispatchable unit.
#
# **Emission factors (tCO2/MWh)** are derived as EF = specific fuel consumption (SFC) x net calorific value x CO2
# factor, from official data only:
# - SFC: Statistik PLN 2024 Tables 24 (fuel use) and 26 (production by fuel), PLN-owned Jawa-Bali units
#   (Indonesia Power Jawabali, Nusantara Power Jawabali, UIK Tanjung Jati B, UIK JAMALI): coal 0.5794 kg/kWh,
#   gas 8.135 MMSCF/GWh; DJK-ESDM Laporan Kinerja 2024, Table 31 (PLN fleet, Q4 2024): coal 0.6272 kg/kWh,
#   gas 0.0084 MMBTU/kWh, oil 0.2789 L/kWh. IPP fuel use is not published, so only PLN-owned units give an SFC.
# - NCV and CO2 factors: ESDM "Faktor Emisi BBM dan Batubara" (coal by quality class, natural gas 57.64 t/TJ,
#   gasoil CN48 73.28 t/TJ); energy contents of gas and gasoil from HEESI 2024 conversion factors
#   (1 MSCF = 0.1796 BOE, 1 kL gasoil = 6.4871 BOE, 1 MWh = 0.613 BOE). Gas: NCV = 0.9 x GCV.
# - The coal quality actually burned is not reported, so coal EF spans 0.91 (Jawa SFC, low-rank class) to 1.24
#   (national SFC, national average coal). The base value 0.95 is kept: it lies inside this envelope, near the end
#   that matches the independent official grid operating margin (Jamali 2019, section 7); IPP plants, which supply
#   half of Jawa-Bali coal output, are mostly large supercritical units, so the system value sits below the
#   PLN-owned bottom-up value. CCGT is updated from 0.40 (below the envelope) to 0.445 (Jawa PLN-owned gas SFC).
#   Peaker 0.70 lies between the fossil share of B35 bio-diesel (0.51) and pure gasoil (0.78).
# - Section 16 repeats the headline results for every corner of this envelope (`EF_SETS`).

# %%
EF = {"coal": 0.95, "ccgt": 0.445, "peaker": 0.70, "solar": 0.0, "wind": 0.0, "hydro": 0.0, "geo": 0.0}
EF_SETS = {   # tCO2/MWh; see derivation above (all values from official data except the v7 literature set)
    "base_v8":       dict(coal=0.95,  ccgt=0.445, peaker=0.70),
    "low":           dict(coal=0.913, ccgt=0.433, peaker=0.506),  # Jawa SFC x low-rank coal; IPCC gas; B35 fossil
    "PLN_own_avg":   dict(coal=1.144, ccgt=0.445, peaker=0.779),  # Jawa SFC x national average coal; pure gasoil
    "high":          dict(coal=1.238, ccgt=0.511, peaker=0.779),  # national SFC x average coal; gas without NCV cut
    "v7_literature": dict(coal=0.95,  ccgt=0.40,  peaker=0.70),
}

# Indonesian grids: CALIBRATED (v8, Nov 2026) so that simulated annual energy shares match official 2024 mixes.
# Capacities below are *effective dispatch* capacities found by calibration, NOT nameplate values.
# Sources:
#  - Jawa-Bali SYSTEM peak 2024 = 32,759 MW (9 Oct 2024 18:00; PLN UIP2B Jamali, Statistik Sistem Jawa Madura
#    Bali 2024). v10 correction: the 42,635.66 MW in Statistik PLN 2024 (Summary, Table 3) is the SUM of the peak
#    loads of the generating units (Indonesia Power, Nusantara Power, UIP2B, Tanjung Jati B, ...), not the
#    coincident system peak, and its "load factor" (63.86 %) is defined on that sum. Net production 2024 =
#    230,615 GWh (UIP2B) -> system load factor ~80 %. Demand shape: empirical P2B curves (see P2B_* above).
#    Mix 2024 (Statistik PLN 2024, Table 26 by fuel incl. IPP, Table 23 total incl. purchases; rows "Jawa" + "UID
#    Bali"): total 238,507 GWh; coal 178,471 (74.83%), biomass co-firing in coal units 1,240 (0.52%),
#    gas 40,021 (16.78%), oil 1,448 (0.61%), non-thermal renewables 17,327 (7.26%).
#    Model classes: coal = coal + co-firing (75.35), gas_oil = 17.39, ren = 7.26.
#  - Sumatera (interconnected system: Aceh, Sumut, Sumbar, Riau excl. Kepri, S2JB, Lampung, UIP3B/UIK Sumatera,
#    Indonesia Power and Nusantara Power Sumatera; Bangka-Belitung and Batam excluded), same tables:
#    total 54,234 GWh; coal 47.15 + co-firing 0.63, gas 23.62, oil 1.71, non-thermal renewables 26.89 (%).
#    Peak = 7,386 MW (Apr 2024; Anggara et al., Jurnal Rekayasa Elektro Sriwijaya 6(1), 2024) is used only as a
#    scale parameter; the official tables do not report a Sumatera system peak.
#  - Sulbagsel peak = 1,773.95 MW (ESDM, RAFI period); wind daya mampu 130 MW (PLTB Sidrap 70 + Tolo 60);
#    energy mix Jan-Mar 2024: coal 50.89, gas 11.02, hydro 29.40, wind 2.26, oil 6.43 (%) (ESDM press release
#    236.Pers/04/SJI/2024, 30 Apr 2024). Wet-season quarter: hydro share is likely above the annual average.
#    Cross-check, all PLN Sulawesi units 2024 (Statistik PLN 2024; north + south, cannot be split):
#    coal 49.9, gas+oil 17.6, renewables incl. biomass 32.6 (%), close to the Sulbagsel target.
# EU_like is a SYNTHETIC high-renewable reference; the real European traces are DE_real and GB_real (below).
CALIBRATION_TARGETS = {
    "Jawa_Bali": dict(coal=75.35, gas_oil=17.39, ren=7.26),
    "Sumatera":  dict(coal=47.78, gas_oil=25.33, ren=26.89),
    "Sulawesi":  dict(coal=50.9, gas_oil=17.45, ren=31.7),
}
GRIDS = {
    "EU_like":   dict(peak=60000, base_frac=0.70, coal=12000, coal_min=0.35, ccgt=25000, peaker=22000,
                      solar=55000, wind=60000, hydro_cf_mw=4000, geo=0, cloud=0.35, wind_cf=0.25,
                      tz_shift=0),   # SYNTHETIC
    "Jawa_Bali": dict(peak=31654, base_frac=0.72, p2b=True, coal=22250, coal_min=0.4, ccgt=12000, ccgt_must=4250,
                      peaker=12000, solar=900, wind=0, hydro_cf_mw=921, geo=836, cloud=0.45, wind_cf=0.0, tz_shift=0),
    # v10 calibration: max demand 32,759 MW. The UIP2B generation stacks (pp. 43-44) show coal following load (about
    # 11.5-15.7 GW on the 2024 minimum day, 20.5-23.4 GW on the peak day) on top of a near-constant gas block, so a
    # take-or-pay gas block (`ccgt_must`) is added and the effective coal capacity is kept within the observed range;
    # grid search (coal 20.5-23.5 GW, gas block 1-6 GW, renewables scale) -> 22,250 MW / 4,250 MW / 0.775,
    # mix error <= 0.09 pp. Section 18 repeats E1 for the ends of the feasible range (21,250/3,750 and 23,500/4,500).
    "Sumatera":  dict(peak=7001, base_frac=0.70, coal=2675, coal_min=0.3, ccgt=2000, peaker=2000,
                      solar=40, wind=0, hydro_cf_mw=874, geo=681, cloud=0.45, wind_cf=0.0, tz_shift=0),
    "Sulawesi":  dict(peak=1650, base_frac=0.75, coal=700, coal_min=0.3, ccgt=200, peaker=400,
                      solar=10, wind=130, hydro_cf_mw=402, geo=0, cloud=0.45, wind_cf=0.30, tz_shift=-1),
}
# note: "peak" is a shape parameter; the simulated annual maximum equals the sourced peak.

# Jawa-Bali empirical load shape (v10). Source: PT PLN (Persero) UIP2B Jamali, Statistik Sistem Jawa Madura Bali
# 2024 (2025): half-hourly curves of the 2024 peak weekday (9 Oct, 32,759 MW), Saturday (12 Oct, 30,450 MW) and
# Sunday (22 Sep, 28,699 MW), pp. 29-31, and the 2024 weekly peak curve, p. 35. Curves were DIGITIZED from the
# figures (300 dpi; digitized peaks within 0.3-0.9 % of the printed values). Values below = load at hh:30 / day max.
P2B_WEEKDAY  = np.array([0.8126, 0.8021, 0.7900, 0.7834, 0.8088, 0.8238, 0.8102, 0.8625, 0.9239, 0.9511, 0.9670, 0.9475, 0.9474, 0.9907, 0.9891, 0.9740, 0.9605, 1.0000, 0.9812, 0.9657, 0.9425, 0.9051, 0.8680, 0.8388])
P2B_SATURDAY = np.array([0.8826, 0.8678, 0.8482, 0.8411, 0.8637, 0.8570, 0.8290, 0.8564, 0.9054, 0.9360, 0.9489, 0.9385, 0.9311, 0.9504, 0.9440, 0.9325, 0.9338, 1.0000, 0.9861, 0.9636, 0.9437, 0.9056, 0.8693, 0.8433])
P2B_SUNDAY   = np.array([0.8629, 0.8443, 0.8253, 0.8162, 0.8322, 0.8295, 0.7962, 0.7972, 0.8256, 0.8506, 0.8653, 0.8682, 0.8713, 0.8840, 0.8854, 0.8907, 0.9069, 0.9934, 0.9922, 0.9828, 0.9627, 0.9284, 0.8850, 0.8580])
P2B_WEEKLY_PEAK_MW = np.array([29097, 29656, 30064, 30101, 29565, 29260, 29452, 30934, 30550, 30597, 30101, 30597, 30706, 30153, 27849, 29221, 30590, 31606, 31192, 31342, 31535, 31663, 31635, 31278, 30840, 30863, 30571, 30473, 30266, 30612, 30866, 30618, 30719, 31008, 31150, 31390, 31684, 31218, 31775, 32129, 32715, 32404, 32460, 32640, 31728, 32168, 32098, 31767, 30992, 30496, 30597, 30202, 29345])                    # weeks 1-53 of 2024
P2B_DAY_SCALE = dict(sat=30450 / 32759, sun=28699 / 32759)   # Saturday / Sunday peak relative to weekday peak
# Holidays (day index from 1 Jan 2024 = Monday): Idul Fitri 10-11 Apr (peak 20,498 MW, p. 32), cuti bersama
# 8, 9, 12, 15 Apr (treated as Sundays), Christmas 25 Dec (peak 25,418 MW, p. 33). Holidays use the Sunday shape.
P2B_HOLIDAYS = {98: "sun", 99: "sun", 100: 20498, 101: 20498, 102: "sun", 105: "sun", 359: 25418}

def p2b_shape(hours=None, flat=False):
    hours = hours or HOURS
    env = P2B_WEEKLY_PEAK_MW / P2B_WEEKLY_PEAK_MW.max()
    if flat: env = np.ones_like(env)                                  # no seasonal envelope, no holidays
    out = np.empty(hours)
    for day in range(hours // 24):
        e = env[min(day // 7, 52)]; dow = day % 7
        s, f = (P2B_WEEKDAY, 1.0) if dow < 5 else (P2B_SATURDAY, P2B_DAY_SCALE["sat"]) if dow == 5 else \
               (P2B_SUNDAY, P2B_DAY_SCALE["sun"])
        if day in P2B_HOLIDAYS and not flat:
            h = P2B_HOLIDAYS[day]
            s, f = P2B_SUNDAY, (P2B_DAY_SCALE["sun"] if h == "sun" else h / P2B_WEEKLY_PEAK_MW.max() / e)
        out[day * 24:(day + 1) * 24] = s * f * e
    return out

def demand_profile(p, rng, hours=HOURS):
    h = np.arange(hours)
    hod = (h + p["tz_shift"]) % 24
    daily = 0.5 * (1 + np.sin((hod - 13) / 24 * 2 * np.pi))          # evening-ish peak
    evening = np.exp(-0.5 * ((hod - 19) / 2.0) ** 2)
    shape = p["base_frac"] + (1 - p["base_frac"]) * (0.55 * daily + 0.45 * evening)
    weekly = np.where(((h // 24) % 7) >= 5, 0.94, 1.0)
    if p.get("p2b"):                                                  # v10: empirical Jawa-Bali shape (UIP2B 2024)
        shape, weekly = p2b_shape(hours, flat=bool(p.get("p2b_flat"))), 1.0
    noise = 1 + 0.02 * rng.standard_normal(hours)
    cal = np.ones(hours)                                              # optional calendar effects (v9, section 18)
    if p.get("growth"):                                               # demand growth within the year, mean 1
        cal = cal * (1 + p["growth"] * (h / hours - 0.5))
    for start_day, n_days, factor in p.get("holidays", []):           # e.g. Lebaran week(s)
        cal[start_day * 24:(start_day + n_days) * 24] *= factor
    return p["peak"] * shape * weekly * noise * cal

def solar_profile(p, rng, hours=HOURS):
    h = np.arange(hours)
    hod = (h + p["tz_shift"]) % 24
    clear = np.clip(np.sin((hod - 6) / 12 * np.pi), 0, None)          # 06–18 tropical day
    day_cloud = np.repeat(1 - p["cloud"] * rng.beta(2, 2, hours // 24 + 1), 24)[:hours]
    return clear * day_cloud

def wind_profile(p, rng, hours=HOURS):
    x = np.zeros(hours); x[0] = p["wind_cf"]
    for t in range(1, hours):                                         # AR(1) capacity factor
        x[t] = 0.97 * x[t - 1] + 0.03 * p["wind_cf"] + 0.04 * rng.standard_normal()
    return np.clip(x, 0, 1)

def dispatch(p, rng, solar_scale=1.0, extra_load=None):
    """Merit-order dispatch. p['gas_first'] >= 0.5 puts CCGT before coal above the coal minimum
    (coal at its minimum stable level is always on). Default: coal before CCGT (Indonesian practice)."""
    D = demand_profile(p, rng)
    if extra_load is not None:
        D = D + extra_load                                            # data-centre load added on top
    sol = p["solar"] * solar_scale * solar_profile(p, rng)
    wnd = p["wind"] * wind_profile(p, rng)
    gmust = min(p.get("ccgt_must", 0.0), p["ccgt"])                    # v10: take-or-pay gas block, always on
    must = p["geo"] * 0.9 + p["hydro_cf_mw"] + sol + wnd + gmust
    cmin = p["coal"] * p["coal_min"]
    resid = D - must
    curtail = np.clip(cmin - resid, 0, None)                         # renewables curtailed under coal min
    rem = np.clip(resid - cmin, 0, None)                              # load above coal minimum
    if p.get("gas_first", 0) >= 0.5:
        order = [("ccgt", p["ccgt"] - gmust), ("coal_x", p["coal"] - cmin), ("peaker", p["peaker"])]
    else:
        order = [("coal_x", p["coal"] - cmin), ("ccgt", p["ccgt"] - gmust), ("peaker", p["peaker"])]
    out, marg = {}, np.full(len(D), -1.0)
    for name, cap in order:
        g = np.minimum(rem, cap); rem = rem - g; out[name] = g
        ef = EF["coal"] if name == "coal_x" else EF[name]
        marg = np.where((g > 1e-6) & (g < cap - 1e-6), ef, marg)       # partly loaded unit = marginal
        marg = np.where((g >= cap - 1e-6) & (rem > 1e-6), -1.0, marg)   # fully loaded and more needed -> next
    unserved = rem
    coal = cmin + out["coal_x"]
    out["ccgt"] = out["ccgt"] + gmust
    gen_em = coal * EF["coal"] + out["ccgt"] * EF["ccgt"] + out["peaker"] * EF["peaker"]
    gen = D - unserved
    avg_ci = gen_em / gen * 1000.0                                    # gCO2/kWh
    # nothing above coal minimum: marginal is renewables if curtailed, else coal at minimum
    marg = np.where(marg < 0, np.where(curtail > 0, 0.0, EF["coal"]), marg)
    mef = marg * 1000.0
    return pd.DataFrame(dict(demand=D, solar=sol, wind=wnd, coal=coal, ccgt=out["ccgt"], peaker=out["peaker"],
                             unserved=unserved, avg_ci=avg_ci, mef=mef)).astype(np.float32)

# %% [md]
# ### Optional: load real hourly traces (Electricity Maps free historical CSV)
# Real European traces: Electricity Maps hourly 2024 data (direct carbon intensity) for Germany (DE) and
# Great Britain (GB), as redistributed in the HPE `sustain-cluster` repository (pinned commit below).
# Electricity Maps data are licensed ODbL: cite Electricity Maps as the data source. Real traces give
# **average CI only**; marginal analysis stays model-based. Leap year: the last 24 h are dropped (8760 h).

# %%
CI_REPO = ("https://raw.githubusercontent.com/HewlettPackard/sustain-cluster/"
           "3f6ea95cb835b89ba50b0ef76d66d14b8037643e/data/carbon_intensity")
REAL_CSV = {"DE_real": f"{CI_REPO}/DE-LU/2024/DE-LU_2024_hourly.csv",
            "GB_real": f"{CI_REPO}/GB/2024/GB_2024_hourly.csv"}
REAL_GRIDS = list(REAL_CSV)
def load_em_csv(path):
    df = pd.read_csv(path)
    col = [c for c in df.columns if "Carbon Intensity" in c and "direct" in c.lower()][0]
    return df[col].to_numpy()[:HOURS].astype(float)

# %%
TRACE_FILE = f"{OUT}/traces.npz"
traces = {}
if os.path.exists(TRACE_FILE):
    z = np.load(TRACE_FILE)
    names = sorted({k.split("__")[0] for k in z.files})
    for n in names:
        traces[n] = pd.DataFrame({k.split("__")[1]: z[k] for k in z.files if k.startswith(n + "__")})
    print("Loaded traces from Drive:", names)
else:
    for name, p in GRIDS.items():
        traces[name] = dispatch(p, np.random.default_rng(SEED + zlib.crc32(name.encode()) % 1000))
    for name, path in REAL_CSV.items():
        ci = load_em_csv(path).astype(np.float32)
        traces[name] = pd.DataFrame(dict(avg_ci=ci, mef=np.full_like(ci, np.nan)))
    np.savez_compressed(TRACE_FILE, **{f"{n}__{c}": df[c].to_numpy() for n, df in traces.items() for c in df.columns})

def grid_stats(v):
    ci = v.avg_ci.to_numpy()
    days = ci[:HOURS // 24 * 24].reshape(-1, 24)
    return dict(mean_ci=ci.mean(), sd_ci=ci.std(), cv_ci=ci.std() / ci.mean(),
                mean_daily_range=(days.max(1) - days.min(1)).mean(), mean_mef=v.mef.mean(),
                unserved_hours=int((v["unserved"] > 0).sum()) if "unserved" in v else 0)
summ = pd.DataFrame({k: grid_stats(v) for k, v in traces.items()}).T.round(3)
summ.to_csv(f"{OUT}/grid_summary.csv")
print(summ)

def energy_mix(v):
    E = float(v.demand.sum() - v.unserved.sum())
    return dict(coal=100 * float(v.coal.sum()) / E, gas_oil=100 * float(v.ccgt.sum() + v.peaker.sum()) / E,
                ren=100 * (E - float(v.coal.sum() + v.ccgt.sum() + v.peaker.sum())) / E,
                max_demand_MW=float(v.demand.max()))
# Load factor is NOT a calibration target; it is a diagnostic. Jawa-Bali system load factor 2024 = net production
# 230,615 GWh / (32,759 MW x 8,784 h) = 80.1 % (UIP2B). The Statistik PLN "load factor" (Jawa 63.86 %, UIP3B Sumatera
# 67.59 %) divides by a sum of unit peaks, so it is not a system load factor; no system value is available for
# Sumatera, which keeps the analytical shape.
OFFICIAL_LF = {"Jawa_Bali": 80.1, "Sumatera": float("nan")}
calib = []
for g, tgt in CALIBRATION_TARGETS.items():
    m = energy_mix(traces[g])
    for k, t in tgt.items():
        calib.append(dict(grid=g, item=k, target_pct=t, simulated_pct=round(m[k], 2), abs_err_pp=round(abs(m[k] - t), 2)))
    calib.append(dict(grid=g, item="max_demand_MW", target_pct=np.nan, simulated_pct=round(m["max_demand_MW"], 1), abs_err_pp=np.nan))
    lf = 100 * float(traces[g].demand.mean() / traces[g].demand.max())
    calib.append(dict(grid=g, item="load_factor_diagnostic", target_pct=OFFICIAL_LF.get(g, np.nan),
                      simulated_pct=round(lf, 2), abs_err_pp=round(abs(lf - OFFICIAL_LF[g]), 2) if OFFICIAL_LF.get(g, np.nan) == OFFICIAL_LF.get(g, np.nan) else np.nan))
calib = pd.DataFrame(calib)
calib.to_csv(f"{OUT}/table0_calibration_check.csv", index=False)
print(calib)
mark("traces")

fig, ax = plt.subplots(figsize=(9, 3.5))
for k, v in traces.items():
    ax.plot(v.avg_ci.to_numpy()[24 * 100: 24 * 107], label=k)
ax.set_xlabel("Hour (one sample week)"); ax.set_ylabel("Average CI (gCO2/kWh)"); ax.legend(fontsize=8)
watermark(ax); save_fig(fig, "fig1_ci_week.png")

# %% [md]
# ## 2. Workload and heuristic policies
# Synthetic delay-tolerant batch jobs (Poisson arrivals), fixed seed. Each job: duration, power, slack,
# home island and latency budget. Island-to-island latency matrix is ILLUSTRATIVE (ms).

# %%
N_JOBS = 20_000
ISLANDS = ["Jawa_Bali", "Sumatera", "Sulawesi"]
LAT = pd.DataFrame([[2, 25, 35], [25, 2, 45], [35, 45, 2]], index=ISLANDS, columns=ISLANDS)

def make_jobs(rng, n=N_JOBS):
    return pd.DataFrame(dict(
        arrival=rng.integers(0, HOURS - 72, n),
        dur=rng.integers(1, 7, n),
        kw=rng.uniform(5, 50, n),
        slack=rng.choice([0, 2, 6, 12, 24], n, p=[.2, .2, .2, .2, .2]),
        home=rng.choice(ISLANDS, n, p=[.7, .2, .1]),
        lat_budget=rng.choice([10, 30, 50], n, p=[.4, .4, .2]),
    ))
jobs = make_jobs(np.random.default_rng(SEED))

FORECAST_SIGMA = 0.10   # relative AR(1) forecast error used for the "noisy" policies
_VIEW, _CUM = {}, {}

def clear_cache():
    _VIEW.clear(); _CUM.clear()

def view(site, signal, sigma=0.0):
    """What the scheduler sees: true signal, or true signal x (1 + AR(1) relative forecast error)."""
    key = (site, signal, round(sigma, 6), id(traces[site]))
    if key not in _VIEW:
        x = traces[site][signal].to_numpy(dtype=np.float64)
        if sigma > 0:
            r = np.random.default_rng(SEED + zlib.crc32(f"{site}|{signal}".encode()) % 997)
            z = r.standard_normal(len(x)); e = np.zeros(len(x))
            for t in range(1, len(x)):
                e[t] = 0.9 * e[t - 1] + np.sqrt(1 - 0.81) * sigma * z[t]
            x = x * (1 + e)
        _VIEW[key] = x
    return _VIEW[key]

def cum(site, signal, sigma=0.0):
    key = (site, signal, round(sigma, 6), id(traces[site]))
    if key not in _CUM:
        _CUM[key] = np.concatenate([[0.0], np.cumsum(view(site, signal, sigma))])
    return _CUM[key]

def window_cost(sig, start, dur):
    """sig: raw hourly array (used by PPO env)."""
    return float(np.sum(sig[start:start + dur], dtype=np.float64))

def plan(jj, site, policy, signal="avg_ci", sigma=0.0):
    """Hours used by each job (flattened) and the MW drawn in those hours.
    agnostic: run at arrival; temporal: best contiguous block within slack; waitawhile: best d hours
    (interruptible) within arrival .. arrival + slack + d."""
    arr, dur = jj.arrival.to_numpy(), jj.dur.to_numpy()
    sl, mw = jj.slack.to_numpy(), jj.kw.to_numpy() / 1000.0
    if policy == "waitawhile":
        v = view(site, signal, sigma)
        hours = np.concatenate([a0 + np.argsort(v[a0:a0 + s_ + d], kind="stable")[:d]
                                for a0, d, s_ in zip(arr, dur, sl)])
        return hours, np.repeat(mw, dur)
    start = arr.copy()
    if policy == "temporal":
        c = cum(site, signal, sigma)
        for d in np.unique(dur):
            W = c[d:] - c[:-d]                                        # cost of a d-hour block starting at t
            for s_ in np.unique(sl[dur == d]):
                if s_ == 0:
                    continue
                k = (dur == d) & (sl == s_)
                am = np.lib.stride_tricks.sliding_window_view(W, s_ + 1).argmin(axis=1)
                start[k] = arr[k] + am[arr[k]]
    offs = np.arange(dur.sum()) - np.repeat(np.cumsum(dur) - dur, dur)
    return np.repeat(start, dur) + offs, np.repeat(mw, dur)

def _single_site(jj, site, policy, signal, sigma):
    hours, w = plan(jj, site, policy, signal, sigma)
    ta, tm = view(site, "avg_ci"), view(site, "mef")
    return np.array([np.sum(w * ta[hours]), np.sum(w * tm[hours])])

def load_profile(jj, site, policy, signal="avg_ci", sigma=0.0):
    hours, w = plan(jj, site, policy, signal, sigma)
    return np.bincount(hours, weights=w, minlength=HOURS)[:HOURS]

def run_policy(jobs, policy, signal="avg_ci", sites=None, sigma=0.0):
    """Emissions (MW x gCO2/kWh units) under average and marginal accounting.
    policy: agnostic | temporal | waitawhile | spatial | (temporal with sites = spatio-temporal)."""
    if sites is None:
        tot = np.zeros(2)
        for site, jj in jobs.groupby("home"):
            tot += _single_site(jj, site, policy, signal, sigma)
        return float(tot[0]), float(tot[1])
    em_avg = em_mef = 0.0
    for j in jobs.itertuples():
        cands = [s for s in sites if LAT.loc[j.home, s] <= j.lat_budget]
        best = None
        for s in cands:
            c = cum(s, signal, sigma)
            last = j.arrival if policy in ("agnostic", "spatial") else j.arrival + j.slack
            starts = np.arange(j.arrival, last + 1)
            vals = c[starts + j.dur] - c[starts]
            k = int(np.argmin(vals))
            if best is None or vals[k] < best[0]:
                best = (vals[k], s, int(starts[k]))
        _, s, t = best
        e = j.kw / 1000.0
        ca, cm = cum(s, "avg_ci"), cum(s, "mef")
        em_avg += e * (ca[t + j.dur] - ca[t])
        em_mef += e * (cm[t + j.dur] - cm[t])
    return em_avg, em_mef

# %%
T1 = f"{OUT}/table1_heuristic_savings.csv"
if not done("heuristic"):
    have = read_rows(T1)
    have_keys = set(zip(have.grid, have.policy)) if len(have) else set()
    POLS = [("temporal_avgCI", "avg_ci"), ("temporal_MEF", "mef"),
            ("temporal_avgCI_noisy", "avg_ci"), ("temporal_MEF_noisy", "mef"),
            ("waitawhile_avgCI", "avg_ci"), ("waitawhile_MEF", "mef")]
    tasks = [(g, pol, sig) for g in ["EU_like"] + ISLANDS for pol, sig in POLS]
    tasks += [(g, pol, sig) for g in REAL_GRIDS for pol, sig in POLS if sig == "avg_ci"]   # no MEF for real traces
    tasks += [("Indonesia_3islands", "spatial_avgCI", None), ("Indonesia_3islands", "spatiotemporal_avgCI", None)]
    complete = True
    for g, pol, sig in tasks:
        if (g, pol) in have_keys:
            continue
        if not budget_ok(f"{g}/{pol}"):
            complete = False; break
        if g == "Indonesia_3islands":
            jj = jobs
            base = run_policy(jj, "agnostic")
            r = run_policy(jj, "temporal" if pol.startswith("spatiotemporal") else "spatial", "avg_ci", sites=ISLANDS)
        else:
            jj = jobs.assign(home=g) if g in ["EU_like"] + REAL_GRIDS else jobs[jobs.home == g]
            base = run_policy(jj, "agnostic")
            kind = "waitawhile" if pol.startswith("waitawhile") else "temporal"
            r = run_policy(jj, kind, sig, sigma=FORECAST_SIGMA if pol.endswith("noisy") else 0.0)
        append_row(T1, dict(grid=g, policy=pol, n_jobs=len(jj),
                            saving_avg_acct_pct=round(100 * (1 - r[0] / base[0]), 2),
                            saving_mef_acct_pct=round(100 * (1 - r[1] / base[1]), 2) if base[1] > 0 else np.nan))
        print("done:", g, pol)
    if complete:
        mark("heuristic")
print(read_rows(T1))
gc.collect()

# %% [md]
# ## 3. PPO transfer: trained on the European-style grid, tested on Indonesian grids
# Single-job "wait or run" decision. State: normalised CI forecast for the next 24 h, remaining slack,
# job duration. Reward: −emissions (avg-CI signal) when the job runs; deadline forces execution.

# %%
if RUN_PPO:
    import importlib.util, subprocess, sys
    if importlib.util.find_spec("stable_baselines3") is None or importlib.util.find_spec("gymnasium") is None:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "gymnasium", "stable-baselines3"], check=True)
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class WaitRunEnv(gym.Env):
        def __init__(self, grid, seed=SEED):
            super().__init__()
            self.sig = traces[grid]["avg_ci"].to_numpy()
            self.scale = self.sig.mean()
            self.rng = np.random.default_rng(seed)
            self.observation_space = spaces.Box(-5, 5, (26,), np.float32)
            self.action_space = spaces.Discrete(2)  # 0 wait, 1 run
        def _obs(self):
            f = self.sig[self.t:self.t + 24] / self.scale - 1
            return np.concatenate([f, [self.left / 24, self.dur / 6]]).astype(np.float32)
        def reset(self, seed=None, options=None):
            self.t = int(self.rng.integers(0, HOURS - 80)); self.dur = int(self.rng.integers(1, 7))
            self.left = int(self.rng.choice([2, 6, 12, 24])); self.t0 = self.t
            return self._obs(), {}
        def step(self, a):
            if a == 1 or self.left == 0:
                cost = window_cost(self.sig, self.t, self.dur) / (self.scale * self.dur)
                ref = window_cost(self.sig, self.t0, self.dur) / (self.scale * self.dur)
                return self._obs(), float(ref - cost), True, False, {}
            self.t += 1; self.left -= 1
            return self._obs(), 0.0, False, False, {}

    from stable_baselines3.common.callbacks import BaseCallback

    PPO_SEEDS = [42, 43, 44]          # report mean +/- sd across seeds
    PPO_TRAIN_GRID = "DE_real"        # trained on the real German 2024 trace
    PPO_EVAL_GRIDS = REAL_GRIDS + ["EU_like"] + ISLANDS

    class DriveCheckpointAndBudget(BaseCallback):
        """Saves the model to Drive periodically and stops cleanly at the time budget."""
        def __init__(self, ckpt):
            super().__init__(); self.ckpt = ckpt
        def _on_step(self):
            if self.num_timesteps % PPO_CKPT_EVERY == 0:
                self.model.save(self.ckpt)
            if time_left() <= 0:
                self.model.save(self.ckpt)
                print("[time budget] PPO checkpoint saved; re-run to resume.")
                return False
            return True

    def eval_ppo(model, grid, n=3000):
        env = WaitRunEnv(grid, seed=SEED + 7); base = pol = orc = 0.0
        for _ in range(n):
            o, _ = env.reset(); t0, d, s_ = env.t, env.dur, env.left
            base += window_cost(env.sig, t0, d)
            orc += min(window_cost(env.sig, t, d) for t in range(t0, t0 + s_ + 1))
            finished = False
            while not finished:
                a, _ = model.predict(o, deterministic=True)
                if a == 1 or env.left == 0:
                    pol += window_cost(env.sig, env.t, d); finished = True
                else:
                    o, _, finished, _, _ = env.step(0)
        return dict(grid=grid, ppo_saving_pct=100 * (1 - pol / base), oracle_saving_pct=100 * (1 - orc / base),
                    ppo_share_of_oracle_pct=100 * (1 - pol / base) / max(1e-9, (1 - orc / base)))

    T2 = f"{OUT}/table2_ppo_transfer_by_seed.csv"
    for sd in PPO_SEEDS:
        tag = f"ppo_seed{sd}"
        if done(tag) or not budget_ok(tag):
            continue
        ckpt = f"{OUT}/{tag}.zip"
        env_train = WaitRunEnv(PPO_TRAIN_GRID, seed=sd)
        if os.path.exists(ckpt):
            model = PPO.load(ckpt, env=env_train); model.set_random_seed(sd)
            open(f"{OUT}/.{tag}_resumed", "w").write(str(model.num_timesteps))
            print(f"[{tag}] resumed at {model.num_timesteps} timesteps")
        else:
            model = PPO("MlpPolicy", env_train, seed=sd, verbose=0, n_steps=1024)
        remaining = PPO_TIMESTEPS - model.num_timesteps
        if remaining > 0:
            model.learn(remaining, callback=DriveCheckpointAndBudget(ckpt), reset_num_timesteps=False)
            model.save(ckpt)
        if model.num_timesteps < PPO_TIMESTEPS:
            break                                            # stopped by time budget
        have = read_rows(T2)
        have_g = set(have[have.seed == sd].grid) if len(have) else set()
        complete = True
        for g in PPO_EVAL_GRIDS:
            if g in have_g:
                continue
            if not budget_ok(f"{tag} eval {g}"):
                complete = False; break
            row = {k: round(v, 3) if isinstance(v, float) else v for k, v in eval_ppo(model, g).items()}
            append_row(T2, dict(seed=sd, **row))
        if complete:
            mark(tag, timesteps=int(model.num_timesteps), resumed=os.path.exists(f"{OUT}/.{tag}_resumed"))
        del env_train, model; gc.collect()

    by_seed = read_rows(T2)
    if len(by_seed):
        print(by_seed)
        agg = by_seed.groupby("grid", sort=False)[["ppo_saving_pct", "oracle_saving_pct", "ppo_share_of_oracle_pct"]] \
                     .agg(["mean", "std", "count"]).round(2)
        agg.to_csv(f"{OUT}/table2_ppo_transfer_summary.csv")
        print(agg)
    if all(done(f"ppo_seed{sd}") for sd in PPO_SEEDS):
        mark("ppo")

# %% [md]
# ## 4. Regime sweep: solar penetration on Sumatera (probe of the transition hypothesis)

# %%
T3 = f"{OUT}/table3_solar_sweep.csv"
if RUN_LEGACY and not done("sweep"):
    have = read_rows(T3)
    have_s = set(have.solar_scale) if len(have) else set()
    sub = jobs[jobs.home == "Sumatera"].copy()
    original = traces["Sumatera"]
    complete = True
    for scale in [1, 5, 10, 20, 30, 40, 50, 60]:
        if scale in have_s:
            continue
        if not budget_ok(f"sweep x{scale}"):
            complete = False; break
        p = dict(GRIDS["Sumatera"])
        tr = dispatch(p, np.random.default_rng(SEED + 99), solar_scale=scale)
        traces["Sumatera"] = tr
        base = run_policy(sub, "agnostic")
        r = run_policy(sub, "temporal", "avg_ci")
        append_row(T3, dict(solar_scale=scale, solar_MW=p["solar"] * scale,
                            solar_energy_share_pct=round(100 * float(tr.solar.sum() / tr.demand.sum()), 3),
                            cv_ci=round(float(tr.avg_ci.std() / tr.avg_ci.mean()), 3),
                            mean_ci=round(float(tr.avg_ci.mean()), 3),
                            saving_avg_acct_pct=round(100 * (1 - r[0] / base[0]), 3),
                            saving_mef_acct_pct=round(100 * (1 - r[1] / base[1]), 3) if base[1] > 0 else np.nan))
        del tr; clear_cache(); gc.collect()
    traces["Sumatera"] = original; clear_cache()
    if complete:
        mark("sweep")
sw = read_rows(T3)
if len(sw):
    sw = sw.sort_values("solar_scale"); print(sw)

if len(sw):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(sw.solar_energy_share_pct, sw.saving_avg_acct_pct, "o-", label="average-CI accounting")
    ax.plot(sw.solar_energy_share_pct, sw.saving_mef_acct_pct, "s--", label="marginal accounting")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Solar energy share (%)"); ax.set_ylabel("Temporal-shifting saving (%)"); ax.legend(fontsize=8)
    watermark(ax); save_fig(fig, "fig2_solar_sweep.png")

# %% [md]
# ## 5. Global sensitivity analysis (Sobol): which grid properties decide the sign of the saving?
# Inputs varied jointly (Saltelli/Sobol sampling, SALib): coal minimum stable level, coal and CCGT capacity
# scaling, solar build-out, merit order (coal-first vs gas-first) and forecast error of the scheduler's signal.
# Outputs: emission saving under **marginal** accounting for (Y1) the average-CI scheduler and (Y2) the
# MEF-guided scheduler, plus (Y3 = Y2 - Y1) the advantage of the marginal signal.
# Rows are written to Drive one by one, so an interrupted run resumes where it stopped.

# %%
!pip -q install SALib
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

SOBOL_N = 512            # base sample; evaluations = N * (2D + 2) per grid
SOBOL_GRIDS = ["Sumatera", "Jawa_Bali"]
PROBLEM = {
    "num_vars": 6,
    "names": ["coal_min", "coal_scale", "ccgt_scale", "solar_scale", "gas_first", "forecast_sigma"],
    "bounds": [[0.2, 0.8], [0.85, 1.3], [0.5, 1.5], [1, 60], [0, 1], [0.0, 0.2]],
}
X = sobol_sample.sample(PROBLEM, SOBOL_N, calc_second_order=False, seed=SEED)
np.save(f"{OUT}/sobol_X.npy", X)

def evaluate(grid, x):
    p = dict(GRIDS[grid])
    p.update(coal_min=x[0], coal=p["coal"] * x[1], ccgt=p["ccgt"] * x[2], gas_first=x[4])
    original = traces[grid]
    traces[grid] = dispatch(p, np.random.default_rng(SEED + 99), solar_scale=x[3]); clear_cache()
    jj = jobs[jobs.home == grid]
    base = run_policy(jj, "agnostic")
    r_avg = run_policy(jj, "temporal", "avg_ci", sigma=x[5])
    r_mef = run_policy(jj, "temporal", "mef", sigma=x[5])
    tr = traces[grid]
    out = dict(y1_avgCI_sched_mef_saving=100 * (1 - r_avg[1] / base[1]),
               y2_MEF_sched_mef_saving=100 * (1 - r_mef[1] / base[1]),
               y_avgCI_sched_avg_saving=100 * (1 - r_avg[0] / base[0]),
               unserved_hours=int((tr.unserved > 0).sum()), cv_ci=float(tr.avg_ci.std() / tr.avg_ci.mean()),
               solar_share_pct=100 * float(tr.solar.sum() / tr.demand.sum()))
    out["y3_MEF_advantage"] = out["y2_MEF_sched_mef_saving"] - out["y1_avgCI_sched_mef_saving"]
    traces[grid] = original; clear_cache()
    return out

for grid in (SOBOL_GRIDS if RUN_LEGACY else []):
    TS = f"{OUT}/sobol_runs_{grid}.csv"
    if done(f"sobol_{grid}"):
        continue
    have = read_rows(TS)
    start_i = int(have.i.max()) + 1 if len(have) else 0
    complete = True
    t0 = time.time()
    for i in range(start_i, len(X)):
        if not budget_ok(f"sobol {grid} row {i}"):
            complete = False; break
        append_row(TS, dict(i=i, **dict(zip(PROBLEM["names"], X[i])), **evaluate(grid, X[i])))
        if i % 100 == 0:
            print(f"{grid}: {i}/{len(X)} rows, {time.time() - t0:.0f}s"); gc.collect()
    if complete:
        mark(f"sobol_{grid}", n_rows=len(X))

rows = []
for grid in SOBOL_GRIDS:
    if not done(f"sobol_{grid}"):
        continue
    R = read_rows(f"{OUT}/sobol_runs_{grid}.csv").sort_values("i")
    print(f"\n{grid}: unserved-energy rows = {(R.unserved_hours > 0).sum()} of {len(R)}")
    for y in ["y1_avgCI_sched_mef_saving", "y2_MEF_sched_mef_saving", "y3_MEF_advantage"]:
        Si = sobol_analyze.analyze(PROBLEM, R[y].to_numpy(), calc_second_order=False, seed=SEED)
        for k, name in enumerate(PROBLEM["names"]):
            rows.append(dict(grid=grid, output=y, input=name, S1=Si["S1"][k], S1_conf=Si["S1_conf"][k],
                             ST=Si["ST"][k], ST_conf=Si["ST_conf"][k]))
        print(f"  {y}: mean {R[y].mean():.2f}, share of runs with saving < 0: {100 * (R[y] < 0).mean():.1f}%")
if rows:
    SI = pd.DataFrame(rows).round(4)
    SI.to_csv(f"{OUT}/table4_sobol_indices.csv", index=False)
    print(SI[SI.output == "y1_avgCI_sched_mef_saving"])
    fig, axes = plt.subplots(1, len(SOBOL_GRIDS), figsize=(5.5 * len(SOBOL_GRIDS), 3.6), squeeze=False)
    for ax, grid in zip(axes[0], SOBOL_GRIDS):
        d = SI[(SI.grid == grid) & (SI.output == "y1_avgCI_sched_mef_saving")]
        xx = np.arange(len(d))
        ax.bar(xx - 0.2, d.S1, 0.4, yerr=d.S1_conf, label="S1 (first order)")
        ax.bar(xx + 0.2, d.ST, 0.4, yerr=d.ST_conf, label="ST (total)")
        ax.set_xticks(xx); ax.set_xticklabels(d.input, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{grid}: avg-CI scheduler, marginal accounting", fontsize=9); watermark(ax)
    axes[0][0].legend(fontsize=8)
    save_fig(fig, "fig3_sobol_indices.png")

# %% [md]
# ## 6. Closing the circularity gap: consequential accounting + mis-specified marginal signals
# Two problems with sections 2 and 5: (i) emissions were counted by summing the model's own MEF, which is the
# same quantity the MEF scheduler optimises; (ii) the scheduler saw the *true* MEF (up to random noise).
# Here (i) every policy's hourly data-centre load is **added to demand and the grid is re-dispatched**; the
# saving is the change in total system emissions (consequential accounting), so herding, saturation of the
# marginal unit and unit switches are captured. (ii) The MEF scheduler uses a signal from a **planner model**
# whose parameters differ from the true system (randomly perturbed, or simply the default calibration).

# %%
GRID_SEED = {name: SEED + zlib.crc32(name.encode()) % 1000 for name in GRIDS}
LOAD_SHARES = [0.005, 0.02, 0.05]      # flexible data-centre load as share of mean grid demand

def total_emissions(tr):
    return float((tr.coal * EF["coal"] + tr.ccgt * EF["ccgt"] + tr.peaker * EF["peaker"]).sum())  # tCO2

def scaled_loads(jj, site, share, policies, mean_demand):
    raw = {name: load_profile(jj, site, *spec) for name, spec in policies.items()}
    f = share * mean_demand / raw["agnostic"].mean()          # same energy for every policy
    return {k: v * f for k, v in raw.items()}

def consequential(p, seed, solar_scale, loads):
    base = dispatch(p, np.random.default_rng(seed), solar_scale)
    e0 = total_emissions(base)
    out = {}
    for name, L in loads.items():
        tr = dispatch(p, np.random.default_rng(seed), solar_scale, extra_load=L)
        out[name] = (total_emissions(tr) - e0, int((tr.unserved > 0).sum()))
    return out

POL5 = {"agnostic": ("agnostic",), "temporal_avgCI_noisy": ("temporal", "avg_ci", FORECAST_SIGMA),
        "temporal_MEF_noisy": ("temporal", "mef", FORECAST_SIGMA),
        "waitawhile_avgCI": ("waitawhile", "avg_ci", FORECAST_SIGMA),
        "waitawhile_MEF": ("waitawhile", "mef", FORECAST_SIGMA)}

T5 = f"{OUT}/table5_consequential.csv"
if not done("consequential"):
    have = read_rows(T5)
    have_k = set(zip(have.grid, have.load_share)) if len(have) else set()
    complete = True
    for g in ISLANDS:
        jj = jobs[jobs.home == g]
        meanD = float(traces[g].demand.mean())
        for share in LOAD_SHARES:
            if (g, share) in have_k:
                continue
            if not budget_ok(f"consequential {g} {share}"):
                complete = False; break
            loads = scaled_loads(jj, g, share, POL5, meanD)
            res = consequential(GRIDS[g], GRID_SEED[g], 1.0, loads)
            d0 = res["agnostic"][0]
            for name in POL5:
                L = loads[name]
                pred = float(np.sum(L * traces[g].mef.to_numpy(np.float64))) / 1000.0     # MEF-sum prediction, t
                pred0 = float(np.sum(loads["agnostic"] * traces[g].mef.to_numpy(np.float64))) / 1000.0
                append_row(T5, dict(grid=g, load_share=share, mean_load_MW=round(share * meanD, 1), policy=name,
                                    delta_emissions_t=round(res[name][0], 1), unserved_hours=res[name][1],
                                    saving_consequential_pct=round(100 * (1 - res[name][0] / d0), 3),
                                    saving_mef_sum_pct=round(100 * (1 - pred / pred0), 3)))
        if not complete:
            break
    if complete:
        mark("consequential")
t5 = read_rows(T5)
if len(t5):
    print(t5.pivot_table(index=["grid", "load_share"], columns="policy", values="saving_consequential_pct").round(2))

# %%
MISSPEC_K = 256
T6 = f"{OUT}/table6_misspecification.csv"
BOUNDS = dict(zip(PROBLEM["names"], PROBLEM["bounds"]))

def params_from(grid, coal_min, coal_scale, ccgt_scale, gas_first):
    p = dict(GRIDS[grid]); p.update(coal_min=coal_min, coal=p["coal"] * coal_scale,
                                    ccgt=p["ccgt"] * ccgt_scale, gas_first=gas_first)
    return p

if not done("misspec"):
    have = read_rows(T6)
    have_k = set(zip(have.grid, have.k)) if len(have) else set()
    complete = True
    for g in ["Sumatera", "Jawa_Bali"]:
        rr = np.random.default_rng(SEED + 500 + zlib.crc32(g.encode()) % 100)
        jj = jobs[jobs.home == g]
        default_planner = dispatch(GRIDS[g], np.random.default_rng(GRID_SEED[g]), 1.0)
        for k in range(MISSPEC_K):
            u = {n: rr.uniform(*BOUNDS[n]) for n in ["coal_min", "coal_scale", "ccgt_scale", "solar_scale", "gas_first"]}
            pert = dict(coal_min=float(np.clip(u["coal_min"] * rr.uniform(0.8, 1.2), 0.1, 0.9)),
                        coal_scale=u["coal_scale"] * rr.uniform(0.85, 1.15),
                        ccgt_scale=u["ccgt_scale"] * rr.uniform(0.8, 1.2),
                        solar_scale=u["solar_scale"] * rr.uniform(0.7, 1.3),
                        gas_first=(1 - u["gas_first"]) if rr.uniform() < 0.25 else u["gas_first"])
            if (g, k) in have_k:
                continue
            if not budget_ok(f"misspec {g} {k}"):
                complete = False; break
            p_true = params_from(g, u["coal_min"], u["coal_scale"], u["ccgt_scale"], u["gas_first"])
            p_plan = params_from(g, pert["coal_min"], pert["coal_scale"], pert["ccgt_scale"], pert["gas_first"])
            seed = SEED + 99
            original = traces[g]
            true_tr = dispatch(p_true, np.random.default_rng(seed), u["solar_scale"])
            plan_tr = dispatch(p_plan, np.random.default_rng(seed), pert["solar_scale"])
            true_tr["mef_planner"] = plan_tr.mef.to_numpy()
            true_tr["mef_default"] = default_planner.mef.to_numpy()
            traces[g] = true_tr; clear_cache()
            pols = {"agnostic": ("agnostic",),
                    "avgCI": ("temporal", "avg_ci", FORECAST_SIGMA),
                    "MEF_oracle": ("temporal", "mef", FORECAST_SIGMA),
                    "MEF_planner_perturbed": ("temporal", "mef_planner", FORECAST_SIGMA),
                    "MEF_planner_default": ("temporal", "mef_default", FORECAST_SIGMA)}
            loads = scaled_loads(jj, g, 0.02, pols, float(true_tr.demand.mean()))
            res = consequential(p_true, seed, u["solar_scale"], loads)
            traces[g] = original; clear_cache()
            d0 = res["agnostic"][0]
            row = dict(grid=g, k=k, **{f"true_{n}": round(v, 4) for n, v in u.items()},
                       solar_share_pct=round(100 * float(true_tr.solar.sum() / true_tr.demand.sum()), 3),
                       unserved_any=int(max(v[1] for v in res.values()) > 0))
            row.update({f"saving_{n}": round(100 * (1 - res[n][0] / d0), 3) for n in pols if n != "agnostic"})
            append_row(T6, row)
            if k % 50 == 0:
                print(f"misspec {g}: {k}/{MISSPEC_K}"); gc.collect()
        if not complete:
            break
    if complete:
        mark("misspec")

m6 = read_rows(T6)
if len(m6):
    m6 = m6[m6.unserved_any == 0]
    cols = ["saving_avgCI", "saving_MEF_oracle", "saving_MEF_planner_perturbed", "saving_MEF_planner_default"]
    summ6 = []
    for g, d in m6.groupby("grid"):
        for c in cols:
            summ6.append(dict(grid=g, policy=c.replace("saving_", ""), n=len(d), mean=d[c].mean(),
                              p5=d[c].quantile(0.05), p95=d[c].quantile(0.95), share_negative_pct=100 * (d[c] < 0).mean(),
                              share_beats_avgCI_pct=100 * (d[c] > d["saving_avgCI"]).mean() if c != "saving_avgCI" else np.nan))
    summ6 = pd.DataFrame(summ6).round(2)
    summ6.to_csv(f"{OUT}/table6_misspecification_summary.csv", index=False)
    print(summ6)

# %% [md]
# ## 7. Plausibility check against official grid emission factors (ESDM)
# ESDM publishes annual grid emission factors per interconnected system (Kepmen ESDM 163.K/HK.02/MEM.S/2021,
# values for 2019), not hourly signals. The *simple operating margin* (OM) is the generation-weighted emission
# rate of fossil (non-must-run) plants, so it is compared with the model's fossil-only emission rate.
# Official values are read from the DJK-ESDM table "Faktor Emisi GRK Sistem Ketenagalistrikan Tahun 2019"
# (nilai-fe-grk-sistem-ketenagalistrikan-tahun-2019.pdf). The operating margin (OM) is the generation-weighted
# emission factor of the dispatched fossil fleet, so it is compared with `model_fossil_only`. The modelled
# Sulawesi grid is Sulbagsel; the 2019 table only has Sulselbar (South + West Sulawesi), a smaller system.
# Years and fleet differ from the 2024-2026 calibration targets, so this is a plausibility check, not a
# validation of hourly behaviour.

# %%
OFFICIAL_EF = {   # tCO2/MWh, DJK-ESDM table "Faktor Emisi GRK Sistem Ketenagalistrikan Tahun 2019"
    # official_grid: row name in the table. CM = combined margin, ex-post, OM=0.5/BM=0.5.
    "Jawa_Bali": dict(official_grid="Jamali (no. 38)",     om_2019=0.80, bm_2019=0.94, cm_2019=0.87),
    "Sumatera":  dict(official_grid="Sumatera (no. 125)",  om_2019=0.77, bm_2019=1.12, cm_2019=0.94),
    "Sulawesi":  dict(official_grid="Sulselbar (no. 123)", om_2019=0.73, bm_2019=1.17, cm_2019=0.95),
}
rows7 = []
for g in ISLANDS:
    tr = traces[g]
    em = tr.coal * EF["coal"] + tr.ccgt * EF["ccgt"] + tr.peaker * EF["peaker"]
    fos = tr.coal + tr.ccgt + tr.peaker
    rows7.append(dict(grid=g, model_system_avg=round(float(em.sum() / (tr.demand - tr.unserved).sum()), 3),
                      model_fossil_only=round(float(em.sum() / fos.sum()), 3),
                      model_mean_hourly_MEF=round(float(tr.mef.mean()) / 1000, 3), **OFFICIAL_EF[g]))
    rows7[-1]["fossil_vs_OM_pct"] = round(100 * (rows7[-1]["model_fossil_only"] / OFFICIAL_EF[g]["om_2019"] - 1), 1)
t7 = pd.DataFrame(rows7)
t7.to_csv(f"{OUT}/table7_official_ef_check.csv", index=False)
print(t7)

# %% [md]
# ## 8. Sobol with consequential accounting
# Same 6 inputs and Sobol design as section 5, but every output is a change in total system emissions after
# re-dispatch with the data-centre load (2 % of mean demand). Peaker capacity is tripled in these runs so that
# no scenario has unserved energy (otherwise "savings" could come from load that was never served).
# Outputs: avg-CI scheduler, MEF scheduler with the true signal (+ forecast error), MEF scheduler that keeps
# the default calibration (stale planner), and the advantage of the true-MEF scheduler over avg-CI.

# %%
CONSEQ_SHARE = 0.02

def evaluate_conseq(grid, x):
    p = dict(GRIDS[grid])
    p.update(coal_min=x[0], coal=p["coal"] * x[1], ccgt=p["ccgt"] * x[2], gas_first=x[4], peaker=p["peaker"] * 3)
    seed = SEED + 99
    original = traces[grid]
    tr = dispatch(p, np.random.default_rng(seed), solar_scale=x[3])
    tr["mef_default"] = DEFAULT_PLANNER[grid]
    traces[grid] = tr; clear_cache()
    pols = {"agnostic": ("agnostic",), "avgCI": ("temporal", "avg_ci", x[5]),
            "MEF_true": ("temporal", "mef", x[5]), "MEF_default": ("temporal", "mef_default", x[5])}
    jj = jobs[jobs.home == grid]
    loads = scaled_loads(jj, grid, CONSEQ_SHARE, pols, float(tr.demand.mean()))
    res = consequential(p, seed, x[3], loads)
    traces[grid] = original; clear_cache()
    d0 = res["agnostic"][0]
    out = {f"c_{k}": 100 * (1 - res[k][0] / d0) for k in ["avgCI", "MEF_true", "MEF_default"]}
    out["c_MEF_true_minus_avgCI"] = out["c_MEF_true"] - out["c_avgCI"]
    out["unserved_any"] = int(max(v[1] for v in res.values()) > 0)
    out["solar_share_pct"] = 100 * float(tr.solar.sum() / tr.demand.sum())
    return out

DEFAULT_PLANNER = {g: dispatch(GRIDS[g], np.random.default_rng(SEED + 99), 1.0).mef.to_numpy() for g in SOBOL_GRIDS}

# Reuse finished rows from the previous run (same code, same Sobol design) instead of recomputing ~25 min.
for grid in SOBOL_GRIDS:
    prev, TS = f"{BASE}/{PREV_RUN}/sobolC_runs_{grid}.csv", f"{OUT}/sobolC_runs_{grid}.csv"
    if not done(f"sobolC_{grid}") and not os.path.exists(TS) and os.path.exists(prev):
        P = pd.read_csv(prev)
        if len(P) == len(X) and np.allclose(P[PROBLEM["names"]].to_numpy(), X):
            shutil.copy(prev, TS); mark(f"sobolC_{grid}", n_rows=len(X), reused_from=PREV_RUN)
            print(f"sobolC {grid}: reused {len(P)} rows from {PREV_RUN}")

for grid in SOBOL_GRIDS:
    TS = f"{OUT}/sobolC_runs_{grid}.csv"
    if done(f"sobolC_{grid}"):
        continue
    have = read_rows(TS)
    start_i = int(have.i.max()) + 1 if len(have) else 0
    complete = True; t0 = time.time()
    for i in range(start_i, len(X)):
        if not budget_ok(f"sobolC {grid} row {i}"):
            complete = False; break
        append_row(TS, dict(i=i, **dict(zip(PROBLEM["names"], X[i])), **evaluate_conseq(grid, X[i])))
        if i % 200 == 0:
            print(f"sobolC {grid}: {i}/{len(X)} rows, {time.time() - t0:.0f}s"); gc.collect()
    if complete:
        mark(f"sobolC_{grid}", n_rows=len(X))

rowsC = []
for grid in SOBOL_GRIDS:
    if not done(f"sobolC_{grid}"):
        continue
    R = read_rows(f"{OUT}/sobolC_runs_{grid}.csv").sort_values("i")
    print(f"\n{grid}: rows with unserved energy = {int(R.unserved_any.sum())} of {len(R)}")
    for y in ["c_avgCI", "c_MEF_true", "c_MEF_default", "c_MEF_true_minus_avgCI"]:
        Si = sobol_analyze.analyze(PROBLEM, R[y].to_numpy(), calc_second_order=False, seed=SEED)
        for k, name in enumerate(PROBLEM["names"]):
            rowsC.append(dict(grid=grid, output=y, input=name, S1=Si["S1"][k], S1_conf=Si["S1_conf"][k],
                              ST=Si["ST"][k], ST_conf=Si["ST_conf"][k]))
        print(f"  {y}: mean {R[y].mean():.2f}, p5 {R[y].quantile(.05):.2f}, p95 {R[y].quantile(.95):.2f}, "
              f"negative in {100 * (R[y] < 0).mean():.1f}% of runs")
if rowsC:
    SC = pd.DataFrame(rowsC).round(4)
    SC.to_csv(f"{OUT}/table8_sobol_consequential.csv", index=False)
    fig, axes = plt.subplots(1, len(SOBOL_GRIDS), figsize=(5.5 * len(SOBOL_GRIDS), 3.6), squeeze=False)
    for ax, grid in zip(axes[0], SOBOL_GRIDS):
        d = SC[(SC.grid == grid) & (SC.output == "c_avgCI")]
        xx = np.arange(len(d))
        ax.bar(xx - 0.2, d.S1, 0.4, yerr=d.S1_conf, label="S1 (first order)")
        ax.bar(xx + 0.2, d.ST, 0.4, yerr=d.ST_conf, label="ST (total)")
        ax.set_xticks(xx); ax.set_xticklabels(d.input, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{grid}: avg-CI scheduler, consequential", fontsize=9); watermark(ax)
    axes[0][0].legend(fontsize=8)
    save_fig(fig, "fig4_sobol_consequential.png")

# %% [md]
# ## 10. Consequential table without unserved energy (helper functions)
# All v7 stages use **consequential accounting** (re-dispatch with the data-centre load) and peaker capacity x3
# (same convention as section 8), so no "saving" can come from load that was never served. Table 5b repeats
# table 5 under this convention; Sulawesi at 2-5 % load is only reportable from table 5b.

# %%
PEAKER_X = 3

def with_reserve(p):
    q = dict(p); q["peaker"] = p["peaker"] * PEAKER_X
    return q

def ar1_noise(n, sigma, seed):
    r = np.random.default_rng(seed); z = r.standard_normal(n); e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.9 * e[t - 1] + np.sqrt(1 - 0.81) * sigma * z[t]
    return e

def conseq_signals(grid, p_true, solar_scale, seed, signals, share, sigma=FORECAST_SIGMA, tr=None, noise_seed=None):
    """signals: name -> trace column name (str) or hourly array that the scheduler sees (before forecast noise).
    Returns consequential saving (%) of each signal-guided temporal scheduler vs carbon-agnostic.
    Table 5b therefore differs slightly from table 5, which drew a separate noise path per signal."""
    original = traces[grid]
    tr = (dispatch(p_true, np.random.default_rng(seed), solar_scale) if tr is None else tr).copy()
    # common random numbers: every signal gets the SAME AR(1) forecast-error path, so differences between
    # signals come from the signal itself, not from different noise draws
    e = ar1_noise(len(tr), sigma, SEED + zlib.crc32(grid.encode()) % 997 if noise_seed is None else noise_seed)
    pols = {"agnostic": ("agnostic",)}
    for name, sig in signals.items():
        x = tr[sig].to_numpy(np.float64) if isinstance(sig, str) else np.asarray(sig, dtype=np.float64)
        tr[f"sig_{name}"] = (x * (1 + e)).astype(np.float32)
        pols[name] = ("temporal", f"sig_{name}", 0.0)
    traces[grid] = tr; clear_cache()
    loads = scaled_loads(jobs[jobs.home == grid], grid, share, pols, float(tr.demand.mean()))
    res = consequential(p_true, seed, solar_scale, loads)
    traces[grid] = original; clear_cache()
    d0 = res["agnostic"][0]
    out = {n: 100 * (1 - res[n][0] / d0) for n in signals}
    out["unserved_any"] = int(max(v[1] for v in res.values()) > 0)
    out["solar_share_pct"] = 100 * float(tr.solar.sum() / tr.demand.sum())
    return out

T5B = f"{OUT}/table5b_consequential_reserve.csv"
if not done("consequential_reserve"):
    have = read_rows(T5B)
    have_k = set(zip(have.grid, have.load_share)) if len(have) else set()
    complete = True
    for g in ISLANDS:
        for share in LOAD_SHARES:
            if (g, share) in have_k:
                continue
            if not budget_ok(f"consequential_reserve {g} {share}"):
                complete = False; break
            r = conseq_signals(g, with_reserve(GRIDS[g]), 1.0, GRID_SEED[g],
                               {"avgCI": "avg_ci", "MEF": "mef"}, share, tr=traces[g])
            append_row(T5B, dict(grid=g, load_share=share, saving_avgCI=round(r["avgCI"], 3),
                                 saving_MEF=round(r["MEF"], 3), unserved_any=r["unserved_any"]))
        if not complete:
            break
    if complete:
        mark("consequential_reserve")
print(read_rows(T5B))

# %% [md]
# ## 11. Ablation of the marginal-signal structure
# Which parts of the structural signal carry its value? The true system is fixed; only the signal the scheduler
# sees changes. Variants (all built from the same weather year):
# - `MEF_full`: merit order + coal minimum stable level + curtailment (the proposed signal)
# - `MEF_no_coalmin`: planner ignores the coal minimum stable level (no must-run coal, no curtailment hours)
# - `MEF_wrong_order`: planner assumes the opposite merit order (coal-first <-> gas-first)
# - `MEF_hour_profile`: only a 24-value hour-of-day MEF table (what a static time-of-use table would give)
# - `net_load`: demand minus solar and wind (a common proxy that needs no emission model)
# - `avgCI`: average carbon intensity
# Scenarios: the calibrated grids (k = 0) plus `ABL_K` random grids drawn from the Sobol bounds.

# %%
ABL_K = 64
ABL_SHARES = [0.005, 0.02]
T9 = f"{OUT}/table9_ablation.csv"
ABL_SIGNALS = ["MEF_full", "MEF_no_coalmin", "MEF_wrong_order", "MEF_hour_profile", "net_load", "avgCI"]

def ablation_signals(p_true, solar_scale, seed, tr):
    sig = {"MEF_full": "mef", "avgCI": "avg_ci"}
    sig["MEF_no_coalmin"] = dispatch(dict(p_true, coal_min=0.0), np.random.default_rng(seed), solar_scale).mef.to_numpy()
    flip = dict(p_true, gas_first=0.0 if p_true.get("gas_first", 0) >= 0.5 else 1.0)
    sig["MEF_wrong_order"] = dispatch(flip, np.random.default_rng(seed), solar_scale).mef.to_numpy()
    m = tr.mef.to_numpy(np.float64)
    sig["MEF_hour_profile"] = np.tile(m[:HOURS // 24 * 24].reshape(-1, 24).mean(0), HOURS // 24 + 1)[:HOURS]
    sig["net_load"] = (tr.demand - tr.solar - tr.wind).to_numpy()
    return sig

if not done("ablation"):
    have = read_rows(T9)
    have_k = set(zip(have.grid, have.k, have.load_share)) if len(have) else set()
    complete = True
    jobs_list = [(g, 0) for g in ISLANDS] + [(g, k) for g in SOBOL_GRIDS for k in range(1, ABL_K + 1)]
    for g, k in jobs_list:
        if all((g, k, sh) in have_k for sh in ABL_SHARES):
            continue
        if not budget_ok(f"ablation {g} {k}"):
            complete = False; break
        if k == 0:
            x = dict(coal_min=GRIDS[g]["coal_min"], coal_scale=1.0, ccgt_scale=1.0, solar_scale=1.0,
                     gas_first=float(GRIDS[g].get("gas_first", 0)))
        else:
            rr = np.random.default_rng(SEED + 700 + 1000 * SOBOL_GRIDS.index(g) + k)
            x = {n: float(rr.uniform(*BOUNDS[n])) for n in ["coal_min", "coal_scale", "ccgt_scale", "solar_scale", "gas_first"]}
            x["gas_first"] = float(x["gas_first"] >= 0.5)
        p_true = with_reserve(params_from(g, x["coal_min"], x["coal_scale"], x["ccgt_scale"], x["gas_first"]))
        seed = SEED + 99
        tr = dispatch(p_true, np.random.default_rng(seed), x["solar_scale"])
        sig = ablation_signals(p_true, x["solar_scale"], seed, tr)
        for sh in ABL_SHARES:
            if (g, k, sh) in have_k:
                continue
            r = conseq_signals(g, p_true, x["solar_scale"], seed, sig, sh, tr=tr)
            append_row(T9, dict(grid=g, k=k, load_share=sh, **{f"x_{n}": round(v, 4) for n, v in x.items()},
                                solar_share_pct=round(r["solar_share_pct"], 3), unserved_any=r["unserved_any"],
                                **{f"saving_{n}": round(r[n], 3) for n in ABL_SIGNALS}))
        if k % 16 == 0:
            gc.collect()
    if complete:
        mark("ablation")

t9 = read_rows(T9)
if len(t9):
    t9 = t9[t9.unserved_any == 0]
    rows9 = []
    for (g, sh, base), d in t9.groupby(["grid", "load_share", t9.k == 0]):
        full = d["saving_MEF_full"].mean()
        for n in ABL_SIGNALS:
            c = d[f"saving_{n}"]
            rows9.append(dict(grid=g, load_share_pct=100 * sh, scenarios="calibrated" if base else f"random (n={len(d)})",
                              signal=n, mean=c.mean(), p5=c.quantile(0.05), share_negative_pct=100 * (c < 0).mean(),
                              retained_vs_full_pct=100 * c.mean() / full if full != 0 else np.nan,
                              share_beats_avgCI_pct=100 * (c > d["saving_avgCI"]).mean() if n != "avgCI" else np.nan))
    T9S = pd.DataFrame(rows9).round(2)
    T9S.to_csv(f"{OUT}/table9_ablation_summary.csv", index=False)
    print(T9S.to_string())

# %% [md]
# ## 12. Regime map: where does the sign of the saving flip?
# Grid of solar build-out x coal minimum stable level, for both merit orders, at 2 % data-centre load.
# Colour = consequential saving of the average-CI scheduler; the black line is the zero contour.

# %%
RM_SOLAR = [1, 2, 4, 7, 12, 20, 30, 45, 60]
RM_COALMIN = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
T10 = f"{OUT}/table10_regime_map.csv"
if not done("regime_map"):
    have = read_rows(T10)
    have_k = set(zip(have.grid, have.gas_first, have.coal_min, have.solar_scale)) if len(have) else set()
    complete = True
    for g in SOBOL_GRIDS:
        for gf in [0.0, 1.0]:
            for cm in RM_COALMIN:
                for sc in RM_SOLAR:
                    if (g, gf, cm, sc) in have_k:
                        continue
                    if not budget_ok(f"regime_map {g}"):
                        complete = False; break
                    p_true = with_reserve(params_from(g, cm, 1.0, 1.0, gf))
                    r = conseq_signals(g, p_true, sc, SEED + 99, {"avgCI": "avg_ci", "MEF": "mef"}, CONSEQ_SHARE)
                    append_row(T10, dict(grid=g, gas_first=gf, coal_min=cm, solar_scale=sc,
                                         solar_share_pct=round(r["solar_share_pct"], 3),
                                         saving_avgCI=round(r["avgCI"], 3), saving_MEF=round(r["MEF"], 3),
                                         unserved_any=r["unserved_any"]))
                if not complete: break
            if not complete: break
        if not complete: break
    if complete:
        mark("regime_map")

t10 = read_rows(T10)
if done("regime_map"):
    fig, axes = plt.subplots(len(SOBOL_GRIDS), 2, figsize=(10, 3.6 * len(SOBOL_GRIDS)), squeeze=False)
    for i, g in enumerate(SOBOL_GRIDS):
        for j, gf in enumerate([0.0, 1.0]):
            d = t10[(t10.grid == g) & (t10.gas_first == gf)]
            Z = d.pivot(index="coal_min", columns="solar_scale", values="saving_avgCI")
            share = d.groupby("solar_scale").solar_share_pct.mean().reindex(Z.columns)
            ax = axes[i][j]; lim = np.nanmax(np.abs(Z.values))
            im = ax.pcolormesh(share.values, Z.index.values, Z.values, cmap="RdBu", vmin=-lim, vmax=lim, shading="nearest")
            ax.contour(share.values, Z.index.values, Z.values, levels=[0], colors="k", linewidths=1.2)
            ax.set_xscale("log"); ax.set_xlabel("Solar energy share (%)"); ax.set_ylabel("Coal minimum stable level")
            ax.set_title(f"{g}, {'gas-first' if gf else 'coal-first'}: avg-CI saving (%)", fontsize=9)
            fig.colorbar(im, ax=ax); watermark(ax)
    save_fig(fig, "fig5_regime_map.png")

# %% [md]
# ## 13. Calibration age: how fast does a marginal signal go stale?
# The true grid moves along a solar build-out path (optionally switching to gas-first half-way, e.g. after a
# policy change). Planners: `MEF_recalibrated` (re-calibrated every step), `MEF_lag1` (one step behind),
# `MEF_stale` (calibrated once at the start), and the average-CI scheduler.

# %%
CA_SOLAR = [1, 2, 4, 7, 12, 20, 30, 45, 60]
T11 = f"{OUT}/table11_calibration_age.csv"
if not done("calib_age"):
    have = read_rows(T11)
    have_k = set(zip(have.grid, have.path, have.step)) if len(have) else set()
    complete = True
    for g in SOBOL_GRIDS:
        for path in ["solar_only", "solar_then_gas_first"]:
            seed = SEED + 99
            p0 = with_reserve(GRIDS[g])
            stale = dispatch(p0, np.random.default_rng(seed), CA_SOLAR[0]).mef.to_numpy()
            for step, sc in enumerate(CA_SOLAR):
                if (g, path, step) in have_k:
                    continue
                if not budget_ok(f"calib_age {g} {path} {step}"):
                    complete = False; break
                gf = 1.0 if (path == "solar_then_gas_first" and step >= len(CA_SOLAR) // 2) else float(GRIDS[g].get("gas_first", 0))
                gf_prev = 1.0 if (path == "solar_then_gas_first" and step - 1 >= len(CA_SOLAR) // 2) else float(GRIDS[g].get("gas_first", 0))
                p_true = dict(p0, gas_first=gf)
                lag = dispatch(dict(p0, gas_first=gf_prev), np.random.default_rng(seed), CA_SOLAR[max(step - 1, 0)]).mef.to_numpy()
                r = conseq_signals(g, p_true, sc, seed, {"avgCI": "avg_ci", "MEF_recalibrated": "mef",
                                                          "MEF_lag1": lag, "MEF_stale": stale}, CONSEQ_SHARE)
                append_row(T11, dict(grid=g, path=path, step=step, solar_scale=sc, gas_first=gf,
                                     solar_share_pct=round(r["solar_share_pct"], 3), unserved_any=r["unserved_any"],
                                     **{f"saving_{n}": round(r[n], 3) for n in ["avgCI", "MEF_recalibrated", "MEF_lag1", "MEF_stale"]}))
            if not complete: break
        if not complete: break
    if complete:
        mark("calib_age")

t11 = read_rows(T11)
if done("calib_age"):
    fig, axes = plt.subplots(len(SOBOL_GRIDS), 2, figsize=(10, 3.4 * len(SOBOL_GRIDS)), squeeze=False)
    for i, g in enumerate(SOBOL_GRIDS):
        for j, path in enumerate(["solar_only", "solar_then_gas_first"]):
            d = t11[(t11.grid == g) & (t11.path == path)].sort_values("step"); ax = axes[i][j]
            for n, st in [("MEF_recalibrated", "o-"), ("MEF_lag1", "s--"), ("MEF_stale", "^:"), ("avgCI", "x-")]:
                ax.plot(d.solar_share_pct, d[f"saving_{n}"], st, label=n, ms=4)
            ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log")
            ax.set_xlabel("Solar energy share (%)"); ax.set_ylabel("Consequential saving (%)")
            ax.set_title(f"{g}: {path}", fontsize=9); watermark(ax)
    axes[0][0].legend(fontsize=7)
    save_fig(fig, "fig6_calibration_age.png")

# %% [md]
# ## 14. Sobol convergence
# The Sobol design is nested (the first N*(D+2) rows of a larger design equal the smaller design; checked below),
# so N = 256 is a subset of the existing N = 512 runs and only rows 4096-8191 are new for N = 1024.
# Stable total indices (ST) across N show the regime attribution is not a sampling artefact.

# %%
SOBOL_N_CONV = 1024
X_CONV = sobol_sample.sample(PROBLEM, SOBOL_N_CONV, calc_second_order=False, seed=SEED)
NESTED = bool(np.allclose(X_CONV[:len(X)], X))
print("Sobol design nested:", NESTED)
if not done("sobol_convergence") and NESTED:
    complete = True
    for grid in SOBOL_GRIDS:
        TE = f"{OUT}/sobolC_ext_runs_{grid}.csv"
        have = read_rows(TE)
        start_i = int(have.i.max()) + 1 if len(have) else len(X)
        t0 = time.time()
        for i in range(start_i, len(X_CONV)):
            if not budget_ok(f"sobol_convergence {grid} row {i}"):
                complete = False; break
            append_row(TE, dict(i=i, **dict(zip(PROBLEM["names"], X_CONV[i])), **evaluate_conseq(grid, X_CONV[i])))
            if i % 400 == 0:
                print(f"sobol ext {grid}: {i}/{len(X_CONV)} rows, {time.time() - t0:.0f}s"); gc.collect()
        if not complete:
            break
    if complete:
        mark("sobol_convergence", n_rows=len(X_CONV))

if done("sobol_convergence"):
    D2 = PROBLEM["num_vars"] + 2
    rows12 = []
    for grid in SOBOL_GRIDS:
        R = pd.concat([read_rows(f"{OUT}/sobolC_runs_{grid}.csv"),
                       read_rows(f"{OUT}/sobolC_ext_runs_{grid}.csv")]).sort_values("i")
        for N in [SOBOL_N // 2, SOBOL_N, SOBOL_N_CONV]:
            sub = R[R.i < N * D2]
            for y in ["c_avgCI", "c_MEF_true_minus_avgCI"]:
                Si = sobol_analyze.analyze(PROBLEM, sub[y].to_numpy(), calc_second_order=False, seed=SEED)
                for k, name in enumerate(PROBLEM["names"]):
                    rows12.append(dict(grid=grid, N=N, output=y, input=name, ST=Si["ST"][k], ST_conf=Si["ST_conf"][k],
                                       S1=Si["S1"][k], S1_conf=Si["S1_conf"][k]))
    T12 = pd.DataFrame(rows12).round(4)
    T12.to_csv(f"{OUT}/table12_sobol_convergence.csv", index=False)
    print(T12[T12.output == "c_avgCI"].pivot_table(index=["grid", "input"], columns="N", values="ST"))

# %% [md]
# ## 16. Emission-factor envelope (v8): are the headline results robust to the unreported coal quality?
# Official data bound the emission factors but do not pin them down (section 1). Every set in `EF_SETS` is applied
# to the true system and to the signals; we repeat (a) the plausibility check against the official operating margin,
# (b) table 5b (consequential savings with reserve) and (c) the regime-map sign counts. The `base_v8` rows must equal
# sections 7, 10 and 12 (printed check).

# %%
T16R = f"{OUT}/table16_ef_envelope_runs.csv"
T16 = f"{OUT}/table16_ef_envelope.csv"
EF_BASE = dict(EF)
if not done("ef_sensitivity"):
    have = read_rows(T16R)
    have_k = set(zip(have.ef_set, have.kind, have.grid, have.key)) if len(have) else set()
    complete = True
    for es, vals in EF_SETS.items():
        EF.update(vals); clear_cache()
        try:
            for g in ISLANDS:
                tr = dispatch(GRIDS[g], np.random.default_rng(GRID_SEED[g]))
                k = (es, "om_check", g, "-")
                if k not in have_k:
                    em = tr.coal * EF["coal"] + tr.ccgt * EF["ccgt"] + tr.peaker * EF["peaker"]
                    fos = float((tr.coal + tr.ccgt + tr.peaker).sum())
                    append_row(T16R, dict(ef_set=es, kind="om_check", grid=g, key="-",
                                          v1=round(float(em.sum()) / fos, 4),
                                          v2=round(float(em.sum() / (tr.demand - tr.unserved).sum()), 4)))
                for share in [0.005, 0.02]:
                    k = (es, "conseq", g, str(share))
                    if k in have_k:
                        continue
                    if not budget_ok(f"ef_sensitivity {es} {g}"):
                        complete = False; break
                    r = conseq_signals(g, with_reserve(GRIDS[g]), 1.0, GRID_SEED[g],
                                       {"avgCI": "avg_ci", "MEF": "mef"}, share, tr=tr)
                    append_row(T16R, dict(ef_set=es, kind="conseq", grid=g, key=str(share),
                                          v1=round(r["avgCI"], 3), v2=round(r["MEF"], 3)))
                if not complete: break
            for g in SOBOL_GRIDS:
                if not complete: break
                for gf in [0.0, 1.0]:
                    for cm in RM_COALMIN:
                        for sc in RM_SOLAR:
                            key = f"{gf}|{cm}|{sc}"
                            if (es, "regime", g, key) in have_k:
                                continue
                            if not budget_ok(f"ef_sensitivity regime {es} {g}"):
                                complete = False; break
                            r = conseq_signals(g, with_reserve(params_from(g, cm, 1.0, 1.0, gf)), sc, SEED + 99,
                                               {"avgCI": "avg_ci", "MEF": "mef"}, CONSEQ_SHARE)
                            append_row(T16R, dict(ef_set=es, kind="regime", grid=g, key=key,
                                                  v1=round(r["avgCI"], 3), v2=round(r["MEF"], 3)))
                        if not complete: break
                    if not complete: break
        finally:
            EF.clear(); EF.update(EF_BASE); clear_cache()
        if not complete:
            break
    if complete:
        mark("ef_sensitivity")

if done("ef_sensitivity"):
    R = read_rows(T16R)
    rows16 = []
    for es, vals in EF_SETS.items():
        d = R[R.ef_set == es]
        row = dict(ef_set=es, **{f"ef_{k}": v for k, v in vals.items()},
                   coal_to_ccgt=round(vals["coal"] / vals["ccgt"], 2))
        for g in ISLANDS:
            o = d[(d.kind == "om_check") & (d.grid == g)]
            row[f"{g}_fossil_EF"] = float(o.v1.iloc[0])
            row[f"{g}_fossil_vs_OM2019_pct"] = round(100 * (float(o.v1.iloc[0]) / OFFICIAL_EF[g]["om_2019"] - 1), 1)
            c = d[(d.kind == "conseq") & (d.grid == g) & (d.key == "0.02")]
            row[f"{g}_avgCI_2pct"] = float(c.v1.iloc[0]); row[f"{g}_MEF_2pct"] = float(c.v2.iloc[0])
        for g in SOBOL_GRIDS:
            for gf, lab in [(0.0, "coalfirst"), (1.0, "gasfirst")]:
                rg = d[(d.kind == "regime") & (d.grid == g) & d.key.str.startswith(f"{gf}|")]
                row[f"{g}_{lab}_avgCI_neg_cells"] = f"{int((rg.v1 < 0).sum())}/{len(rg)}"
                row[f"{g}_{lab}_MEF_min"] = float(rg.v2.min())
        rows16.append(row)
    t16 = pd.DataFrame(rows16)
    t16.to_csv(T16, index=False)
    print(t16.T)
    # consistency check: base_v8 must reproduce sections 10 and 12
    b = R[(R.ef_set == "base_v8") & (R.kind == "conseq") & (R.key == "0.02")]
    t5b = read_rows(T5B); t5b = t5b[np.isclose(t5b.load_share, 0.02)]
    chk = b.merge(t5b, on="grid")
    print("base_v8 == table 5b:", bool(np.allclose(chk.v1, chk.saving_avgCI) and np.allclose(chk.v2, chk.saving_MEF)))

# %% [md]
# ## 17. Replicates over weather and demand realisations (v9)
# Sections 10, 12 and 13 use one synthetic year. Here each is repeated for 10 seeds; a seed changes the demand
# noise, the daily cloud cover and the forecast-error path. Results are reported as mean ± SD across seeds.

# %%
REP_SEEDS = [SEED + 1000 + 37 * k for k in range(10)]
T17A, T17B, T17C = (f"{OUT}/table17a_rep_consequential.csv", f"{OUT}/table17b_rep_regime_map.csv",
                    f"{OUT}/table17c_rep_calibration_age.csv")
SIG2 = {"avgCI": "avg_ci", "MEF": "mef"}
if not done("rep_seeds"):
    complete = True
    A = read_rows(T17A); ka = set(zip(A.seed, A.grid, A.load_share)) if len(A) else set()
    B = read_rows(T17B); kb = set(zip(B.seed, B.grid, B.gas_first, B.coal_min, B.solar_scale)) if len(B) else set()
    C = read_rows(T17C); kc = set(zip(C.seed, C.grid, C.path, C.step)) if len(C) else set()
    for sd in REP_SEEDS:
        for g in ISLANDS:
            for share in LOAD_SHARES:
                if (sd, g, share) in ka: continue
                if not budget_ok("rep_seeds E1"): complete = False; break
                r = conseq_signals(g, with_reserve(GRIDS[g]), 1.0, sd, SIG2, share, noise_seed=sd + 7)
                append_row(T17A, dict(seed=sd, grid=g, load_share=share, saving_avgCI=round(r["avgCI"], 3),
                                      saving_MEF=round(r["MEF"], 3), unserved_any=r["unserved_any"]))
            if not complete: break
        if not complete: break
        for g in SOBOL_GRIDS:
            for gf in [0.0, 1.0]:
                for cm in RM_COALMIN:
                    for sc in RM_SOLAR:
                        if (sd, g, gf, cm, sc) in kb: continue
                        if not budget_ok("rep_seeds regime"): complete = False; break
                        r = conseq_signals(g, with_reserve(params_from(g, cm, 1.0, 1.0, gf)), sc, sd, SIG2,
                                           CONSEQ_SHARE, noise_seed=sd + 7)
                        append_row(T17B, dict(seed=sd, grid=g, gas_first=gf, coal_min=cm, solar_scale=sc,
                                              solar_share_pct=round(r["solar_share_pct"], 3),
                                              saving_avgCI=round(r["avgCI"], 3), saving_MEF=round(r["MEF"], 3),
                                              unserved_any=r["unserved_any"]))
                    if not complete: break
                if not complete: break
            if not complete: break
        if not complete: break
        for g in SOBOL_GRIDS:
            for path in ["solar_only", "solar_then_gas_first"]:
                p0 = with_reserve(GRIDS[g])
                stale = dispatch(p0, np.random.default_rng(sd), CA_SOLAR[0]).mef.to_numpy()
                for step, sc in enumerate(CA_SOLAR):
                    if (sd, g, path, step) in kc: continue
                    if not budget_ok("rep_seeds calib_age"): complete = False; break
                    half = len(CA_SOLAR) // 2
                    gf = 1.0 if (path == "solar_then_gas_first" and step >= half) else 0.0
                    gf_prev = 1.0 if (path == "solar_then_gas_first" and step - 1 >= half) else 0.0
                    lag = dispatch(dict(p0, gas_first=gf_prev), np.random.default_rng(sd),
                                   CA_SOLAR[max(step - 1, 0)]).mef.to_numpy()
                    r = conseq_signals(g, dict(p0, gas_first=gf), sc, sd,
                                       {"avgCI": "avg_ci", "MEF_recalibrated": "mef", "MEF_lag1": lag,
                                        "MEF_stale": stale}, CONSEQ_SHARE, noise_seed=sd + 7)
                    append_row(T17C, dict(seed=sd, grid=g, path=path, step=step, solar_scale=sc, gas_first=gf,
                                          solar_share_pct=round(r["solar_share_pct"], 3), unserved_any=r["unserved_any"],
                                          **{f"saving_{n}": round(r[n], 3) for n in
                                             ["avgCI", "MEF_recalibrated", "MEF_lag1", "MEF_stale"]}))
                if not complete: break
            if not complete: break
        if not complete: break
    if complete:
        mark("rep_seeds")

if done("rep_seeds"):
    A, B, C = read_rows(T17A), read_rows(T17B), read_rows(T17C)
    print("unserved runs:", int(A.unserved_any.sum()), int(B.unserved_any.sum()), int(C.unserved_any.sum()))
    s17a = A.groupby(["grid", "load_share"])[["saving_avgCI", "saving_MEF"]].agg(["mean", "std", "min", "max"]).round(2)
    s17a.to_csv(f"{OUT}/table17a_summary.csv"); print(s17a)
    per = (B.assign(neg=B.saving_avgCI < 0).groupby(["seed", "grid", "gas_first"])
           .agg(avgCI_neg_cells=("neg", "sum"), MEF_min=("saving_MEF", "min"), MEF_neg_cells=("saving_MEF", lambda x: int((x < 0).sum()))))
    s17b = per.groupby(["grid", "gas_first"]).agg(["mean", "std", "min", "max"]).round(2)
    s17b.to_csv(f"{OUT}/table17b_summary.csv"); print(s17b)
    s17b2 = (B.groupby(["grid", "gas_first", "coal_min", "solar_scale"])
             .agg(solar_share_pct=("solar_share_pct", "mean"), avgCI_mean=("saving_avgCI", "mean"),
                  avgCI_sd=("saving_avgCI", "std"), avgCI_share_negative=("saving_avgCI", lambda x: float((x < 0).mean())))
             .round(3).reset_index())
    s17b2.to_csv(f"{OUT}/table17b_cells.csv", index=False)
    cols = ["saving_avgCI", "saving_MEF_recalibrated", "saving_MEF_lag1", "saving_MEF_stale"]
    s17c = C.groupby(["grid", "path", "step"]).agg(solar_share_pct=("solar_share_pct", "mean"),
                                                  **{f"{c}_mean": (c, "mean") for c in cols},
                                                  **{f"{c}_sd": (c, "std") for c in cols}).round(2).reset_index()
    s17c.to_csv(f"{OUT}/table17c_summary.csv", index=False); print(s17c)
    # Fig. 8: harm valley with seed bands (coal-first, avg-CI scheduler)
    fig, axes = plt.subplots(1, len(SOBOL_GRIDS), figsize=(10, 3.6), squeeze=False)
    for i, g in enumerate(SOBOL_GRIDS):
        ax = axes[0][i]
        for cm in [0.2, 0.4, 0.6, 0.8]:
            d = s17b2[(s17b2.grid == g) & (s17b2.gas_first == 0.0) & np.isclose(s17b2.coal_min, cm)].sort_values("solar_scale")
            ax.errorbar(d.solar_share_pct, d.avgCI_mean, yerr=d.avgCI_sd, marker="o", ms=3, capsize=2, label=f"coal min {cm}")
        ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log")
        ax.set_xlabel("Solar energy share (%)"); ax.set_ylabel("avg-CI consequential saving (%)")
        ax.set_title(f"{g}, coal-first: mean ± SD over {len(REP_SEEDS)} seeds", fontsize=9); watermark(ax)
    axes[0][0].legend(fontsize=7)
    save_fig(fig, "fig8_valley_replicates.png")

# %% [md]
# ## 18. Load-shape robustness (v10)
# Jawa-Bali now uses the empirical UIP2B 2024 shape (weekday / Saturday / Sunday curves, weekly peak envelope,
# holidays); observed daily minimum / daily peak = 0.78 (weekday), 0.83 (Saturday), 0.79 (Sunday), 0.74 (Idul Fitri).
# Variants, each re-calibrated to the same 2024 energy mix:
# - Jawa-Bali `p2b_flat_season`: the same daily curves without the weekly envelope and holidays;
# - Jawa-Bali `coal_cap_low` / `coal_cap_high`: the two ends of the feasible calibration range of effective coal
#   capacity and take-or-pay gas block (21,250/3,750 MW and 23,500/4,500 MW; base 22,250/4,250 MW);
# - Jawa-Bali `analytic_bf_0.70`: the analytical v9 shape on the v10 grid, for comparison with the empirical shape;
# - Sumatera `analytic_bf_x`: the analytical shape with base fraction x (0.50-0.75), a sweep of the daily swing;
# - Sumatera `calendar`: analytical shape with growth and holidays (no system load curve is published for Sumatera).
# E1 (2 % load) is repeated for every variant; the regime map for Jawa-Bali `analytic_bf_0.70` (the v9 base shape,
# for comparison) and Sumatera `calendar`.

# %%
T18, T18C = f"{OUT}/table18_load_shape.csv", f"{OUT}/table18_load_shape_calibration.csv"
CALENDAR = dict(growth=0.05, holidays=[(99, 7, 0.65), (106, 7, 0.85), (358, 7, 0.90)])
BF_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
ANALYTIC_PEAK = {"Jawa_Bali": 31013}       # analytical shape: simulated maximum ~32,759 MW
OBSERVED_SWING = {"Jawa_Bali": (0.74, 0.83)}
REGIME_VARIANTS = {"Jawa_Bali": ["analytic_bf_0.70"], "Sumatera": ["calendar"]}

def energy_mix(p, seed):
    tr = dispatch(p, np.random.default_rng(seed))
    E = float((tr.demand - tr.unserved).sum())
    coal, gas = float(tr.coal.sum()), float((tr.ccgt + tr.peaker).sum())
    return dict(coal=100 * coal / E, gas_oil=100 * gas / E, ren=100 * (E - coal - gas) / E,
                unserved=int((tr.unserved > 0).sum()))

def load_stats(p, seed):
    d = demand_profile(p, np.random.default_rng(seed))
    days = d[:len(d) // 24 * 24].reshape(-1, 24)
    return 100 * float(d.mean() / d.max()), float((days.min(1) / days.max(1)).mean())

def fit_base_frac(p, seed, lf_target):
    lo, hi = 0.05, p["base_frac"]
    for _ in range(40):
        mid = (lo + hi) / 2
        if load_stats(dict(p, base_frac=mid), seed)[0] > lf_target: hi = mid
        else: lo = mid
    return round((lo + hi) / 2, 4)

def recalibrate(g, p):
    tgt = CALIBRATION_TARGETS[g]; best = None
    for f in np.linspace(0.5, 1.5, 21):
        for cs in np.linspace(0.7, 1.2, 26):
            q = dict(p, hydro_cf_mw=round(GRIDS[g]["hydro_cf_mw"] * f), geo=round(GRIDS[g]["geo"] * f),
                     coal=round(GRIDS[g]["coal"] * cs))
            m = energy_mix(q, GRID_SEED[g])
            err = max(abs(m[k] - tgt[k]) for k in tgt)
            if m["unserved"] == 0 and (best is None or err < best[0]):
                best = (err, q, m)
    return best

def shape_variants(g):
    base = dict(GRIDS[g]); v = {"base": base}
    if g == "Jawa_Bali":
        v["p2b_flat_season"] = dict(base, p2b_flat=True)
        v["coal_cap_low"] = dict(base, coal=21250, ccgt_must=3750)    # ends of the feasible calibration range
        v["coal_cap_high"] = dict(base, coal=23500, ccgt_must=4500)
        v["analytic_bf_0.70"] = dict(base, p2b=False, peak=ANALYTIC_PEAK[g], base_frac=0.70)   # v9 shape, comparison
        return v
    v["calendar"] = dict(base, **CALENDAR)
    for bf in BF_SWEEP:                                               # Sumatera: sweep of the daily swing
        v[f"analytic_bf_{bf:.2f}"] = dict(base, base_frac=bf)
    return v

if not done("load_shape"):
    complete = True
    cal_rows = read_rows(T18C); have_cal = set(zip(cal_rows.grid, cal_rows.variant)) if len(cal_rows) else set()
    have = read_rows(T18); hk = set(zip(have.grid, have.variant, have.kind, have.key)) if len(have) else set()
    for g in SOBOL_GRIDS:
        for name, p in shape_variants(g).items():
            if name in ("base", "coal_cap_low", "coal_cap_high"):      # already calibrated (section 1)
                q, m = dict(p), energy_mix(p, GRID_SEED[g])
                err = max(abs(m[k] - CALIBRATION_TARGETS[g][k]) for k in CALIBRATION_TARGETS[g])
            else:
                err, q, m = recalibrate(g, p)
            lf, mm = load_stats(q, GRID_SEED[g])
            if (g, name) not in have_cal:
                append_row(T18C, dict(grid=g, variant=name, empirical_p2b=int(bool(q.get("p2b"))),
                                      base_frac=np.nan if q.get("p2b") else q["base_frac"], calendar=int("holidays" in q),
                                      load_factor_pct=round(lf, 2), daily_min_to_max=round(mm, 3),
                                      lf_system=OFFICIAL_LF[g], coal_MW=q["coal"], hydro_cf_mw=q["hydro_cf_mw"],
                                      geo=q["geo"], max_mix_err_pp=round(err, 2) if err == err else np.nan,
                                      **{f"sim_{k}": round(m[k], 2) for k in ["coal", "gas_oil", "ren"]}))
            if (g, name, "conseq", "0.02") not in hk:
                if not budget_ok("load_shape"): complete = False; break
                r = conseq_signals(g, with_reserve(q), 1.0, GRID_SEED[g], SIG2, CONSEQ_SHARE)
                append_row(T18, dict(grid=g, variant=name, kind="conseq", key="0.02", load_factor_pct=round(lf, 2),
                                     daily_min_to_max=round(mm, 3), solar_share_pct=round(r["solar_share_pct"], 3),
                                     saving_avgCI=round(r["avgCI"], 3), saving_MEF=round(r["MEF"], 3),
                                     unserved_any=r["unserved_any"]))
            if name in REGIME_VARIANTS[g]:
                for gf in [0.0, 1.0]:
                    for cm in RM_COALMIN:
                        for sc in RM_SOLAR:
                            key = f"{gf}|{cm}|{sc}"
                            if (g, name, "regime", key) in hk: continue
                            if not budget_ok("load_shape regime"): complete = False; break
                            r = conseq_signals(g, with_reserve(dict(q, coal_min=cm, gas_first=gf)), sc, SEED + 99,
                                               SIG2, CONSEQ_SHARE)
                            append_row(T18, dict(grid=g, variant=name, kind="regime", key=key, load_factor_pct=round(lf, 2),
                                                 daily_min_to_max=round(mm, 3), solar_share_pct=round(r["solar_share_pct"], 3),
                                                 saving_avgCI=round(r["avgCI"], 3), saving_MEF=round(r["MEF"], 3),
                                                 unserved_any=r["unserved_any"]))
                        if not complete: break
                    if not complete: break
            if not complete: break
        if not complete: break
    if complete:
        mark("load_shape")

if done("load_shape"):
    print(read_rows(T18C).to_string())
    R18 = read_rows(T18); print("unserved runs:", int(R18.unserved_any.sum()))
    print(R18[R18.kind == "conseq"][["grid", "variant", "load_factor_pct", "daily_min_to_max", "saving_avgCI", "saving_MEF"]].to_string())
    rg = R18[R18.kind == "regime"].copy()
    rg[["gas_first", "coal_min", "solar_scale"]] = rg.key.str.split("|", expand=True).astype(float)
    s18 = rg.assign(neg=rg.saving_avgCI < 0).groupby(["grid", "variant", "gas_first"]).agg(
        cells=("neg", "size"), avgCI_neg_cells=("neg", "sum"), MEF_min=("saving_MEF", "min"),
        MEF_neg_cells=("saving_MEF", lambda x: int((x < 0).sum()))).reset_index()
    s18.to_csv(f"{OUT}/table18_summary.csv", index=False); print(s18.to_string())
    for g, vs in REGIME_VARIANTS.items():
        for v in vs:
            pv = rg[(rg.grid == g) & (rg.variant == v) & (rg.gas_first == 0.0)].pivot(
                index="coal_min", columns="solar_share_pct", values="saving_avgCI")
            print(f"{g} coal-first, avg-CI saving, variant {v}:"); print(pv.round(1))
    sw = R18[(R18.kind == "conseq") & R18.variant.str.startswith("analytic") & (R18.grid == "Sumatera")].sort_values("daily_min_to_max")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(sw.daily_min_to_max, sw.saving_avgCI, "o-", label="Sumatera avg-CI (analytical shape sweep)")
    ax.plot(sw.daily_min_to_max, sw.saving_MEF, "s--", label="Sumatera MEF (analytical shape sweep)")
    for g, (lo, hi) in OBSERVED_SWING.items():
        ax.axvspan(lo, hi, color="grey", alpha=0.15, label=f"{g} observed 2024 (UIP2B)")
    emp = R18[(R18.kind == "conseq") & (R18.grid == "Jawa_Bali") & ~R18.variant.str.startswith("analytic")]
    ax.plot(emp.daily_min_to_max, emp.saving_avgCI, "k*", ms=9, ls="none", label="Jawa_Bali empirical shape variants, avg-CI")
    ax.plot(emp.daily_min_to_max, emp.saving_MEF, "kD", ms=6, mfc="none", ls="none", label="Jawa_Bali empirical shape variants, MEF")
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Daily minimum / daily peak load")
    ax.set_ylabel("Consequential saving at 2% load (%)"); ax.legend(fontsize=7); watermark(ax)
    save_fig(fig, "fig9_daily_swing.png")

# %% [md]
# ## 19. Mechanism of the harm valley (v9)
# Which unit class is marginal at each hour of the day, and where does the average intensity send the load?
# Jawa-Bali and Sumatera, coal-first, calibrated coal minimum, at three solar levels.

# %%
T19 = f"{OUT}/table19_mechanism.csv"
MECH_SOLAR = [1, 12, 45]
if not done("mechanism"):
    rows = []
    for g in SOBOL_GRIDS:
        p = with_reserve(GRIDS[g]); cmin = p["coal"] * p["coal_min"]
        for sc in MECH_SOLAR:
            tr = dispatch(p, np.random.default_rng(SEED + 99), sc)
            m = tr.mef.to_numpy() / 1000.0
            at_min = tr.coal.to_numpy() <= cmin + 1e-3
            cls = np.select([np.isclose(m, 0.0), np.isclose(m, EF["coal"]) & at_min, np.isclose(m, EF["coal"]),
                             np.isclose(m, EF["ccgt"]), np.isclose(m, EF["peaker"])],
                            ["curtailed_RE", "coal_at_min", "coal_above_min", "ccgt", "peaker"], "other")
            hod = np.arange(len(tr)) % 24
            aci = tr.avg_ci.to_numpy()
            low = aci <= np.quantile(aci, 0.25)                             # the hours an avg-CI scheduler favours
            share = 100 * float(tr.solar.sum() / tr.demand.sum())
            for h in range(24):
                sel = hod == h
                rows.append(dict(grid=g, solar_scale=sc, solar_share_pct=round(share, 3), hour=h,
                                 aci_mean=round(float(aci[sel].mean()), 1), mef_mean=round(float(tr.mef.to_numpy()[sel].mean()), 1),
                                 **{f"pct_{c}": round(100 * float((cls[sel] == c).mean()), 1) for c in
                                    ["curtailed_RE", "coal_at_min", "coal_above_min", "ccgt", "peaker"]}))
            rows.append(dict(grid=g, solar_scale=sc, solar_share_pct=round(share, 3), hour=-1,
                             aci_mean=round(float(aci[low].mean()), 1), mef_mean=round(float(tr.mef.to_numpy()[low].mean()), 1),
                             **{f"pct_{c}": round(100 * float((cls[low] == c).mean()), 1) for c in
                                ["curtailed_RE", "coal_at_min", "coal_above_min", "ccgt", "peaker"]}))
    pd.DataFrame(rows).to_csv(T19, index=False)
    mark("mechanism")

if done("mechanism"):
    t19 = read_rows(T19)
    print("hour = -1 rows: marginal class in the 25% lowest-ACI hours (where an avg-CI scheduler moves load)")
    print(t19[t19.hour == -1].T)
    cl = ["curtailed_RE", "coal_at_min", "coal_above_min", "ccgt", "peaker"]
    fig, axes = plt.subplots(len(SOBOL_GRIDS), len(MECH_SOLAR), figsize=(12, 3.3 * len(SOBOL_GRIDS)), squeeze=False)
    for i, g in enumerate(SOBOL_GRIDS):
        for j, sc in enumerate(MECH_SOLAR):
            d = t19[(t19.grid == g) & (t19.solar_scale == sc) & (t19.hour >= 0)].sort_values("hour"); ax = axes[i][j]
            ax.stackplot(d.hour, *[d[f"pct_{c}"] for c in cl], labels=cl, alpha=0.8)
            ax.set_ylim(0, 100); ax.set_xlabel("Hour of day"); ax.set_ylabel("Hours with this marginal class (%)")
            ax2 = ax.twinx(); ax2.plot(d.hour, d.aci_mean, "k-", lw=1.4, label="mean ACI")
            ax2.set_ylabel("ACI (gCO2/kWh)")
            ax.set_title(f"{g}, solar share {d.solar_share_pct.iloc[0]:.1f}%", fontsize=9); watermark(ax)
    axes[0][0].legend(fontsize=6, loc="lower left")
    save_fig(fig, "fig7_mechanism.png")

# %% [md]
# ## 20. Sensitivity to the jobs' slack distribution (v9)
# The base workload draws slack uniformly from {0, 2, 6, 12, 24} h. Two alternatives: shorter slack {0, 2, 6, 12} h
# and longer slack {12, 24, 48} h. Consequential test at 2 % load on the three calibrated grids.

# %%
T20 = f"{OUT}/table20_slack.csv"
SLACK_SETS = {"base_0-24h": None, "short_0-12h": [0, 2, 6, 12], "long_12-48h": [12, 24, 48]}
if not done("slack_sens"):
    have = read_rows(T20); hk = set(zip(have.variant, have.grid)) if len(have) else set()
    jobs_base = jobs; complete = True
    try:
        for name, opts in SLACK_SETS.items():
            if opts is None:
                jobs = jobs_base
            else:
                rr = np.random.default_rng(SEED + 2000)
                jobs = jobs_base.assign(slack=rr.choice(opts, len(jobs_base)))
            clear_cache()
            for g in ISLANDS:
                if (name, g) in hk: continue
                if not budget_ok("slack_sens"): complete = False; break
                r = conseq_signals(g, with_reserve(GRIDS[g]), 1.0, GRID_SEED[g], SIG2, CONSEQ_SHARE)
                append_row(T20, dict(variant=name, grid=g, mean_slack_h=round(float(jobs[jobs.home == g].slack.mean()), 2),
                                     saving_avgCI=round(r["avgCI"], 3), saving_MEF=round(r["MEF"], 3),
                                     unserved_any=r["unserved_any"]))
            if not complete: break
    finally:
        jobs = jobs_base; clear_cache()
    if complete:
        mark("slack_sens")
if done("slack_sens"):
    print(read_rows(T20))

# %% [md]
# ## 15. Provenance + automatic ZIP to Drive (only when every stage is finished)
# A SHA-256 checksum of every result file is stored in `run_metadata.json`.

# %%
STAGES = (["traces", "heuristic"] + (["sweep"] + [f"sobol_{g}" for g in SOBOL_GRIDS] if RUN_LEGACY else [])
          + ["consequential", "misspec"] + [f"sobolC_{g}" for g in SOBOL_GRIDS] + (["ppo"] if RUN_PPO else [])
          + ["consequential_reserve", "ablation", "regime_map", "calib_age", "sobol_convergence", "ef_sensitivity"]
          + ["rep_seeds", "load_shape", "mechanism", "slack_sens"])
pending = [st for st in STAGES if not done(st)]
if pending:
    print("Not finished yet:", pending, "-> Runtime > Run all again to resume (results are safe in Drive).")
else:
    meta = dict(seed=SEED, data_verified=DATA_VERIFIED, hours=HOURS, n_jobs=N_JOBS, ppo=RUN_PPO,
                ppo_timesteps=PPO_TIMESTEPS, grids=GRIDS, real_csv=REAL_CSV, ef=EF, ef_sets=EF_SETS,
                stages={st: json.load(open(f"{OUT}/.done_{st}")) for st in STAGES},
                zipped=time.strftime("%Y-%m-%d %H:%M:%S"))
    files = sorted(f for f in os.listdir(OUT) if f.endswith((".csv", ".png", ".pdf", ".npy")))
    meta["sha256"] = {f: hashlib.sha256(open(f"{OUT}/{f}", "rb").read()).hexdigest() for f in files}
    json.dump(meta, open(f"{OUT}/run_metadata.json", "w"), indent=1, default=str)
    stage_dir = f"/tmp/{RUN_TAG}_export"
    shutil.rmtree(stage_dir, ignore_errors=True); os.makedirs(stage_dir)
    for f in files + ["run_metadata.json"]:
        shutil.copy(f"{OUT}/{f}", stage_dir)
    zip_path = shutil.make_archive(f"{BASE}/{ZIP_TAG}_results_{time.strftime('%Y%m%d_%H%M')}", "zip", stage_dir)
    print("ZIP saved to Drive:", zip_path)
    print(json.dumps(meta["sha256"], indent=1))
print(f"Session time used: {(time.time() - START) / 60:.1f} min of {MAX_RUNTIME_MIN}")
