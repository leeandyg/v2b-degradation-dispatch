"""
V2B + PV Dispatch Optimization (Pyomo + Gurobi)

Mode B: V2B without degradation penalty (w = 0)
       Set w > 0 for Mode C (degradation-aware dispatch)
"""

import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import rainflow

# ============================================================
# SYSTEM PARAMETERS
# ============================================================

E_BATT = 60.0            # EV battery capacity (kWh)
P_CHARGER = 9.6          # Bidirectional charger max power (kW)
ETA = 0.9487             # One-way efficiency (sqrt(0.90) round-trip)
SOC_MIN = 0.20           # Minimum SOC
SOC_MAX = 0.90           # Maximum SOC
SOC_INIT = 0.50          # Initial SOC at midnight
SOC_TERMINAL = 0.50      # Minimum SOC at end of day (keeps comparison fair across modes)
SOC_DEPART_MIN = 0.50    # Minimum SOC at departure (8 AM)
DRIVING_KWH = 10.0       # Energy consumed during daily commute (kWh)
DEPART_HOUR = 8          # EV departs at 8 AM
RETURN_HOUR = 18         # EV returns at 6 PM
NEM_EXPORT_RATE = 0.05   # Cost approximation (avoid NEM 3.0: ~$0.05/kWh flat)

# --- Degradation penalty weight ($/kWh of battery throughput) ---
# Mode B: W = 0.0  (no penalty -- unconstrained V2B)
# Mode C: W > 0.0  (e.g., 0.03 -- degradation-aware dispatch)
W_DEGRADATION = 0.0       # <-- CHANGE THIS FOR MODE C


# ============================================================
# DATA LOADING
# ============================================================

def load_pv_data(filepath, month, day):
    """
    Load hourly PV generation (kW) for a specific day from PVWatts CSV.
    PVWatts file has 31 metadata lines, then a header + 8760 hourly rows.
    Uses 'AC System Output (W)' column, converted to kW.
    """
    df = pd.read_csv(filepath, skiprows=31)
    df.columns = [
        "Month", "Day", "Hour",
        "Beam_Irradiance", "Diffuse_Irradiance",
        "Ambient_Temp", "Wind_Speed", "Albedo",
        "POA_Irradiance", "Cell_Temp",
        "DC_Output_W", "AC_Output_W",
    ]
    day_data = df[(df["Month"] == month) & (df["Day"] == day)]
    pv_kw = day_data["AC_Output_W"].values / 1000.0  # W -> kW
    return pv_kw  # length-24 array


def load_load_data(filepath, month, day, df=None):
    """
    Load residential load profile for a specific day.

    Source: ResStock up00-ca-single-family_detached.csv (trimmed to the columns
    used here -> Data/resstock_ca_sfd_load.csv)
    - Timestamps have no timezone offset -- parse directly, no UTC conversion.
    - Scales to a single-home average using the 'units_represented' column
      (number of CA single-family households the sample represents).
    - Subtracts EV charging from total electricity to avoid double-counting,
      since EV charging is a decision variable in our model.
    - Aggregates 15-min intervals to hourly.

    If df is provided, skips reading the CSV (use for repeated calls).
    """
    # Column names -- defined here so they're easy to audit
    TOTAL_COL = "out.electricity.total.energy_consumption..kwh"
    EV_COL    = "out.electricity.ev_charging.energy_consumption..kwh"
    UNITS_COL = "units_represented"
    TS_COL    = "timestamp"

    if df is None:
        df = pd.read_csv(filepath)

    # Verify expected columns exist (catches wrong file or renamed columns)
    for col in [TS_COL, TOTAL_COL, EV_COL, UNITS_COL]:
        if col not in df.columns:
            raise KeyError(f"Expected column not found in load CSV: '{col}'")



    # Parse timestamps without UTC conversion (no timezone offset in this file)
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    target_date = f"2018-{month:02d}-{day:02d}"
    day_data = df[df[TS_COL].dt.strftime("%Y-%m-%d") == target_date].copy()

    if len(day_data) != 96:
        raise ValueError(
            f"Expected 96 intervals for {target_date}, got {len(day_data)}. "
            "Check that the file covers this date."
        )

    # Per-home load = (aggregate total - aggregate EV) / units_represented
    units = day_data[UNITS_COL].values
    home_load_kwh = (day_data[TOTAL_COL].values - day_data[EV_COL].values) / units

    # Aggregate 15-min -> hourly (sum groups of 4; result is kWh/h = avg kW)
    hourly = home_load_kwh.reshape(24, 4).sum(axis=1)

    return hourly  # length-24 array, units = kWh per hour


def get_tou_prices(season):
    """
    SCE TOU-D-4-9PM (Schedule effective Jan 2026, before baseline credit).

    Summer (Jun-Sep), weekdays:
        On-Peak   4-9 PM  (hours 16-20):  $0.58/kWh
        Off-Peak  all other hours:        $0.34/kWh

    Winter (Oct-May), weekdays — 3 tiers:
        Mid-Peak       4-9 PM   (hours 16-20):       $0.51/kWh
        Off-Peak       9PM-8AM  (hours 21-23, 0-7):  $0.37/kWh
        Super Off-Peak 8AM-4PM  (hours 8-15):        $0.33/kWh
    """
    prices = np.zeros(24)
    for h in range(24):
        if season == "summer":
            if 16 <= h < 21:
                prices[h] = 0.58   # on-peak
            else:
                prices[h] = 0.34   # off-peak
        else:  # winter
            if 16 <= h < 21:
                prices[h] = 0.51   # mid-peak
            elif 8 <= h < 16:
                prices[h] = 0.33   # super off-peak
            else:
                prices[h] = 0.37   # off-peak (overnight)
    return prices


# ============================================================
# OPTIMIZATION MODEL
# ============================================================

def build_model(pv, load, buy_prices, sell_prices, w=0.0, mode="B"):
    """
    Build the V2B MILP model in Pyomo.

    mode="A": Smart charging only (P_dis forced to 0, w ignored).
    mode="B": V2B, no degradation penalty (w=0).
    mode="C": V2B with degradation penalty (w>0).
    """
    T = range(24)
    away_hours = set(range(DEPART_HOUR, RETURN_HOUR))  # {8, 9, ..., 17}

    m = pyo.ConcreteModel("V2B")

    # --- Sets ---
    m.T = pyo.Set(initialize=T)
    m.T_soc = pyo.Set(initialize=range(25))  # SOC at start of each hour + end of day

    # --- Parameters ---
    m.pv   = pyo.Param(m.T, initialize=dict(enumerate(pv)))
    m.demand = pyo.Param(m.T, initialize=dict(enumerate(load)))
    m.pi_buy  = pyo.Param(m.T, initialize=dict(enumerate(buy_prices)))
    m.pi_sell = pyo.Param(m.T, initialize=dict(enumerate(sell_prices)))

    # --- Decision Variables ---
    m.P_ch   = pyo.Var(m.T, within=pyo.NonNegativeReals)  # charge power (kW)
    m.P_dis  = pyo.Var(m.T, within=pyo.NonNegativeReals)  # discharge power (kW)
    m.P_buy  = pyo.Var(m.T, within=pyo.NonNegativeReals)  # grid import (kW)
    m.P_sell = pyo.Var(m.T, within=pyo.NonNegativeReals)  # grid export (kW)
    m.SOC    = pyo.Var(m.T_soc, bounds=(SOC_MIN, SOC_MAX))
    m.delta  = pyo.Var(m.T, within=pyo.Binary)            # 1=charging, 0=discharging

    # ---- OBJECTIVE ----
    # Grid cost  -  export credit  +  degradation penalty
    def obj_rule(m):
        return sum(
            m.pi_buy[t] * m.P_buy[t]
            - m.pi_sell[t] * m.P_sell[t]
            + w * (m.P_ch[t] + m.P_dis[t])   # <-- degradation penalty term
            for t in m.T
        )
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ---- CONSTRAINTS ----

    # 1. Power balance (every hour)
    #    PV + discharge + grid_buy = load + charge + grid_sell
    def power_balance_rule(m, t):
        return (m.pv[t] + m.P_dis[t] + m.P_buy[t]
                == m.demand[t] + m.P_ch[t] + m.P_sell[t])
    m.c_power_bal = pyo.Constraint(m.T, rule=power_balance_rule)

    # 2. SOC tracking (home hours only)
    #    SOC[t+1] = SOC[t] + (eta * P_ch[t] - P_dis[t] / eta) / E_batt
    def soc_tracking_rule(m, t):
        # Skip away hours and the return-hour transition
        if t in away_hours:
            return pyo.Constraint.Skip
        return (m.SOC[t + 1] == m.SOC[t]
                + (ETA * m.P_ch[t] - m.P_dis[t] / ETA) / E_BATT)
    m.c_soc_track = pyo.Constraint(m.T, rule=soc_tracking_rule)

    # 3. Initial SOC
    m.c_soc_init = pyo.Constraint(expr=m.SOC[0] == SOC_INIT)

    # 4. Departure SOC >= 50%
    m.c_soc_depart = pyo.Constraint(expr=m.SOC[DEPART_HOUR] >= SOC_DEPART_MIN)

    # 5. Driving consumption: SOC at return = SOC at departure - driving energy
    m.c_driving = pyo.Constraint(
        expr=m.SOC[RETURN_HOUR] == m.SOC[DEPART_HOUR] - DRIVING_KWH / E_BATT
    )

    # 5b. Terminal SOC: battery must end the day >= SOC_TERMINAL
    # Prevents the optimizer from freely draining the battery at end-of-day,
    # which would make Mode B look artificially better than Mode C.
    m.c_soc_terminal = pyo.Constraint(expr=m.SOC[24] >= SOC_TERMINAL)

    # 6. Charger power limits + mutual exclusion (no simultaneous charge & discharge)
    def charge_limit_rule(m, t):
        return m.P_ch[t] <= P_CHARGER * m.delta[t]
    m.c_ch_limit = pyo.Constraint(m.T, rule=charge_limit_rule)

    def discharge_limit_rule(m, t):
        return m.P_dis[t] <= P_CHARGER * (1 - m.delta[t])
    m.c_dis_limit = pyo.Constraint(m.T, rule=discharge_limit_rule)

    # 7. EV unavailable while away -- no charging or discharging
    def away_charge_rule(m, t):
        if t in away_hours:
            return m.P_ch[t] == 0
        return pyo.Constraint.Skip
    m.c_away_ch = pyo.Constraint(m.T, rule=away_charge_rule)

    def away_discharge_rule(m, t):
        if t in away_hours:
            return m.P_dis[t] == 0
        return pyo.Constraint.Skip
    m.c_away_dis = pyo.Constraint(m.T, rule=away_discharge_rule)

    # 8. Mode A: smart charging only -- no V2B discharge at any hour
    if mode == "A":
        def no_discharge_rule(m, t):
            return m.P_dis[t] == 0
        m.c_no_dis = pyo.Constraint(m.T, rule=no_discharge_rule)

    return m


# ============================================================
# SOLVE & EXTRACT
# ============================================================

def solve_model(model, solver_name="gurobi"):
    """Solve the MILP with Gurobi (or specify another solver)."""
    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(model, tee=False)

    status = result.solver.termination_condition
    if status != pyo.TerminationCondition.optimal:
        print(f"WARNING: Solver terminated with status: {status}")

    return result


def _extract_soc(model):
    """Extract SOC values, linearly interpolating away hours (9..17)."""
    away = set(range(DEPART_HOUR + 1, RETURN_HOUR))  # {9, 10, ..., 17}
    soc = []
    for t in range(25):
        if t in away:
            soc.append(None)  # placeholder -- will interpolate below
        else:
            soc.append(pyo.value(model.SOC[t]))
    # Interpolate away-hour gaps between SOC[DEPART_HOUR] and SOC[RETURN_HOUR]
    soc_depart = soc[DEPART_HOUR]
    soc_return = soc[RETURN_HOUR]
    for t in away:
        frac = (t - DEPART_HOUR) / (RETURN_HOUR - DEPART_HOUR)
        soc[t] = soc_depart + frac * (soc_return - soc_depart)
    return soc


def extract_results(model):
    """Pull decision-variable values into a plain dict for analysis."""
    T = list(model.T)
    res = {
        "P_ch":   [pyo.value(model.P_ch[t])   for t in T],
        "P_dis":  [pyo.value(model.P_dis[t])   for t in T],
        "P_buy":  [pyo.value(model.P_buy[t])   for t in T],
        "P_sell": [pyo.value(model.P_sell[t])   for t in T],
        "SOC":    _extract_soc(model),
        "PV":     [pyo.value(model.pv[t])       for t in T],
        "Load":   [pyo.value(model.demand[t])   for t in T],
        "cost":   pyo.value(model.obj),
    }
    return res


# ============================================================
# RAINFLOW DEGRADATION ANALYSIS
# ============================================================

def rainflow_analysis(soc_profile):
    """
    Run rainflow cycle counting on the SOC profile (home hours only).
    Returns degradation index using power-law stress: sum(n_i * DoD_i^2).
    """
    cycles = rainflow.count_cycles(soc_profile)
    # Each entry: (range, count)  -- range = full cycle depth
    degradation_index = sum(count * (dod ** 2.0) for dod, count in cycles)
    total_cycles = sum(count for _, count in cycles)
    return {
        "cycles": cycles,
        "degradation_index": degradation_index,
        "total_cycles": total_cycles,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_dispatch(results, title="V2B Dispatch", out_dir="Results"):
    """Dispatch schedule and SOC trajectory."""
    hours = np.arange(24)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # -- Top panel: power flows --
    pv  = np.array(results["PV"])
    buy = np.array(results["P_buy"])
    dis = np.array(results["P_dis"])
    ch  = np.array(results["P_ch"])
    sell = np.array(results["P_sell"])
    load = np.array(results["Load"])

    ax1.bar(hours, pv,  label="PV Generation",    color="gold",   alpha=0.8)
    ax1.bar(hours, buy, bottom=pv, label="Grid Import", color="red", alpha=0.5)
    ax1.bar(hours, dis, bottom=pv + buy, label="EV Discharge (V2B)", color="green", alpha=0.7)
    ax1.bar(hours, -ch,  label="EV Charging",  color="blue",  alpha=0.5)
    ax1.bar(hours, -sell, bottom=-ch, label="Grid Export", color="purple", alpha=0.5)
    ax1.plot(hours, load, "k-o", label="Home Load", linewidth=2, markersize=4)
    ax1.axhline(y=0, color="gray", linewidth=0.5)
    ax1.axvspan(DEPART_HOUR, RETURN_HOUR, alpha=0.08, color="gray")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8)

    # -- Bottom panel: SOC --
    soc = np.array(results["SOC"]) * 100  # percent
    soc_hours = np.arange(25)

    # Fill in away-period SOC for visualization (linear interp from depart to return)
    soc_plot = soc.copy()
    soc_depart = soc[DEPART_HOUR]
    soc_return = soc[RETURN_HOUR]
    for t in range(DEPART_HOUR + 1, RETURN_HOUR):
        frac = (t - DEPART_HOUR) / (RETURN_HOUR - DEPART_HOUR)
        soc_plot[t] = soc_depart + frac * (soc_return - soc_depart)

    ax2.plot(soc_hours, soc_plot, "b-o", linewidth=2, markersize=4)
    ax2.axhline(y=SOC_MIN * 100, color="r", linestyle="--", alpha=0.5, label=f"SOC bounds ({SOC_MIN*100:.0f}%-{SOC_MAX*100:.0f}%)")
    ax2.axhline(y=SOC_MAX * 100, color="r", linestyle="--", alpha=0.5)
    ax2.axhline(y=SOC_DEPART_MIN * 100, color="orange", linestyle="--", alpha=0.5, label=f"Departure min ({SOC_DEPART_MIN*100:.0f}%)")
    ax2.axvspan(DEPART_HOUR, RETURN_HOUR, alpha=0.08, color="gray", label="EV Away")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("SOC (%)")
    ax2.set_title("EV Battery State of Charge")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_tradeoff_curve(csv_rows, season, out_dir):
    """Plot cost vs. degradation tradeoff for a single season's w-sweep."""
    # Separate Mode A from the w-sweep (Modes B/C)
    sweep_rows = [r for r in csv_rows if r["season"] == season and r["mode_type"] != "A"]
    mode_a = [r for r in csv_rows if r["season"] == season and r["mode_type"] == "A"]
    sweep_rows.sort(key=lambda r: r["w"])

    w_vals = [r["w"] for r in sweep_rows]
    costs  = [r["cost_actual"] for r in sweep_rows]
    degs   = [r["rainflow_degradation"] for r in sweep_rows]

    fig, ax = plt.subplots(figsize=(8, 5))

    # W-sweep curve
    ax.plot(degs, costs, "o-", color="tab:blue", linewidth=2, markersize=7,
            label="V2B w-sweep (Modes B/C)")

    # Annotate each point with its w-value
    for w, deg, cost in zip(w_vals, degs, costs):
        label = "w=0" if w == 0 else f"{w:.2f}"
        ax.annotate(label, (deg, cost), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="gray")

    # Mode A reference point
    if mode_a:
        a = mode_a[0]
        ax.plot(a["rainflow_degradation"], a["cost_actual"], "*",
                color="red", markersize=15, zorder=5,
                label="Mode A (smart charge, no V2B)")

    ax.set_xlabel("Rainflow Degradation Index")
    ax.set_ylabel("Daily Electricity Cost ($)")
    ax.set_title(f"Cost vs. Degradation Tradeoff -- {season.title()}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"tradeoff_{season}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATA_DIR = "Data"
    RESULTS_DIR = os.path.join("Results", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    LOAD_FILE = f"{DATA_DIR}/resstock_ca_sfd_load.csv"
    PV_FILE   = f"{DATA_DIR}/pvwatts_pasadena_hourly.csv"

    os.makedirs(RESULTS_DIR)

    # Degradation penalty sweep (finer near transition zones ~0.05-0.10)
    W_VALUES = [0, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095,
                0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]
    SEASONS  = [("summer", 7, 15), ("winter", 1, 15)]

    # Build scenario list: Mode A (1 per season) + Mode B/C w-sweep
    # Each entry: (season, month, day, w, mode)
    SCENARIOS = []
    for season, month, day in SEASONS:
        SCENARIOS.append((season, month, day, 0.0, "A"))  # Mode A: smart charge
        for w in W_VALUES:
            mode = "B" if w == 0 else "C"
            SCENARIOS.append((season, month, day, w, mode))
    n_total = len(SCENARIOS)

    # Save dispatch PNGs for these w-values (plus Mode A always)
    DISPATCH_W = {0, 0.10}

    # Pre-load the large load CSV once
    print("Loading residential load CSV...")
    load_df = pd.read_csv(LOAD_FILE)

    csv_rows = []
    txt_path = os.path.join(RESULTS_DIR, "results.txt")
    csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")

    with open(txt_path, "w") as txt:

        for i, (season, month, day, w, mode) in enumerate(SCENARIOS):
            if mode == "A":
                mode_str = "ModeA_smart_charge"
            elif w == 0:
                mode_str = "ModeB_w0"
            else:
                mode_str = f"ModeC_w{w:.3f}"
            label = f"{season.title()} -- {mode_str}"

            # ---- Load input data ----
            pv_kw     = load_pv_data(PV_FILE, month, day)
            load_kw   = load_load_data(LOAD_FILE, month, day, df=load_df)
            buy_price = get_tou_prices(season)
            sell_price = np.full(24, NEM_EXPORT_RATE)

            # ---- Build & solve ----
            model = build_model(pv_kw, load_kw, buy_price, sell_price, w=w, mode=mode)
            solve_model(model)
            results = extract_results(model)

            # ---- Compute metrics ----
            pv_total  = sum(results["PV"])
            grid_buy  = sum(results["P_buy"])
            grid_sell = sum(results["P_sell"])
            v2b_dis   = sum(results["P_dis"])
            ev_charge = sum(results["P_ch"])
            self_con  = ((pv_total - grid_sell) / pv_total * 100) if pv_total > 0 else 0

            actual_cost = (
                sum(buy_price[t] * results["P_buy"][t] for t in range(24))
                - sum(sell_price[t] * results["P_sell"][t] for t in range(24))
            )

            soc_home = results["SOC"][:DEPART_HOUR + 1] + results["SOC"][RETURN_HOUR:]
            deg = rainflow_analysis(soc_home)

            # ---- Write detailed results to txt ----
            block = (
                f"{'='*55}\n"
                f"  V2B Results -- {label}\n"
                f"{'='*55}\n"
                f"  PV total:                {pv_total:.2f} kWh\n"
                f"  Load total:              {sum(results['Load']):.2f} kWh\n"
                f"  Peak TOU:                ${buy_price.max():.5f}/kWh\n"
                f"  Daily electricity cost:  ${actual_cost:.2f}\n"
            )
            if w > 0:
                block += f"  Objective (incl. penalty): ${results['cost']:.2f}  [w={w}]\n"
            block += (
                f"  Grid import:             {grid_buy:.2f} kWh\n"
                f"  Grid export:             {grid_sell:.2f} kWh\n"
                f"  V2B discharged:          {v2b_dis:.2f} kWh\n"
                f"  EV charging:             {ev_charge:.2f} kWh\n"
                f"  PV self-consumption:     {self_con:.1f}%\n"
                f"  Rainflow degradation:    {deg['degradation_index']:.6f}\n"
                f"  Equivalent cycles:       {deg['total_cycles']:.2f}\n"
                f"\n"
            )
            txt.write(block)

            # ---- Collect CSV row ----
            csv_rows.append({
                "season": season,
                "mode_type": mode,
                "mode": mode_str,
                "w": w,
                "cost_actual": round(actual_cost, 4),
                "grid_import_kwh": round(grid_buy, 2),
                "grid_export_kwh": round(grid_sell, 2),
                "v2b_discharged_kwh": round(v2b_dis, 2),
                "ev_charging_kwh": round(ev_charge, 2),
                "pv_total_kwh": round(pv_total, 2),
                "pv_self_consumption_pct": round(self_con, 1),
                "rainflow_degradation": round(deg["degradation_index"], 6),
                "equiv_cycles": round(deg["total_cycles"], 2),
            })

            # ---- Dispatch plot for Mode A, w=0, and key w-values ----
            extra = ""
            if mode == "A" or w in DISPATCH_W:
                png_path = plot_dispatch(results, title=f"V2B {label}", out_dir=RESULTS_DIR)
                extra = f"  -> {png_path}"

            # ---- Terminal: one-liner per scenario ----
            print(f"  [{i+1:2d}/{n_total}] {label:30s}  cost=${actual_cost:.2f}{extra}")

    # ---- Write CSV ----
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---- Tradeoff curves (one per season) ----
    for season, _, _ in SEASONS:
        tc_path = plot_tradeoff_curve(csv_rows, season, RESULTS_DIR)
        print(f"  Tradeoff curve: {tc_path}")

    print(f"\nResults saved to:")
    print(f"  {txt_path}")
    print(f"  {csv_path}")
    print(f"  {RESULTS_DIR}/")
