"""
Project 21: Non-premixed counterflow C2H4/air flame with H2O dilution + PAH formation.

Configuration:
- Fuel stream: C2H4 + H2O (mole fractions: 1-h2o_frac, h2o_frac)
- Oxidizer:    air (X_O2 = 0.21, X_N2 = 0.79)
- T_fuel = T_oxidizer = 300 K
- Pressures:    1, 10 bar
- H2O fractions in fuel: 0%, 4%, 8%, 12% (mole basis)
- 4 different residence times via varying fuel mass flux (oxidizer flux fixed)
- Total cases: 2 x 4 x 4 = 32

Strategy
--------
The CRECK 368-species/14228-reaction mechanism is too stiff to converge
counterflow flames "from cold" in a straightforward way. We use a
continuation strategy:

1. SEED: solve one flame at p=1 bar, x_H2O=0, m_fuel = baseline.
   This is the only "expensive" solve (~10-30 min).
2. RESTORE + INCREMENT: every subsequent flame restores the most-recently
   converged neighbouring solution (in P or x_H2O space) and re-solves with
   small parameter changes. Each continuation step is fast (<1 min).

Inner residence-time loop:
- Per (P, x_H2O) combo, sweep fuel mdot from high -> low (or low -> high)
  with restoration between steps.

Run time
--------
Approximately 2-5 hours total on a modern laptop, depending on convergence.
Memory: ~2 GB peak.
"""

import cantera as ct
import numpy as np
import pandas as pd
import os, time, json
from pathlib import Path

# ---- USER CONFIG ----
MECH_FILE      = "creck_c2h4_pah.yaml"
OUTPUT_DIR     = Path("results")
SEED_FILE      = OUTPUT_DIR / "seed_flame.yaml"
RESULTS_CSV    = OUTPUT_DIR / "all_results.csv"
PROFILES_DIR   = OUTPUT_DIR / "profiles"
SOLUTIONS_DIR  = OUTPUT_DIR / "solutions"

# Operating sweep
PRESSURES_BAR  = [1.0, 10.0]
H2O_FRACTIONS  = [0.00, 0.04, 0.08, 0.12]    # mole fraction in fuel stream
# Residence-time sweep: vary fuel mdot, keep oxidizer mdot constant.
# Lower mdot -> longer residence time. We pick 4 levels.
MDOT_OX        = 0.30                        # kg/m^2/s  (constant)
MDOT_FUEL_LIST = [0.06, 0.10, 0.14, 0.20]    # kg/m^2/s  (4 cases)

WIDTH          = 0.02     # 2 cm separation between fuel/oxidizer plates
T_INLET        = 300.0    # K  (project default when unspecified)

# Tracked PAH and key combustion species
TRACKED_SPECIES = [
    'C2H4','O2','N2','H2O',                      # reactants/diluents
    'CO','CO2','H2','OH','H','O',                # major
    'C2H2','CH4','C3H3','C4H2','C4H4',           # PAH precursors
    'C6H6','C6H5','C7H8',                        # benzene + toluene
    'C10H8','C10H7',                             # naphthalene
    'C12H8','C12H10',                            # acenaphthylene / biphenyl
    'C14H10',                                    # phenanthrene/anthracene
    'C16H10',                                    # pyrene (A4)
    'C18H10',                                    # heavier PAH if present
]

# ---- HELPERS ----
def setup_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PROFILES_DIR.mkdir(exist_ok=True)
    SOLUTIONS_DIR.mkdir(exist_ok=True)

def case_id(p_bar, h2o, mfuel):
    return f"P{p_bar:04.1f}bar_H2O{int(round(h2o*100)):02d}pct_mf{mfuel:.3f}"

def build_flame(gas, p_pa, h2o_frac, mdot_fuel, mdot_ox, T_in=T_INLET, width=WIDTH):
    """Construct a fresh CounterflowDiffusionFlame with given BCs."""
    f = ct.CounterflowDiffusionFlame(gas, width=width)
    f.P = p_pa
    f.fuel_inlet.mdot = mdot_fuel
    f.fuel_inlet.X = {'C2H4': 1.0 - h2o_frac, 'H2O': h2o_frac}
    f.fuel_inlet.T = T_in
    f.oxidizer_inlet.mdot = mdot_ox
    f.oxidizer_inlet.X = {'O2': 0.21, 'N2': 0.79}
    f.oxidizer_inlet.T = T_in
    f.transport_model = 'mixture-averaged'
    return f

# ---- SOLVERS ----
def solve_seed(gas, seed_path):
    """
    Solve the very first flame: p=1 bar, no H2O, baseline mdot.
    Use careful staged approach: loose -> tight refinement.
    Once solved, save to disk.
    """
    print(f"\n{'='*64}\nSEED FLAME  p=1 bar, x_H2O=0, mdot_f={MDOT_FUEL_LIST[1]:.3f}\n{'='*64}")
    p_pa = 1.0e5
    f = build_flame(gas, p_pa, 0.0, MDOT_FUEL_LIST[1], MDOT_OX)

    # Speed knobs (Cantera 3.2)
    f.set_max_jac_age(50, 50)
    f.flame.set_steady_tolerances(default=(1e-4, 1e-9))
    f.flame.set_transient_tolerances(default=(1e-4, 1e-9))

    t0 = time.time()
    # Stage 1: very loose, get something
    f.set_refine_criteria(ratio=10.0, slope=0.6, curve=0.8, prune=0.0)
    print("[seed] Stage 1: loose auto-solve...")
    f.solve(loglevel=1, auto=True)
    print(f"  -> Tmax = {f.T.max():.1f} K, pts = {f.flame.n_points}, t = {time.time()-t0:.1f}s")

    # Stage 2: tighten
    f.set_refine_criteria(ratio=4.0, slope=0.2, curve=0.4, prune=0.05)
    print("[seed] Stage 2: refine...")
    f.solve(loglevel=1, refine_grid=True)
    print(f"  -> Tmax = {f.T.max():.1f} K, pts = {f.flame.n_points}, t = {time.time()-t0:.1f}s")

    # Stage 3: final
    f.set_refine_criteria(ratio=3.0, slope=0.12, curve=0.25, prune=0.04)
    print("[seed] Stage 3: final refine...")
    f.solve(loglevel=1, refine_grid=True)
    print(f"  -> Tmax = {f.T.max():.1f} K, pts = {f.flame.n_points}, t = {time.time()-t0:.1f}s")

    f.save(str(seed_path), name="solution", overwrite=True)
    print(f"[seed] Saved to {seed_path}")
    return f

def solve_continuation(gas, p_pa, h2o_frac, mdot_fuel, restore_path, save_path,
                       loglevel=0):
    """Solve a flame by restoring a nearby converged solution."""
    f = build_flame(gas, p_pa, h2o_frac, mdot_fuel, MDOT_OX)
    f.restore(str(restore_path), name="solution")
    # Update BCs after restore
    f.P = p_pa
    f.fuel_inlet.mdot = mdot_fuel
    f.fuel_inlet.X = {'C2H4': 1.0 - h2o_frac, 'H2O': h2o_frac}
    f.fuel_inlet.T = T_INLET
    f.oxidizer_inlet.mdot = MDOT_OX
    f.oxidizer_inlet.X = {'O2': 0.21, 'N2': 0.79}
    f.oxidizer_inlet.T = T_INLET

    # Speed knobs (Cantera 3.2)
    f.set_max_jac_age(50, 50)        # reuse Jacobian aggressively — biggest free win
    f.flame.set_steady_tolerances(default=(1e-4, 1e-9))
    f.flame.set_transient_tolerances(default=(1e-4, 1e-9))

    f.set_refine_criteria(ratio=4.0, slope=0.15, curve=0.25, prune=0.05)
    f.solve(loglevel=loglevel, refine_grid=True)
    f.save(str(save_path), name="solution", overwrite=True)
    return f

# ---- POST-PROCESSING ----
def extract_metrics(f, p_bar, h2o_frac, mdot_fuel, mdot_ox):
    """Pull scalar metrics from a solved flame."""
    z = f.grid                   # axial coordinate, m
    T = f.T
    rho = f.density
    u = f.velocity               # axial velocity, m/s

    # Strain rate proxy: maximum |du/dz| on oxidizer side
    dudz = np.gradient(u, z)
    a_max = float(np.max(np.abs(dudz)))

    # Residence time estimate: domain transit time using mass-flux average
    # tau = integral (rho/|rho*u|) dz across reactive zone
    # Practical proxy: tau ~ width / (mdot_total / rho_avg)
    rho_avg = float(np.mean(rho))
    mdot_tot = mdot_fuel + mdot_ox
    tau = float(z[-1] - z[0]) * rho_avg / mdot_tot

    Tmax = float(T.max())
    z_Tmax = float(z[T.argmax()])

    row = {
        'p_bar': p_bar,
        'h2o_frac_fuel': h2o_frac,
        'mdot_fuel': mdot_fuel,
        'mdot_ox': mdot_ox,
        'mdot_ratio': mdot_ox / mdot_fuel,
        'strain_rate_max_1ps': a_max,
        'residence_time_s': tau,
        'T_max_K': Tmax,
        'z_Tmax_m': z_Tmax,
        'n_grid_points': int(f.flame.n_points),
    }
    # PAH peak mole fractions + locations
    for sp in TRACKED_SPECIES:
        if sp in f.gas.species_names:
            i = f.gas.species_index(sp)
            X = f.X[i, :]
            row[f'X_{sp}_peak'] = float(X.max())
            row[f'z_{sp}_peak_m'] = float(z[X.argmax()])
        else:
            row[f'X_{sp}_peak'] = np.nan
            row[f'z_{sp}_peak_m'] = np.nan
    return row

def save_profile_csv(f, csv_path):
    """Save full axial profiles for one case to CSV."""
    z = f.grid
    data = {'z_m': z, 'T_K': f.T, 'u_m_per_s': f.velocity, 'rho_kg_per_m3': f.density}
    for sp in TRACKED_SPECIES:
        if sp in f.gas.species_names:
            i = f.gas.species_index(sp)
            data[f'X_{sp}'] = f.X[i, :]
    pd.DataFrame(data).to_csv(csv_path, index=False)

# ---- MAIN SWEEP ----
def main():
    setup_dirs()
    print(f"Loading mechanism: {MECH_FILE}")
    t0 = time.time()
    gas = ct.Solution(MECH_FILE)
    print(f"  -> {gas.n_species} species, {gas.n_reactions} reactions, "
          f"loaded in {time.time()-t0:.1f}s\n")

    # --- 1. Seed ---
    if not SEED_FILE.exists():
        solve_seed(gas, SEED_FILE)
    else:
        print(f"[seed] Reusing existing {SEED_FILE}")

    # --- 2. Sweep ---
    rows = []
    # Build a careful continuation order:
    #   for each pressure (1 -> 10 bar):
    #     for each H2O frac (0 -> 12%):
    #       for each mdot (start from baseline, sweep both ways):
    # Always restore from the most-recent successful solve.

    last_solution = SEED_FILE
    total_cases = len(PRESSURES_BAR) * len(H2O_FRACTIONS) * len(MDOT_FUEL_LIST)
    case_n = 0

    # Order: ramp pressure, then H2O, then mdot
    for p_bar in PRESSURES_BAR:
        p_pa = p_bar * 1e5
        for h2o_frac in H2O_FRACTIONS:
            for mdot_f in MDOT_FUEL_LIST:
                case_n += 1
                cid = case_id(p_bar, h2o_frac, mdot_f)
                save_path = SOLUTIONS_DIR / f"{cid}.yaml"
                csv_path  = PROFILES_DIR  / f"{cid}.csv"

                if save_path.exists():
                    print(f"[{case_n}/{total_cases}] SKIP (already done): {cid}")
                    last_solution = save_path
                    # Reload for metrics
                    f = build_flame(gas, p_pa, h2o_frac, mdot_f, MDOT_OX)
                    f.restore(str(save_path), name="solution")
                else:
                    print(f"[{case_n}/{total_cases}] {cid}", flush=True)
                    t1 = time.time()
                    try:
                        f = solve_continuation(
                            gas, p_pa, h2o_frac, mdot_f,
                            restore_path=last_solution,
                            save_path=save_path,
                            loglevel=0
                        )
                        last_solution = save_path
                        print(f"   OK Tmax={f.T.max():.1f}K  "
                              f"pts={f.flame.n_points}  "
                              f"dt={time.time()-t1:.1f}s", flush=True)
                    except Exception as e:
                        print(f"   FAIL: {type(e).__name__}: {e}")
                        # Try harder: restart from seed
                        print("   Retrying from seed...")
                        try:
                            f = solve_continuation(
                                gas, p_pa, h2o_frac, mdot_f,
                                restore_path=SEED_FILE,
                                save_path=save_path, loglevel=0)
                            last_solution = save_path
                            print(f"   OK on retry. Tmax={f.T.max():.1f}K")
                        except Exception as e2:
                            print(f"   FAIL on retry: {e2}")
                            continue

                # Record metrics + profile
                row = extract_metrics(f, p_bar, h2o_frac, mdot_f, MDOT_OX)
                row['case_id'] = cid
                rows.append(row)
                save_profile_csv(f, csv_path)

                # Save running CSV
                pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    print(f"\nAll done. Results in {RESULTS_CSV}, profiles in {PROFILES_DIR}")

if __name__ == "__main__":
    main()
