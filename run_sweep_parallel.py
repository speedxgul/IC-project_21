"""
Parallel sweep driver for M2 Pro (or any multicore machine).

Strategy
--------
The 32 cases are organized as 8 'chains' of 4 cases each:
  chain = (pressure, h2o_frac); within a chain, sweep mdot_fuel.
Within each chain, mdot continuation MUST be serial (each step restores
from the prior). Across chains, everything is independent → run in parallel.

The seed flame (1 bar, 0% H2O) must finish first because all chains
start from it. After that, fire all 8 chains across the worker pool.

M2 Pro has 6 perf + 4 efficiency cores. 4-6 parallel workers is the sweet spot
(memory pressure becomes a problem above 6 because each worker holds the
mechanism in memory ≈ 200-300 MB).

Usage
-----
    python run_sweep_parallel.py --workers 4

The seed flame is solved serially first (so all chains start from it).
Then chains run concurrently. Each chain writes its own results — they
get aggregated at the end.

Idempotent: completed cases are skipped. Crash → just rerun.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import time
import pandas as pd

# Reuse functions from the original run_sweep
from run_sweep import (
    MECH_FILE, OUTPUT_DIR, SEED_FILE, RESULTS_CSV,
    PROFILES_DIR, SOLUTIONS_DIR,
    PRESSURES_BAR, H2O_FRACTIONS, MDOT_FUEL_LIST, MDOT_OX,
    T_INLET, WIDTH,
    setup_dirs, case_id, build_flame,
    solve_seed, solve_continuation,
    extract_metrics, save_profile_csv,
)
import cantera as ct


def run_chain(args):
    """
    Run one chain: fixed (p_bar, h2o_frac), sweep all mdot_fuel.
    Returns list of result rows.
    
    Each worker loads its own copy of the mechanism (Cantera objects don't
    pickle / share between processes safely).
    """
    p_bar, h2o_frac = args
    p_pa = p_bar * 1e5
    
    print(f"[chain p={p_bar} H2O={h2o_frac*100:.0f}%] starting", flush=True)
    t0 = time.time()
    gas = ct.Solution(MECH_FILE)
    print(f"[chain p={p_bar} H2O={h2o_frac*100:.0f}%] mech loaded in {time.time()-t0:.1f}s", flush=True)
    
    rows = []
    last_solution = SEED_FILE
    
    # Sort mdot from middle outward (closest to seed first)
    # If seed was at MDOT_FUEL_LIST[1]=0.10, do that first, then ramp out
    seed_mdot = MDOT_FUEL_LIST[1]
    sorted_mdots = sorted(MDOT_FUEL_LIST, key=lambda m: abs(m - seed_mdot))
    
    for mdot_f in sorted_mdots:
        cid = case_id(p_bar, h2o_frac, mdot_f)
        save_path = SOLUTIONS_DIR / f"{cid}.yaml"
        csv_path = PROFILES_DIR / f"{cid}.csv"
        
        if save_path.exists():
            print(f"[chain p={p_bar} H2O={h2o_frac*100:.0f}%] SKIP {cid}", flush=True)
            f = build_flame(gas, p_pa, h2o_frac, mdot_f, MDOT_OX)
            f.restore(str(save_path), name="solution")
            last_solution = save_path
        else:
            t1 = time.time()
            try:
                f = solve_continuation(
                    gas, p_pa, h2o_frac, mdot_f,
                    restore_path=last_solution,
                    save_path=save_path,
                    loglevel=0,
                )
                last_solution = save_path
                print(f"[chain p={p_bar} H2O={h2o_frac*100:.0f}%] OK {cid} "
                      f"Tmax={f.T.max():.0f}K dt={time.time()-t1:.0f}s", flush=True)
            except Exception as e:
                print(f"[chain p={p_bar} H2O={h2o_frac*100:.0f}%] FAIL {cid}: {e}", flush=True)
                # Retry from seed
                try:
                    f = solve_continuation(
                        gas, p_pa, h2o_frac, mdot_f,
                        restore_path=SEED_FILE,
                        save_path=save_path, loglevel=0)
                    last_solution = save_path
                    print(f"[chain p={p_bar} H2O={h2o_frac*100:.0f}%] retry OK {cid}", flush=True)
                except Exception as e2:
                    print(f"[chain p={p_bar} H2O={h2o_frac*100:.0f}%] retry FAIL {cid}: {e2}", flush=True)
                    continue
        
        row = extract_metrics(f, p_bar, h2o_frac, mdot_f, MDOT_OX)
        row['case_id'] = cid
        rows.append(row)
        save_profile_csv(f, csv_path)
    
    print(f"[chain p={p_bar} H2O={h2o_frac*100:.0f}%] DONE in {time.time()-t0:.0f}s "
          f"({len(rows)} cases)", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel chain workers (default 4)")
    args = parser.parse_args()
    
    setup_dirs()
    
    # Step 1: serial seed solve
    if not SEED_FILE.exists():
        print("Solving seed flame (serial)...")
        t0 = time.time()
        gas = ct.Solution(MECH_FILE)
        solve_seed(gas, SEED_FILE)
        print(f"Seed done in {(time.time()-t0)/60:.1f} min")
    else:
        print(f"Seed already exists at {SEED_FILE}")
    
    # Step 2: parallel chains
    chains = [(p, h) for p in PRESSURES_BAR for h in H2O_FRACTIONS]
    print(f"\nLaunching {len(chains)} chains across {args.workers} workers...")
    print(f"Chains: {chains}")
    
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        all_rows = pool.map(run_chain, chains)
    
    # Flatten + write final CSV
    flat = [row for chain_rows in all_rows for row in chain_rows]
    df = pd.DataFrame(flat).sort_values(['p_bar', 'h2o_frac_fuel', 'mdot_fuel'])
    df.to_csv(RESULTS_CSV, index=False)
    
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min")
    print(f"  {len(flat)} cases written to {RESULTS_CSV}")
    print(f"  Now run: python build_report.py")


if __name__ == "__main__":
    # macOS requires 'spawn' for multiprocessing with C-extension libs (Cantera)
    mp.set_start_method('spawn', force=True)
    main()
