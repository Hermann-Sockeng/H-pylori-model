import numpy as np
from pathlib import Path
from src.parameters import get_scenario_parameters
from src.model import setup_model
from src.simulation import run_scenario
from src.plotting import plot_scenario_dynamics, plot_ph_landscape
from src.stability import check_e2_stability, check_e3_stability

def main():
    output_dir = Path("BMB_Figures")
    output_dir.mkdir(exist_ok=True)

    print("H. pylori Infection Model Considering pH - Numerical Solution")
    print("="*60)

    scenario_results = {}
    model_funcs_list = []

    for sc in [1, 2, 3, 4]:
        print(f"\nRunning Scenario {sc}...")
        params = get_scenario_parameters(sc)
        mf = setup_model(params)
        model_funcs_list.append(mf)
        res = run_scenario(sc, mf)
        scenario_results[sc] = res
        
        # Stability analysis (for appropriate scenarios)
        if sc == 2:  # E₂-like equilibrium
            y_final = [res['s'][-1], res['r'][-1], res['g'][-1],
                       res['d1'][-1], res['d2'][-1], res['d3'][-1],
                       (res['H'][-1] - mf['H_0']) / mf['scale_H']]  # back to h
            stab_e2 = check_e2_stability(y_final, mf['model'])
            print("\nStability analysis for E₂ (final state):")
            print(f"  Max real part of eigenvalues: {stab_e2['max_real_part']:.4f}")
            print(f"  Unstable: {stab_e2['unstable']}")
        elif sc == 3 or sc == 4:  # E₃ or oscillatory state
            y_final = [res['s'][-1], res['r'][-1], res['g'][-1],
                       res['d1'][-1], res['d2'][-1], res['d3'][-1],
                       (res['H'][-1] - mf['H_0']) / mf['scale_H']]
            stab_e3 = check_e3_stability(y_final, mf['model'])
            print("\nRouth‑Hurwitz stability for E₃ (final state):")
            print(f"  a1 = {stab_e3['a1']:.4f}, a2 = {stab_e3['a2']:.4f}, a3 = {stab_e3['a3']:.4f}, a4 = {stab_e3['a4']:.4f}")
            print(f"  Conditions: cond1={stab_e3['cond1']}, cond2={stab_e3['cond2']}, cond3={stab_e3['cond3']}, cond4={stab_e3['cond4']}")
            print(f"  Stable: {stab_e3['stable']}")

        # Plot and save
        save_path = output_dir / f'Scenario_{sc}_dynamics.pdf'
        plot_scenario_dynamics(sc, res, mf, save_path=save_path)
        print(f"Saved: {save_path}")

    # pH landscape figure
    print("\nGenerating pH Landscape with thresholds")
    save_path = output_dir / 'pH_Landscape_All_Scenarios.pdf'
    plot_ph_landscape(scenario_results, model_funcs_list, save_path=save_path)

    print("\n" + "="*60)
    print(f"All figures saved in '{output_dir}/'")
    print("="*60)

    # Final summary with stability analysis
    print("\nFINAL SUMMARY OF ALL SCENARIOS:")
    print("-"*60)
    for i in [1,2,3,4]:
        name = ["Successful Eradication", "Resistant Persistence", "Coexistence", "Oscillatory Dynamics"][i-1]
        r = scenario_results[i]
        print(f"\nScenario {i}: {name}")
        print(f"  Final R_s = {r['R_s']:.3f}")
        print(f"  Final R_r = {r['R_r']:.3f}")
        print(f"  Final R_r^σ = {r['R_r_sigma']:.3f}")
        print(f"  Final R_r^φ = {r['R_r_phi']:.3f}")

        R_s, R_r = r['R_s'], r['R_r']
        R_sigma, R_phi = r['R_r_sigma'], r['R_r_phi']
        if R_s < 1 and R_r < 1:
            print("  → Equilibrium E₀ (Infection-free) is stable")
        elif R_s < 1 and R_r > 1:
            if R_r < R_phi:
                print("  → Equilibrium E₂ (Resistant-only with immunity) is stable")
            else:
                print("  → Equilibrium E₁ (Resistant-only without immunity) may exist")
        elif R_s > 1 and R_r > 1:
            if R_r < R_sigma:
                print("  → Equilibrium E* (Coexistence without immunity) may exist")
            elif R_r < R_phi:
                print("  → Equilibrium E₃ (Coexistence with immunity) is stable")
            else:
                print("  → Complex dynamics (multiple equilibria possible)")

if __name__ == "__main__":
    main()
