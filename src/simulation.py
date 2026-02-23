import numpy as np
from scipy.integrate import solve_ivp

def run_scenario(scenario_num, model_funcs, t_span=(0,30), t_eval=None):
    """Run simulation for a given scenario and return results."""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 3000)

    # Initial conditions
    y0 = [0.6, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4]

    sol = solve_ivp(model_funcs['model'], t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8)
    t = sol.t
    s, r, g, d1, d2, d3, h = sol.y
    H = model_funcs['H_from_h'](h)

    # Final reproductive numbers and thresholds
    R_s, R_r = model_funcs['reproductive_numbers'](h[-1])
    R_r_sigma = model_funcs['R_r_sigma_threshold'](h[-1])
    R_r_phi = model_funcs['R_r_phi_threshold'](h[-1])

    # Dietary perturbation for plotting
    gamma_plot = np.zeros_like(t)
    params = model_funcs['params']
    for i, ti in enumerate(t):
        gamma_tot = 0
        for day in range(31):
            for j, mt in enumerate(params['meal_times']):
                t_meal = day + mt
                if ti >= t_meal:
                    gamma_tot += params['gamma_diet_vals'][j] * np.exp(
                        -params['lambda_decay'] * (ti - t_meal))
        gamma_plot[i] = gamma_tot / model_funcs['scale_H']

    # --- Detailed console output (restored) ---
    scenario_titles = [
        "Scenario 1: Successful Eradication",
        "Scenario 2: Resistant Persistence",
        "Scenario 3: Coexistence",
        "Scenario 4: Oscillatory Dynamics"
    ]
    print(f"\n{'='*60}")
    print(f"SCENARIO {scenario_num}: {scenario_titles[scenario_num-1]}")
    print(f"{'='*60}")
    print(f"Key parameters:")
    print(f"  beta_S_max = {params['beta_S_max']}")
    print(f"  beta_R_max = {params['beta_R_max']}")
    print(f"  kappa_i0 = {params['kappa_i0']}")
    print(f"  K = {params['K']}")
    print(f"  rho = {params['rho']}")
    print(f"\nFinal state:")
    print(f"  s_final = {s[-1]:.6f}")
    print(f"  r_final = {r[-1]:.6f}")
    print(f"  g_final = {g[-1]:.6f}")
    print(f"  H_final = {H[-1]:.2f}")
    print(f"  R_s(H_final) = {R_s:.3f}")
    print(f"  R_r(H_final) = {R_r:.3f}")
    print(f"  R_r^σ(H_final) = {R_r_sigma:.3f}")
    print(f"  R_r^φ(H_final) = {R_r_phi:.3f}")

    if scenario_num == 1:
        print(f"\n✓ Eradication successful: R_s < 1 and R_r < 1")
    elif scenario_num == 2:
        print(f"\n✗ Treatment failure: Resistant bacteria persist")
        print(f"  Condition: R_s = {R_s:.3f} < 1, R_r = {R_r:.3f} > 1")
    elif scenario_num == 3:
        print(f"\n⚠ Coexistence: Both strains persist under immune surveillance")
        print(f"  Condition: R_s = {R_s:.3f} > 1, R_r = {R_r:.3f} > 1")
    elif scenario_num == 4:
        print(f"\n↻ Oscillatory dynamics: System shows periodic behavior")
    # --- End of detailed output ---

    return {
        't': t, 's': s, 'r': r, 'g': g,
        'd1': d1, 'd2': d2, 'd3': d3,
        'H': H, 'gamma': gamma_plot,
        'R_s': R_s, 'R_r': R_r,
        'R_r_sigma': R_r_sigma, 'R_r_phi': R_r_phi
    }