import numpy as np
from scipy.integrate import solve_ivp


def run_scenario(scenario_num, model_funcs, use_time_dependent=True, t_span=(0, 30), t_eval=None):
    """
    Run simulation for a specific scenario.
    
    Parameters
    ----------
    scenario_num : int
        Scenario number (1-4)
    model_funcs : dict
        Dictionary returned by setup_model()
    use_time_dependent : bool
        If True, use full time-dependent dietary perturbation.
        If False, use constant average diet.
    t_span : tuple
        (t_start, t_end) in days
    t_eval : array or None
        Time points for output. If None, uses 3000 points.
    
    Returns
    -------
    dict
        Simulation results including time series and final values.
    """
    
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 3000)

    # Initial conditions
    y0 = [0.6, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4]

    # Choose the appropriate ODE model
    if use_time_dependent:
        ode_model = model_funcs['model_time_dependent']
    else:
        ode_model = model_funcs['model_constant_diet']

    # Solve ODE system
    sol = solve_ivp(ode_model, t_span, y0, t_eval=t_eval,
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
    
    if use_time_dependent:
        # Full time-dependent diet
        for i, ti in enumerate(t):
            gamma_tot = 0
            for day in range(31):  # cover simulation period
                for j, mt in enumerate(params['meal_times']):
                    t_meal = day + mt
                    if ti >= t_meal:
                        gamma_tot += params['gamma_diet_vals'][j] * np.exp(
                            -params['lambda_decay'] * (ti - t_meal))
            gamma_plot[i] = gamma_tot / model_funcs['scale_H']
    else:
        # Constant average diet
        gamma_plot[:] = model_funcs['gamma_avg']

    # ========================================================================
    # Detailed console output
    # ========================================================================
    
    scenario_titles = [
        "Scenario 1: Successful Eradication",
        "Scenario 2: Resistant Persistence",
        "Scenario 3: Coexistence",
        "Scenario 4: Oscillatory Dynamics"
    ]
    
    diet_type = "time-dependent" if use_time_dependent else f"constant average (γ_avg = {model_funcs['gamma_avg']:.6f})"
    
    print(f"\n{'='*60}")
    print(f"SCENARIO {scenario_num}: {scenario_titles[scenario_num-1]}")
    print(f"{'='*60}")
    print(f"Diet model: {diet_type}")
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

    # Stability classification
    if scenario_num == 1:
        print(f"\n✓ Eradication successful: R_s < 1 and R_r < 1")
        print(f"  → Equilibrium E₀ (Infection-free) is stable")
    elif scenario_num == 2:
        print(f"\n✗ Treatment failure: Resistant bacteria persist")
        print(f"  Condition: R_s = {R_s:.3f} < 1, R_r = {R_r:.3f} > 1")
        if R_r < R_r_phi:
            print(f"  → Equilibrium E₂ (Resistant-only with immunity) is stable")
        else:
            print(f"  → Equilibrium E₁ (Resistant-only without immunity) may exist")
    elif scenario_num == 3:
        print(f"\n⚠ Coexistence: Both strains persist under immune surveillance")
        print(f"  Condition: R_s = {R_s:.3f} > 1, R_r = {R_r:.3f} > 1")
        if R_r < R_r_sigma:
            print(f"  → Equilibrium E* (Coexistence without immunity) may exist")
        elif R_r < R_r_phi:
            print(f"  → Equilibrium E₃ (Coexistence with immunity) is stable")
        else:
            print(f"  → Complex dynamics (multiple equilibria possible)")
    elif scenario_num == 4:
        print(f"\n↻ Oscillatory dynamics: System shows periodic behavior")

    # ========================================================================
    # Return results
    # ========================================================================
    
    return {
        't': t,
        's': s,
        'r': r,
        'g': g,
        'd1': d1,
        'd2': d2,
        'd3': d3,
        'H': H,
        'gamma': gamma_plot,
        'R_s': R_s,
        'R_r': R_r,
        'R_r_sigma': R_r_sigma,
        'R_r_phi': R_r_phi,
        'use_time_dependent': use_time_dependent,
        'gamma_avg': model_funcs['gamma_avg'] if not use_time_dependent else None
    }