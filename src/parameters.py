import numpy as np

def get_base_parameters():
    """Base parameters from Table 1 of the manuscript."""
    return {
        # Bacterial growth
        'beta_S_max': 16.66,
        'beta_R_max': 9.996,
        'K': 2.1e3,

        # Antibiotics (RIF, CIP, CLT)
        'kappa_i0': 36.0,
        'nu_i': 0.00025,
        'mu': np.array([6.6e-8, 3.8e-8, 3e-9]),
        'M_i': np.array([0.0012, 0.0025, 2.77]),
        'tau_i': np.array([0.96, 0.45, 0.35]),

        # Horizontal gene transfer & immune
        'eta': 1e-5,
        'gamma_immune': 6e-6,
        'rho': 0.6,
        'phi': 1.0,

        # Death / detachment
        'delta_S': 0.0037,
        'delta_R': 0.0037,
        'zeta': 0.5,

        # pH dynamics
        'alpha': 0.01,
        'beta_acid': 0.5,
        'H_max': 7.0,
        'H_0': 2.5,
        'H_opt': 7.0,
        'H_opt_abx': 5.5,
        'sigma_abx': 1.0,

        # Dietary perturbations
        'gamma_diet_vals': np.array([0.10, 0.15, 0.08]),
        'lambda_decay': 2.0,
        'meal_times': np.array([8/24, 12/24, 18/24])
    }

def get_scenario_parameters(scenario):
    """Modify base parameters to achieve each scenario."""
    base = get_base_parameters()
    if scenario == 1:          # Successful Eradication
        base['beta_R_max'] = 0.664
        base['delta_S'] = 0.37
        base['delta_R'] = 0.37
    elif scenario == 2:        # Resistant Persistence (use base, Table1)
        pass
    elif scenario == 3:        # Coexistence
        base['beta_S_max'] = 19.0
        base['beta_R_max'] = 10.0
        base['kappa_i0'] = 0.9
    elif scenario == 4:  # Oscillatory Dynamics
        base['K'] = 2.1e8
    return base
