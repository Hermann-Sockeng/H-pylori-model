import numpy as np

# ============================================================================
# Standalone dietary perturbation functions
# ============================================================================

def gamma_diet_tilde(t, meal_times, gamma_diet_vals, lambda_decay, scale_H, n_days=30):
    """
    Non-dimensional dietary perturbation at time t.
    Returns dimensionless value (already divided by scale_H).
    """
    total = 0.0
    for day in range(n_days):
        for i, mt in enumerate(meal_times):
            t_meal = day + mt
            if t >= t_meal:
                total += gamma_diet_vals[i] * np.exp(-lambda_decay * (t - t_meal))
    return total / scale_H


def compute_average_gamma(meal_times, gamma_diet_vals, lambda_decay, scale_H):
    """
    Compute the time-average of gamma_diet_tilde over one period (1 day).
    
    Analytical formula for a single meal at time t_i:
        avg_i = (gamma_i / lambda_decay) * (1 - exp(-lambda_decay * (1 - t_i))) / scale_H
    
    The total average is the sum over all meals.
    """
    total_avg = 0.0
    for i, mt in enumerate(meal_times):
        # Contribution from meal i
        avg_i = (gamma_diet_vals[i] / lambda_decay) * (1.0 - np.exp(-lambda_decay * (1.0 - mt)))
        total_avg += avg_i
    return total_avg / scale_H


def setup_model(params):
    """Return all model functions for a given parameter set."""
    
    # Extract parameters
    beta_S_max = params['beta_S_max']
    beta_R_max = params['beta_R_max']
    K = params['K']
    kappa_i0 = params['kappa_i0']
    nu_i = params['nu_i']
    mu = params['mu']
    M_i = params['M_i']
    tau_i = params['tau_i']
    eta = params['eta']
    gamma_immune = params['gamma_immune']
    rho = params['rho']
    phi = params['phi']
    delta_S = params['delta_S']
    delta_R = params['delta_R']
    zeta = params['zeta']
    alpha = params['alpha']
    beta_acid = params['beta_acid']
    H_max = params['H_max']
    H_0 = params['H_0']
    H_opt = params['H_opt']
    H_opt_abx = params['H_opt_abx']
    sigma_abx = params['sigma_abx']
    gamma_diet_vals = params['gamma_diet_vals']
    lambda_decay = params['lambda_decay']
    meal_times = params['meal_times']

    # Non-dimensional scaling
    scale_H = H_max - H_0
    nu_i_tilde = nu_i / M_i
    mu_i_tilde = mu * M_i
    eta_tilde = eta * K               # σ in the paper
    gamma_tilde = gamma_immune * phi * K   # φ in the paper
    zeta_tilde = zeta
    Delta_S = delta_S + zeta_tilde
    Delta_R = delta_R + zeta_tilde
    alpha_tilde = alpha * K
    beta_tilde = beta_acid
    M = np.sum(mu_i_tilde)

    # Compute constant average dietary perturbation
    gamma_avg = compute_average_gamma(meal_times, gamma_diet_vals, lambda_decay, scale_H)

    # ========================================================================
    # pH-dependent functions
    # ========================================================================
    
    def H_from_h(h):
        return H_0 + scale_H * h

    def beta_S(h):
        H = H_from_h(h)
        arg = H - H_opt
        return beta_S_max * (1 - np.tanh(arg)**2)

    def beta_R(h):
        H = H_from_h(h)
        arg = H - H_opt
        return beta_R_max * (1 - np.tanh(arg)**2)

    def kappa_i(h):
        H = H_from_h(h)
        arg = (H - H_opt_abx) / sigma_abx
        return kappa_i0 * np.exp(-0.5 * arg**2)

    def Q(h, d1, d2, d3):
        """Composite antibiotic pressure function"""
        kvals = kappa_i(h)
        total = 0
        for i, d in enumerate([d1, d2, d3]):
            denom = nu_i_tilde[i] + d
            total += kvals * d / denom + mu_i_tilde[i] * d
        return total

    def reproductive_numbers(h):
        """Calculate pH-dependent reproductive numbers (at equilibrium d_i=1)"""
        d_eq = 1.0
        k_eq = kappa_i(h)
        Q_eq = 0
        for i in range(3):
            denom = nu_i_tilde[i] + d_eq
            Q_eq += k_eq / denom + mu_i_tilde[i]
        R_s = beta_S(h) / (Q_eq + Delta_S)
        R_r = beta_R(h) / Delta_R
        return R_s, R_r

    def R_r_sigma_threshold(h):
        """Calculate R_r^σ threshold: (1 + σ/β_S) / (1/R_s + σ/β_S)"""
        R_s_val, _ = reproductive_numbers(h)
        betaS = beta_S(h)
        sigma = eta_tilde
        if R_s_val > 0 and betaS > 0:
            num = 1 + sigma / betaS
            den = 1 / R_s_val + sigma / betaS
            return num / den
        return np.inf

    def R_r_phi_threshold(h):
        """Calculate R_r^φ threshold: 1 / (1 - r0_tilde * (1 + φ/β_R))"""
        _, R_r_val = reproductive_numbers(h)
        betaS = beta_S(h)
        betaR = beta_R(h)
        sigma = eta_tilde
        phi_param = gamma_tilde
        if betaS > 0:
            r0_tilde = 1 / (1 + (phi_param + sigma) / betaS)
        else:
            r0_tilde = 0
        if betaR > 0:
            denom = 1 - r0_tilde * (1 + phi_param / betaR)
            if abs(denom) > 1e-10:
                return 1 / denom
            else:
                return np.inf if denom > 0 else -np.inf
        return np.inf

    # ========================================================================
    # ODE systems: time-dependent vs constant average diet
    # ========================================================================

    def model_time_dependent(t, y):
        """Full non-autonomous model with time-dependent dietary perturbation"""
        s, r, g, d1, d2, d3, h = y
        
        betaS = beta_S(h)
        betaR = beta_R(h)
        Qval = Q(h, d1, d2, d3)

        ds = (betaS * s * (1 - (s + r)) - Qval * s -
              eta_tilde * s * r - gamma_tilde * g * s - Delta_S * s)
        
        dr = (betaR * r * (1 - (s + r)) + M * s +
              eta_tilde * s * r - gamma_tilde * g * r - Delta_R * r)
        
        if s + r > 1e-10:
            dg = rho * g * (1 - g / (s + r))
        else:
            dg = -rho * g
        
        dd1 = tau_i[0] * (1 - d1)
        dd2 = tau_i[1] * (1 - d2)
        dd3 = tau_i[2] * (1 - d3)
        
        # Time-dependent diet
        dh = (alpha_tilde * (s + r) * (1 - h) - beta_tilde * h +
              gamma_diet_tilde(t, meal_times, gamma_diet_vals, lambda_decay, scale_H))
        
        return [ds, dr, dg, dd1, dd2, dd3, dh]

    def model_constant_diet(t, y):
        """Autonomous model with constant average dietary perturbation"""
        s, r, g, d1, d2, d3, h = y
        
        betaS = beta_S(h)
        betaR = beta_R(h)
        Qval = Q(h, d1, d2, d3)

        ds = (betaS * s * (1 - (s + r)) - Qval * s -
              eta_tilde * s * r - gamma_tilde * g * s - Delta_S * s)
        
        dr = (betaR * r * (1 - (s + r)) + M * s +
              eta_tilde * s * r - gamma_tilde * g * r - Delta_R * r)
        
        if s + r > 1e-10:
            dg = rho * g * (1 - g / (s + r))
        else:
            dg = -rho * g
        
        dd1 = tau_i[0] * (1 - d1)
        dd2 = tau_i[1] * (1 - d2)
        dd3 = tau_i[2] * (1 - d3)
        
        # Constant average diet
        dh = (alpha_tilde * (s + r) * (1 - h) - beta_tilde * h + gamma_avg)
        
        return [ds, dr, dg, dd1, dd2, dd3, dh]

    # ========================================================================
    # Return all functions and parameters
    # ========================================================================
    
    return {
        'model_time_dependent': model_time_dependent,
        'model_constant_diet': model_constant_diet,
        'H_from_h': H_from_h,
        'beta_S': beta_S,
        'beta_R': beta_R,
        'kappa_i': kappa_i,
        'reproductive_numbers': reproductive_numbers,
        'R_r_sigma_threshold': R_r_sigma_threshold,
        'R_r_phi_threshold': R_r_phi_threshold,
        'alpha_tilde': alpha_tilde,
        'beta_tilde': beta_tilde,
        'gamma_tilde': gamma_tilde,
        'eta_tilde': eta_tilde,
        'Delta_S': Delta_S,
        'Delta_R': Delta_R,
        'scale_H': scale_H,
        'gamma_avg': gamma_avg,
        'H_0': H_0,
        'H_max': H_max,
        'H_opt': H_opt,
        'H_opt_abx': H_opt_abx,
        'params': params
    }