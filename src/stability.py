import numpy as np

def numerical_jacobian(t, y, model_func, eps=1e-6):
    """
    Compute Jacobian matrix of the ODE system at (t, y) using central differences.
    Returns an n x n array.
    """
    n = len(y)
    f0 = np.array(model_func(t, y))
    J = np.zeros((n, n))
    for i in range(n):
        y_pert = y.copy()
        y_pert[i] += eps
        f_pert = np.array(model_func(t, y_pert))
        J[:, i] = (f_pert - f0) / eps
    return J

def extract_biological_jacobian(full_jac):
    """Extract the 4x4 Jacobian for (s, r, g, h) from the full 7x7."""
    return full_jac[:4, :4]

def routh_hurwitz_4(coeffs):
    """
    Check Routh‑Hurwitz conditions for a 4th‑order polynomial:
        λ⁴ + a1 λ³ + a2 λ² + a3 λ + a4 = 0
    Returns a dictionary with the conditions and overall stability.
    """
    a1, a2, a3, a4 = coeffs
    cond1 = a1 > 0
    cond2 = a4 > 0
    cond3 = (a1 * a2 - a3) > 0
    cond4 = (a3 * (a1 * a2 - a3) - a1**2 * a4) > 0
    stable = cond1 and cond2 and cond3 and cond4
    return {
        'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4,
        'cond1': cond1, 'cond2': cond2, 'cond3': cond3, 'cond4': cond4,
        'stable': stable
    }

def check_e2_stability(y, model_func):
    """
    For a state resembling E₂ (s=0, g=r), compute the Jacobian and check if any
    eigenvalue has positive real part. Returns a dict with eigenvalues and stability flag.
    """
    t = 0  # autonomous system, time does not matter
    J_full = numerical_jacobian(t, y, model_func)
    J_bio = extract_biological_jacobian(J_full)
    eigvals = np.linalg.eigvals(J_bio)
    unstable = np.any(eigvals.real > 1e-6)
    return {
        'eigenvalues': eigvals,
        'unstable': unstable,
        'max_real_part': np.max(eigvals.real)
    }

def check_e3_stability(y, model_func):
    """
    For a general coexistence state, compute the characteristic polynomial of the
    4x4 biological Jacobian and apply the Routh‑Hurwitz criterion.
    """
    t = 0
    J_full = numerical_jacobian(t, y, model_func)
    J_bio = extract_biological_jacobian(J_full)
    # numpy.poly returns coefficients from highest degree to constant
    coeffs = np.poly(J_bio)   # length 5 for a 4x4 matrix: [1, a1, a2, a3, a4]
    a1, a2, a3, a4 = coeffs[1:5]
    return routh_hurwitz_4([a1, a2, a3, a4])
