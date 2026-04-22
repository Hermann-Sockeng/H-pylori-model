import matplotlib.pyplot as plt
import numpy as np

def plot_scenario_dynamics(scenario_num, results, model_funcs, save_path=None):
    """Create 3×2 subplot for a single scenario."""
    t = results['t']
    s, r, g = results['s'], results['r'], results['g']
    d1, d2, d3 = results['d1'], results['d2'], results['d3']
    H = results['H']
    gamma = results['gamma']

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.25, wspace=0.2)

    scenario_titles = [
        "Scenario 1: Successful Eradication",
        "Scenario 2: Resistant Persistence",
        "Scenario 3: Coexistence",
        "Scenario 4: Oscillatory Dynamics"
    ]
    
    # Add dietary model info to the title
    diet_label = "Time‑dependent diet" if results.get('use_time_dependent', True) else f"Constant average diet (γ_avg = {results.get('gamma_avg', 0):.5f})"
    
    fig.suptitle(f'{scenario_titles[scenario_num-1]}\n'
                 f'Final R_s = {results["R_s"]:.3f}, Final R_r = {results["R_r"]:.3f}  |  {diet_label}',
                 fontsize=14, fontweight='bold')

    # Sensitive bacteria
    ax = axes[0,0]
    ax.plot(t, s, 'b-', lw=2)
    ax.fill_between(t, 0, s, alpha=0.3, color='b')
    ax.set_ylabel('Sensitive Bacteria (s)')
    ax.set_title('Sensitive Bacteria Dynamics')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(s)*1.1)
    ax.margins(0.02)

    # Resistant bacteria
    ax = axes[0,1]
    ax.plot(t, r, 'r-', lw=2)
    ax.fill_between(t, 0, r, alpha=0.3, color='r')
    ax.set_ylabel('Resistant Bacteria (r)')
    ax.set_title('Resistant Bacteria Dynamics')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(r)*1.1)
    ax.margins(0.02)

    # Immune cells
    ax = axes[1,0]
    ax.plot(t, g, 'g-', lw=2)
    ax.fill_between(t, 0, g, alpha=0.3, color='g')
    ax.set_ylabel('Immune Cells (g)')
    ax.set_title('Immune Response Dynamics')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(g)*1.1)
    ax.margins(0.02)

    # Antibiotics
    ax = axes[1,1]
    ax.plot(t, d1, 'c-', lw=2, label='Rifampicin')
    ax.plot(t, d2, 'm-', lw=2, label='Ciprofloxacin')
    ax.plot(t, d3, 'y-', lw=2, label='Clarithromycin')
    ax.set_ylabel('Antibiotics (dᵢ)')
    ax.set_title('Antibiotic Concentrations')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.margins(0.02)

    # pH dynamics
    ax = axes[2,0]
    ax.plot(t, H, 'purple', lw=2, label='Gastric pH')
    ax.axhline(y=model_funcs['H_0'], color='r', ls='--', alpha=0.5,
               label=f'Baseline (H₀={model_funcs["H_0"]})')
    ax.axhline(y=model_funcs['H_max'], color='b', ls='--', alpha=0.5,
               label=f'Max (H_max={model_funcs["H_max"]})')
    ax.axhline(y=model_funcs['H_opt_abx'], color='g', ls='--', alpha=0.5,
               label=f'Optimal ABX (H={model_funcs["H_opt_abx"]})')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Gastric pH (H)')
    ax.set_title('Gastric pH Dynamics')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='center right')
    ax.set_ylim(1.5, 7.2)
    ax.margins(0.02)

    # Dietary perturbation
    ax = axes[2,1]
    ax.plot(t, gamma * model_funcs['scale_H'], 'orange', lw=2)
    ax.fill_between(t, 0, gamma * model_funcs['scale_H'], alpha=0.3, color='orange')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('pH Perturbation')
    ax.set_title('Dietary Perturbation')
    ax.grid(True, alpha=0.3)
    # Mark meal times (only meaningful for time‑dependent diet)
    if results.get('use_time_dependent', True):
        for day in range(31):
            for mt in model_funcs['params']['meal_times']:
                ax.axvline(x=day + mt, color='r', alpha=0.1, ls=':')
    ax.margins(0.02)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_ph_landscape(scenario_results_list, model_funcs_list, save_dir=None):
    """Create 2×2 plot of pH-dependent reproductive numbers + thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.25, wspace=0.2)
    axes = axes.flatten()

    scenario_titles = [
        "Scenario 1: Successful Eradication",
        "Scenario 2: Resistant Persistence",
        "Scenario 3: Coexistence",
        "Scenario 4: Oscillatory Dynamics"
    ]

    for i, (sc_num, mf) in enumerate(zip([1,2,3,4], model_funcs_list)):
        ax = axes[i]
        params = mf['params']
        H_range = np.linspace(params['H_0'], params['H_max'], 200)
        h_range = (H_range - params['H_0']) / (params['H_max'] - params['H_0'])

        # Compute reproductive numbers and thresholds
        R_s = np.array([mf['reproductive_numbers'](h)[0] for h in h_range])
        R_r = np.array([mf['reproductive_numbers'](h)[1] for h in h_range])
        R_sigma = np.array([mf['R_r_sigma_threshold'](h) for h in h_range])
        R_phi = np.array([mf['R_r_phi_threshold'](h) for h in h_range])

        # Cap thresholds at a high value for plotting (avoid negative or extremely large values)
        R_sigma_plot = np.where(np.isfinite(R_sigma) & (R_sigma > 0), R_sigma, 50)
        R_phi_plot = np.where(np.isfinite(R_phi) & (R_phi > 0), R_phi, 50)

        # Fill zones (only where values are > 1)
        # Eradication zone (both < 1)
        idx_erad = np.where((R_s < 1) & (R_r < 1))[0]
        if len(idx_erad):
            ax.fill_between(H_range[idx_erad], 0, 1, alpha=0.3, color='green', label='Eradication')

        # Resistant persistence (R_s < 1 < R_r)
        idx_res = np.where((R_s < 1) & (R_r > 1))[0]
        if len(idx_res):
            upper = R_r[idx_res]
            ax.fill_between(H_range[idx_res], 1, upper, alpha=0.2, color='orange', label='Resistant persistence')

        # Coexistence (both > 1)
        idx_coex = np.where((R_s > 1) & (R_r > 1))[0]
        if len(idx_coex):
            upper = np.maximum(R_s[idx_coex], R_r[idx_coex])
            ax.fill_between(H_range[idx_coex], 1, upper, alpha=0.2, color='purple', label='Coexistence')

        # Sensitive persistence (R_r < 1 < R_s) - less common
        idx_sens = np.where((R_r < 1) & (R_s > 1))[0]
        if len(idx_sens):
            ax.fill_between(H_range[idx_sens], 1, R_s[idx_sens], alpha=0.2, color='blue', label='Sensitive persistence')

        # Plot lines
        ax.plot(H_range, R_s, 'b-', lw=2, label=r'$\mathcal{R}_s(H)$')
        ax.plot(H_range, R_r, 'r-', lw=2, label=r'$\mathcal{R}_r(H)$')
        ax.plot(H_range, R_sigma_plot, 'b--', lw=1.5, alpha=0.7, label=r'$\mathcal{R}_r^\sigma(H)$')
        ax.plot(H_range, R_phi_plot, 'r--', lw=1.5, alpha=0.7, label=r'$\mathcal{R}_r^\varphi(H)$')
        ax.axhline(y=1, color='k', ls='--', lw=1.5, label=r'$\mathcal{R}=1$')

        # pH reference lines
        ax.axvline(x=mf['H_0'], color='gray', ls=':', alpha=0.5, label=f'Baseline pH={mf["H_0"]}')
        ax.axvline(x=mf['H_opt'], color='blue', ls=':', alpha=0.5, label=f'Optimal growth pH={mf["H_opt"]}')
        ax.axvline(x=mf['H_opt_abx'], color='red', ls=':', alpha=0.5, label=f'Optimal ABX pH={mf["H_opt_abx"]}')

        # Parameter box (no ρ)
        param_text = f"β_S_max = {params['beta_S_max']:.2f}\nβ_R_max = {params['beta_R_max']:.2f}\nκ_i0 = {params['kappa_i0']:.2f}\nK = {params['K']:.1e}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

        ax.set_xlabel('Gastric pH (H)')
        ax.set_ylabel(r'Reproductive Number $\mathcal{R}$')
        ax.set_title(scenario_titles[i], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # X-axis limits (extend to show optimal growth)
        ax.set_xlim(H_range[0]-0.2, H_range[-1]+0.3)

        # Y-axis limits based on scenario
        if sc_num == 1:  # Eradication scenario
            ax.set_ylim(0, 5)
        else:
            all_values = np.concatenate([R_s, R_r, R_sigma_plot, R_phi_plot])
            all_values = all_values[np.isfinite(all_values)]
            y_max = max(5, np.max(all_values) * 1.1) if len(all_values) > 0 else 20
            ax.set_ylim(0, min(20, y_max))

        ax.margins(0.02)
        if i == 0:
            # Place legend in upper right, outside the parameter box
            ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.4, 0.97), framealpha=0.9)
            #ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    plt.suptitle('pH-Dependent Fitness Landscape with Stability Thresholds', fontsize=14, fontweight='bold')
    if save_dir:
        # save_dir is a Path object; construct the full path
        save_path = save_dir / 'pH_Landscape_All_Scenarios.pdf'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig



def plot_diet_comparison(results_td, results_ca, model_funcs_list, save_path=None):
    """
    Create a comparison figure for time-dependent vs constant average diet.
    
    Layout: 4 rows (variables) × 4 columns (scenarios)
    - Rows: Sensitive bacteria, Resistant bacteria, Immune system, Gastric pH
    - Columns: Scenarios 1-4 with their titles
    
    Parameters
    ----------
    results_td : dict
        Dictionary of results from time-dependent diet (keyed by scenario number)
    results_ca : dict
        Dictionary of results from constant average diet (keyed by scenario number)
    model_funcs_list : list
        List of model function dictionaries for each scenario (order: 1,2,3,4)
    save_path : Path or None
        Where to save the figure.
    """
    scenario_titles = [
        "Scenario 1: Successful Eradication",
        "Scenario 2: Resistant Persistence",
        "Scenario 3: Coexistence",
        "Scenario 4: Oscillatory Dynamics"
    ]
    
    # Create figure: 4 rows (variables) × 4 columns (scenarios)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)
    
    # Variable labels for rows
    variable_labels = [
        "Sensitive Bacteria (s)",
        "Resistant Bacteria (r)",
        "Immune System (g)",
        "Gastric pH (H)"
    ]
    
    for col, sc in enumerate([1, 2, 3, 4]):
        # Get results
        td = results_td[sc]
        ca = results_ca[sc]
        mf = model_funcs_list[col]  # same order
        
        # Row 0: Sensitive bacteria
        ax = axes[0, col]
        ax.plot(td['t'], td['s'], 'b-', lw=1.5, label='Time‑dependent')
        ax.plot(ca['t'], ca['s'], 'r--', lw=1.5, label='Constant average')
        ax.set_ylabel(variable_labels[0], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(0.02)
        if col == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Row 1: Resistant bacteria
        ax = axes[1, col]
        ax.plot(td['t'], td['r'], 'b-', lw=1.5, label='Time‑dependent')
        ax.plot(ca['t'], ca['r'], 'r--', lw=1.5, label='Constant average')
        ax.set_ylabel(variable_labels[1], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(0.02)
        
        # Row 2: Immune system
        ax = axes[2, col]
        ax.plot(td['t'], td['g'], 'b-', lw=1.5, label='Time‑dependent')
        ax.plot(ca['t'], ca['g'], 'r--', lw=1.5, label='Constant average')
        ax.set_ylabel(variable_labels[2], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(0.02)
        
        # Row 3: Gastric pH
        ax = axes[3, col]
        ax.plot(td['t'], td['H'], 'b-', lw=1.5, label='Time‑dependent')
        ax.plot(ca['t'], ca['H'], 'r--', lw=1.5, label='Constant average')
        # Add baseline and max pH lines for reference
        ax.axhline(y=mf['H_0'], color='gray', ls=':', alpha=0.5)
        ax.axhline(y=mf['H_max'], color='gray', ls=':', alpha=0.5)
        ax.set_ylabel(variable_labels[3], fontsize=9)
        ax.set_xlabel('Time (days)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(0.02)
        
        # Add scenario title at the top of each column
        axes[0, col].set_title(scenario_titles[col], fontsize=10, fontweight='bold')
    
    fig.suptitle('Comparison of Time‑dependent vs Constant Average Diet', 
                 fontsize=14, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig