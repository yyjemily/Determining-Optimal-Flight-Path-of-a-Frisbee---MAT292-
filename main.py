import numpy as np 
import pandas as pd 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import os 
import glob
import time
import math 
import matplotlib.pyplot as plt

#define constants
MASS = 0.175 
G = 9.81
AREA = 0.057 # m^2
RHO = 1.225
alpha0 = math.pi/6 # Reference angle of attack

def frisbee_ode(t, state, cd0, cda, cl0, cla):
    x, y, vx, vy = state
    v_mag = np.sqrt(vx**2 + vy**2)
    if v_mag < 1e-6: return [0, 0, 0, -G]
    
    theta = np.arctan2(vy, vx)
    alpha = theta 
    
    cl = cl0 + cla * alpha
    cd = cd0 + cda * (alpha - alpha0)**2
    
    accel_factor = (0.5 * RHO * v_mag**2 * AREA) / MASS
    ax = accel_factor * (-cd * np.cos(theta) - cl * np.sin(theta))
    ay = accel_factor * (-cd * np.sin(theta) + cl * np.cos(theta)) - G
    return [vx, vy, ax, ay]

def objective_function(params, t_eval, experimental_data, init_state):
    cd0, cda, cl0, cla = params
    sol = solve_ivp(frisbee_ode, (t_eval[0], t_eval[-1]), init_state, 
                    args=(cd0, cda, cl0, cla), t_eval=t_eval, method='RK45')
    
    if not sol.success or sol.y.shape[1] < len(t_eval):
        return 1e9 
    
    return np.sum((sol.y[0] - experimental_data[:, 0])**2 + 
                  (sol.y[1] - experimental_data[:, 1])**2)

# --- 3. BATCH OPTIMIZATION & PLOTTING ---
results_folder = "csv/" 
csv_files = glob.glob(os.path.join(results_folder, "*_results.csv"))
master_results = []

initial_guess = [0.15, 0.1, 0.1, 1.2]
bounds = [(0, 1), (0, 2), (0, 1), (0, 5)]

for csv_file in csv_files:
    base_name = os.path.basename(csv_file).replace('_results.csv', '')
    df = pd.read_csv(csv_file)
    t_pts = df['Time_s'].values
    exp_data = df[['X_m', 'Y_m']].values
    
    # Estimate Initial Conditions
    dt = t_pts[1] - t_pts[0]
    vx0 = (df['X_m'].iloc[1] - df['X_m'].iloc[0]) / dt
    vy0 = (df['Y_m'].iloc[1] - df['Y_m'].iloc[0]) / dt
    start_state = [df['X_m'].iloc[0], df['Y_m'].iloc[0], vx0, vy0]
    
    # Run Backward Pass (Optimization)
    res = minimize(objective_function, initial_guess, 
                   args=(t_pts, exp_data, start_state), 
                   method='L-BFGS-B', bounds=bounds)
    
    if res.success:
        opt = res.x # Optimized [cd0, cda, cl0, cla]
        
        # Run simulation with optimized parameters
        sol = solve_ivp(frisbee_ode, (t_pts[0], t_pts[-1]), start_state, 
                        args=tuple(opt), t_eval=np.linspace(t_pts[0], t_pts[-1], 100))
        
        # Calculate Cl and Cd for every point in the simulated trajectory
        sim_vx, sim_vy = sol.y[2], sol.y[3]
        sim_alpha = np.arctan2(sim_vy, sim_vx)
        sim_cl = opt[2] + opt[3] * sim_alpha
        sim_cd = opt[0] + opt[1] * (sim_alpha - alpha0)**2


        plot_dir = "final_analysis_plots"
        os.makedirs(plot_dir, exist_ok=True)

        sim_vx, sim_vy = sol.y[2], sol.y[3]
        sim_alpha = np.arctan2(sim_vy, sim_vx) # Angle of attack (radians)
        alpha_deg = np.degrees(sim_alpha)
        
        sim_cl = opt[2] + opt[3] * sim_alpha
        sim_cd = opt[0] + opt[1] * (sim_alpha - alpha0)**2

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        #trajectory 
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t_pts, exp_data[:, 0], 'ro', label='Actual X (m)', markersize=4, alpha=0.4)
        ax1.plot(t_pts, exp_data[:, 1], 'go', label='Actual Y (m)', markersize=4, alpha=0.4)
        ax1.plot(sol.t, sol.y[0], 'r-', label='Predicted X', linewidth=2)
        ax1.plot(sol.t, sol.y[1], 'g-', label='Predicted Y', linewidth=2)
        ax1.set_title(f"Trajectory Optimization: {base_name}", fontsize=14)
        ax1.set_ylabel("Distance (m)")
        ax1.set_xlabel("Time (s)")
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)

        #lift curve
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(alpha_deg, sim_cl, 'b-', linewidth=3)
        ax2.set_title(r"Lift Curve ($C_L$ vs $\alpha$)")
        ax2.set_xlabel(r"Angle of Attack $\alpha$ (deg)")
        ax2.set_ylabel(r"Coefficient $C_L$")
        ax2.grid(True, linestyle=':', alpha=0.6)

        # drag curve 
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(alpha_deg, sim_cd, 'k-', linewidth=3)
        ax3.axvline(np.degrees(alpha0), color='red', linestyle='--', label=r'Reference $\alpha_0$')
        ax3.set_title(r"Drag Polar ($C_D$ vs $\alpha$)")
        ax3.set_xlabel(r"Angle of Attack $\alpha$ (deg)")
        ax3.set_ylabel(r"Coefficient $C_D$")
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend()

        plt.tight_layout()
        final_plot_path = os.path.join(plot_dir, f"{base_name}_full_analysis.png")
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Plot path: {final_plot_path}")
if master_results:
    pd.DataFrame(master_results).to_csv("optimized_aerodynamics_summary.csv", index=False)