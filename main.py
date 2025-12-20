import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import os
import glob
import math
import matplotlib.pyplot as plt

#define constants 
MASS = 0.175
G = 9.81
AREA = 0.057 
RHO = 1.225
alpha0 = math.pi/6  

#define the ode for frisbee
def frisbee_ode(t, state, cd0, cda, cl0, cla):
    x, y, vx, vy = state
    v_mag = np.sqrt(vx**2 + vy**2)
    if v_mag < 1e-6: return [0, 0, 0, -G]
    
    #angle of attack (varying)
    alpha = np.arctan2(vy, vx) 
    
    #lift coefficient 
    cl = cl0 + cla * alpha
    
    #drag coefficient
    cd = cd0 + cda * (alpha - alpha0)**2
    
    accel_factor = (0.5 * RHO * v_mag**2 * AREA) / MASS
    
    #force balance, f =ma  for x and y components
    ax = accel_factor * (-cd * np.cos(alpha) - cl * np.sin(alpha))
    ay = accel_factor * (-cd * np.sin(alpha) + cl * np.cos(alpha)) - G
    return [vx, vy, ax, ay]

#solve ode using RK45 ivp 
def solve_ode_get_error(params, t_eval, experimental_data, init_state):
    cd0, cda, cl0, cla = params
    sol = solve_ivp(frisbee_ode, (t_eval[0], t_eval[-1]), init_state, 
                    args=(cd0, cda, cl0, cla), t_eval=t_eval, method='RK45')
    if not sol.success or sol.y.shape[1] < len(t_eval): return 1e9 
    
    # calculate sum of square errors  
    return np.sum((sol.y[0] - experimental_data[:, 0])**2 + (sol.y[1] - experimental_data[:, 1])**2)

#analyze data 
results_folder = "csv/" 
plot_dir = "aerodynamic_fit_plots"
os.makedirs(plot_dir, exist_ok=True)
csv_files = glob.glob(os.path.join(results_folder, "*_results.csv"))
master_results = []

for csv_file in csv_files:
    base_name = os.path.basename(csv_file).replace('_results.csv', '')
    df = pd.read_csv(csv_file)
    t_pts, exp_data = df['Time_s'].values, df[['X_m', 'Y_m']].values
    
    # guess inital conditions
    dt = t_pts[1] - t_pts[0]
    vx0, vy0 = (exp_data[1,0]-exp_data[0,0])/dt, (exp_data[1,1]-exp_data[0,1])/dt
    start_state = [exp_data[0,0], exp_data[0,1], vx0, vy0]
    
    # use optmizer L-BFGS-B to optimize 
    res = minimize(solve_ode_get_error, [0.15, 0.1, 0.1, 1.2], 
                   args=(t_pts, exp_data, start_state), 
                   method='L-BFGS-B', bounds=[(0,1),(0,2),(0,1),(0,5)])
    
    if res.success:
        opt = res.x
        
        t_fine = np.linspace(t_pts[0], t_pts[-1], 200)
        sol_fine = solve_ivp(frisbee_ode, (t_pts[0], t_pts[-1]), start_state, args=tuple(opt), t_eval=t_fine)
        sol_res = solve_ivp(frisbee_ode, (t_pts[0], t_pts[-1]), start_state, args=tuple(opt), t_eval=t_pts)
        
        # Calculate R^2 and RMSE
        sse = np.sum((exp_data[:, 0] - sol_res.y[0])**2 + (exp_data[:, 1] - sol_res.y[1])**2)
        sst = np.sum((exp_data[:, 0] - np.mean(exp_data[:, 0]))**2 + (exp_data[:, 1] - np.mean(exp_data[:, 1]))**2)
        r2_val = 1 - (sse / sst)
        rmse = np.sqrt(sse / (2 * len(t_pts)))
        
        # Coefficient arrays for plotting
        alpha_fine = np.arctan2(sol_fine.y[3], sol_fine.y[2])
        cl_fine = opt[2] + opt[3] * alpha_fine
        cd_fine = opt[0] + opt[1] * (alpha_fine - alpha0)**2

        #plots 
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Trajectory Plot
        axes[0,0].plot(t_pts, exp_data[:, 0], 'ro', label='Actual X', markersize=3, alpha=0.3)
        axes[0,0].plot(t_pts, exp_data[:, 1], 'go', label='Actual Y', markersize=3, alpha=0.3)
        axes[0,0].plot(sol_fine.t, sol_fine.y[0], 'r-', label='Model X', linewidth=2)
        axes[0,0].plot(sol_fine.t, sol_fine.y[1], 'g-', label='Model Y', linewidth=2)
        axes[0,0].set_title(f"Trajectory Fit ($R^2$: {r2_val:.4f})\nRMSE: {rmse:.4f}m")
        axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()

        #Residual Plot
        axes[0,1].plot(t_pts, exp_data[:, 0]-sol_res.y[0], 'rs-', label='X Error', markersize=4)
        axes[0,1].plot(t_pts, exp_data[:, 1]-sol_res.y[1], 'gs-', label='Y Error', markersize=4)
        axes[0,1].axhline(0, color='black', linewidth=1)
        axes[0,1].set_title("Residual Mapping (Actual - Predicted)"); axes[0,1].grid(True, alpha=0.3); axes[0,1].legend()

        #Linear Lift Plot
        axes[1,0].plot(np.degrees(alpha_fine), cl_fine, 'b-', linewidth=2.5)
        axes[1,0].set_title(r"Linear Lift Curve ($C_L = C_{L0} + C_{L\alpha}\alpha$)")
        axes[1,0].set_xlabel("Angle of Attack (deg)"); axes[1,0].set_ylabel(r"$C_L$"); axes[1,0].grid(True, alpha=0.3)

        #Quadratic Drag Plot
        axes[1,1].plot(np.degrees(alpha_fine), cd_fine, 'k-', linewidth=2.5)
        axes[1,1].axvline(np.degrees(alpha0), color='red', linestyle='--', label=r'Reference $\alpha_0$')
        axes[1,1].set_title(r"Quadratic Drag Polar ($C_D \propto \alpha^2$)")
        axes[1,1].set_xlabel("Angle of Attack (deg)"); axes[1,1].set_ylabel(r"$C_D$"); axes[1,1].grid(True, alpha=0.3); axes[1,1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_name}_analysis.png"), dpi=300)
        plt.close()

        master_results.append({
            'file': base_name, 'CD0': opt[0], 'CDa': opt[1], 
            'CL0': opt[2], 'CLa': opt[3], 'R2': r2_val, 'RMSE_m': rmse
        })

if master_results:
    pd.DataFrame(master_results).to_csv("final_result.csv", index=False)