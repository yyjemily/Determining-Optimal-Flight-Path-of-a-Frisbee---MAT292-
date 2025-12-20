# Determining-Optimal-Flight-Path-of-a-Frisbee---MAT292-
Determining Coefficients to Obtain the Optimal Flight Path of a Frisbee 

## Project Overview:
This project aims to estimate the lift (C_L)and drag (C_D) aerodynamic coefficients of a frisbee by taking experimental video tracking data with OpenCV and Runge-Kutta-Fehlberg (RK45) to simulate the Frisbee's flight path. 

## Methodology
2.1 Physics Model (Forward-Propagation)
Solver: Runge-Kutta-Fehlberg (RK45) with adaptive-step.

2D flight dynamics (x,y,v_x,v_y,).
* **Lift Coefficient**: $C_L = C_{L0} + C_{L\alpha} \cdot \alpha$
* **Drag Coefficient**: $C_D = C_{D0} + C_{D\alpha} \cdot (\alpha - \alpha_0)^2$


## Data Useage 
Environment Setup: 
Run the following command if you are using pip:
pip install -r requirements.txt

Prepare Experimental Data: 
1. Create a folder named "/csv" in the project directory, alternative upload videos to a folder named "/frisbee_real_data" and follow the commands to track the trajectory of the frisbee. 
2. Save OpenCV Lucas-Kanade results as: [Name]_results.csv
3. Required columns: Time_s, X_m, Y_m.

Review Outputs: 
1. aerodynamic_summary.csv: list of optimized coefficients.
2. /aerodynamic_fit_plots: visualization plots for every trial.

## Dependencies
NumPy: Numerical array processing.
SciPy: ODE integration (solve_ivp) and Optimization (minimize).
Pandas: CSV data management.
Matplotlib: 4-panel visualization.