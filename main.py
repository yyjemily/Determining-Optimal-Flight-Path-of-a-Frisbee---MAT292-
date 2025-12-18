import numpy as np 
import cv2 as cv 
import pandas as pd 

#parameters for simulation 
m = 0.175 # mass of the frisbee in kg (USA Ultimate standard)
d = 0.274 # diameter of the frisbee in meters
g = 9.81 # acceleration due to gravity in m/s^2
area = np.pi * (d/2)**2  # cross-sectional area of the frisbee in m^2 
rho_fluid = 1.225 # density of air at sea level in kg/m^3 

#moment of inertia matrix 

#guess position initial conditions (x, y, z) 
pos = np.array([0.0, 0.0, 1.5]) # starting at 1.5 meters above ground

#guess velocity initial conditions (u, v, w)
v = np.array([14.0, 0.0, 14]) # initial velocity in m/s

#guess angle initial conditions (phi, theta, gamma)
ang = np.array([0.0, 0.0, (np.pi)/4]) # initial angular velocity in rad

#parameters as a matrix 
params = np.concatenate((pos, v, ang))

def solve_ODE(params, t):
    #solve ode using RK4 
    C_D0 = 0.08
    C_L0 = 0.15
    C_DA = 2.72
    C_M0 = -0.02
    
    
    

#extract C_D0, C_L0, C_D0, C_M0 