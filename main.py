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
vel = np.array([14.0, 0.0, 2.0]) # initial velocity in m/s

#guess angle initial conditions (phi, theta, gamma)
ang = np.array([0.0, 0.0, (np.pi)/4]) # initial angular velocity in rad


