import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pygame
import pygame.surfarray

# =============================================================================
# Relevant functions for the oil drop simulation 
# =============================================================================

#Initial conditions of oil density 

#Thin layer initial condition 
def thin_layer(n_modes, X, Y):
    # Superposition of sin waves as a random thin layer 
    n_modes = 10
    wavelength_modes = np.random.random((n_modes,1,1))*0.5+0.5
    theta_modes = np.random.random((n_modes,1,1))*2*np.pi
    phi_modes = np.random.random((n_modes,1,1))*np.pi
    layer = lambda x, y : 400*np.sum(np.sin(2*np.pi/wavelength_modes*( np.cos(theta_modes)*x + np.sin(theta_modes)*y ) + phi_modes), axis=0)/n_modes + 400
    rho0 = layer(X,Y)
    return rho0 

# Raining droplets
def drop_shape(rho, amplitude, sigma, x0, y0, index_x, index_y):
    #Gaussian shape for the oil drops 
    distance_squared = (index_x - x0)**2 + (index_y - y0)**2
    gaussian_drop = amplitude * np.exp(-distance_squared / (2 * sigma**2))
    rho += gaussian_drop
    return rho

# Fixed droplets initial condition
def fixed_droplets(rho0, x_drops, y_drops, amplitude, sigma, Ngrid, index_x, index_y):
    x_drops = [i/500 * Ngrid for i in x_drops]
    y_drops = np.copy(x_drops)
    x_drops, y_drops = np.meshgrid(x_drops, y_drops)
    for i in range(len(x_drops)):
        for j in range(len(y_drops)):
            rho = drop_shape(rho0, amplitude, sigma, x_drops[i,j], y_drops[i,j], index_x, index_y)

    return rho 


# =============================================================================
# Velocity field (swirl pattern)
# =============================================================================

def swirl(U,X,Y):
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    r[r == 0] = 1e-10 #avoid singularity in the center 
    U = 5e2  # Strength of the swirl 
    uy = -U * Y / r # x-component velocity
    ux = U * X / r  # y-component velocity 
    return ux, uy

def shear(Umin, Umax, Ngrid):
    y_profile = np.linspace(Umin, Umax, Ngrid)
    ux = np.tile(y_profile[:, np.newaxis], (1, Ngrid))
    uy = np.zeros((Ngrid, Ngrid))
    return ux, uy
