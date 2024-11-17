#########################################################################################
# November 17th 2024
# Louis-Simon Guité, Jérémy Peltier
# McGill Hackathon 2024, Project Oil Spills

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pygame
import pygame.surfarray
from functions import thin_layer, drop_shape, fixed_droplets, swirl, shear

# =============================================================================
# Relevant functions
# =============================================================================

def simulation_timestep(rho0, rho, ux, uy):
    # Propagate with forward-difference in time, central-difference in space
    drho_x = (rho0[2:, 1:-1] - 2 * rho0[1:-1, 1:-1] + rho0[:-2, 1:-1]) / dx**2
    drho_y = (rho0[1:-1, 2:] - 2 * rho0[1:-1, 1:-1] + rho0[1:-1, :-2]) / dy**2

    uxx = ux[1:-1, 1:-1] * (rho0[2:, 1:-1] - rho0[1:-1, 1:-1]) / dx
    uyy = uy[1:-1, 1:-1] * (rho0[1:-1, 2:] - rho0[1:-1, 1:-1]) / dy

    rho[1:-1, 1:-1] = rho0[1:-1, 1:-1] + dt * (D * (drho_x + drho_y) - (uxx + uyy))
    rho0 = np.copy(rho) 
    return rho0, rho

def mouse_draw():
    pygame.init()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    screen = pygame.display.set_mode((Ngrid, Ngrid), 0, 32)
    screen.fill(WHITE)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Export the drawing as a binary NumPy array before quitting
                binary_array = export_drawing_as_binary(screen)
                pygame.quit()
                return binary_array
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:  # Left mouse button down.
                    last = (event.pos[0]-event.rel[0], event.pos[1]-event.rel[1])
                    pygame.draw.line(screen, BLACK, last, event.pos, 1)

        pygame.display.update()
        clock.tick(30)  # Limit the frame rate to 30 FPS.

def export_drawing_as_binary(screen):
    """Convert the current screen to a 2D binary NumPy array."""
    # Convert the pygame surface to a NumPy array
    screen_array = pygame.surfarray.array3d(screen)
    
    # Convert the RGB array to grayscale
    grayscale = np.mean(screen_array, axis=2)
    
    # Convert grayscale to binary: pixels close to white are 0, others are 1
    binary_array = np.where(grayscale < 128, 1, 0)
    
    return binary_array.T  # Transpose to align with (width, height) orientation

def wavelength_to_rgb(wavelength, gamma=0.8):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return [int(R)/255, int(G)/255, int(B)/255]

def rho_to_rgb(rho):
    thickness = k*rho#k*np.ones_like(rho)*400# #Scaling from density of gasoline to its thickness
    lam_tab = wavelength(thickness, theta_nodes, theta_oil)*1e9
    for i in range(Ngrid):
        for j in range(Ngrid):
            w[i,j,:] =np.array(wavelength_to_rgb(lam_tab[i,j])) 
    return w    

# =============================================================================
# Physical parameters of the simulation 
# =============================================================================
np.random.seed(134543)
Ngrid = 750 #500 #Number of spatial grid points
x, dx = np.linspace(-1, 1, Ngrid, retstep = True) #Spatial coordinates and spacing 
y, dy = np.linspace(-1, 1, Ngrid, retstep = True)
X, Y = np.meshgrid(x,y)
index_y, index_x = np.ogrid[:Ngrid, :Ngrid]
D = 2 # Diffusion coefficient
rho0 = np.zeros((Ngrid,Ngrid))
rho = np.copy(rho0)

# =============================================================================
# Velocity field (swirl pattern)
# =============================================================================

velocity_field = "shear"
Umax = 5e2

if velocity_field == 'swirl':
    ux, uy = swirl(Umax, X, Y) #Swirl
elif velocity_field == 'shear':
    ux, uy = shear(0, Umax, Ngrid) #Shear in the x direction 

# =============================================================================
# Temporal information
# =============================================================================
dt = np.minimum(((dx*dy)**2 / (dx**2 + dy**2)) / (5*D), dy / np.max(uy)) 
tau = 2e-4 # Timescale of falling droplets
Nsteps = 100 #Number of temporal iteration
framerate = 100 #Number of iterations between each frames of simulation

# =============================================================================
# Initial conditions of gasoline layer 
# =============================================================================
method = "fixed_droplets"
drop = False #If we drop or not more droplets of oil on top of the initial conditions given by method 

if method == "layer":
    #Thin layer example
    n_modes = 10
    rho0 = thin_layer(10, X, Y)

elif method == "fixed_droplets":
    #Fixed droplets 
    x_drops = [100, 200, 300, 400]
    y_drops = [100, 200, 300, 400]
    amplitude = 400
    sigma = 30/500*Ngrid
    rho0 = fixed_droplets(rho0, x_drops, y_drops, amplitude, sigma, Ngrid, index_x, index_y )

elif method == "drawing":
    #Input based on a drawing given by the user
    index_x_mesh, index_y_mesh = np.meshgrid(index_x, index_y)  
    draw_arr = mouse_draw()
    bin_ind_x, bin_ind_y = np.where(draw_arr!=0)
    X_draw, Y_draw = index_x_mesh[bin_ind_x, bin_ind_y], index_y_mesh[bin_ind_x, bin_ind_y]
    plt.imshow(draw_arr)
    amplitude, sigma = 500,20
    for i in range(len(X_draw)):    
        rho = drop_shape(rho0, amplitude, sigma, X_draw[i], Y_draw[i], index_x, index_y)

    normalize = np.max(rho)/400
    rho /= normalize
    rho0 = np.copy(rho)

# =============================================================================
# Define the viewing angle for each node in the 2D array 
# =============================================================================
r_max = np.sqrt(2) #vecteur max = (1,1)
z = r_max/np.sqrt(3) # Arbitraire : ce parametre nous permet de jouer avec theta_max
theta_max = np.arctan(r_max/z)
radius = np.sqrt(X**2 + Y**2)
theta_nodes = np.arctan(radius/z)

# =============================================================================
# Calculate the wavelength with constructive interference depending on thickness and angle of nodes 
# =============================================================================
k = 0.015e-7 # Arbitrary: allows to scale the density to gasoline thickness 
n_oil = 1.442 # refractive index of oil (petrol, kerosene, etc)
theta_oil = np.arcsin(np.sin(theta_nodes)/n_oil)
wavelength = lambda d, theta_nodes, theta_oil : 2*d/np.cos(theta_oil) * (1 - np.sin(theta_oil) * np.sin(theta_nodes)) #Function that calculates the wavelength 
w = np.zeros((Ngrid,Ngrid,3)) #Array that will contain RGB values to map the wavelength to a matplotlib color 

# =============================================================================
# Run the simulation 
# =============================================================================

# Initialize the plot
fig, ax = plt.subplots(figsize = (6,6))
fig.canvas.draw()
im = ax.imshow(rho_to_rgb(rho0))
ax.axis("off")

# Temporal evolution loop
for t in tqdm(range(Nsteps)):
    
    if drop == True:
        r = np.random.random()
        if (r >= 0.2) and (t % np.ceil(tau/dt) == 0):
            amplitude = np.random.randint(300, 500)
            sigma = np.random.randint(Ngrid//25,Ngrid//10) 
            x0 = np.random.randint(1, Ngrid-1)
            y0 = np.random.randint(1, Ngrid-1)
            rho0 = drop_shape(rho0, amplitude, sigma, x0, y0, index_x, index_y)

    # Update the gasoline density for each time step 
    rho0, rho = simulation_timestep(rho0, rho, ux, uy)
    
    if t == 0 :
        plt.imshow(rho_to_rgb(rho)) # Show the initial diffaction pattern
    
    #Save the simulation frame each 10 iterations
    #if t % framerate == 0:
    if (t == 0) or (t == Nsteps-1):
        outfile = f"frames/frame0{t/framerate:04.0f}.png"
        #plt.imsave(outfile, rho, dpi=100, vmin = 0, vmax = 400)
        plt.imsave(outfile, rho_to_rgb(rho), dpi=100)

    