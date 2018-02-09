##################################################################
#
# This will be a collection of different physical equations
# for modeling the behavior of radio waves in the ionosphere
#
##################################################################

import numpy as np

e = 1.6021e-19 # The elementary charge
m_e = 9.109e-31  # Mass of electron in kg
eps_0 = 8.85418782e-12  # Epsilon naught

def refractive_index(mu, eps):
    # Returns the refractive index of a medium given permittivity and permeability
    return (mu * eps)**0.5

def refraction_angles(n, theta_i):
    # returns the angle of reflection and refraction
    refracted_angle = np.asin(np.sin(theta_i) * n)
    reflected_angle = np.pi - refracted_angle
    return [reflected_angle, refracted_angle]

def plasma_freq(N):  # N is the density of electrons. Returns w_p squared
    return (N*e**2)/(m_e * eps_0)

def dielectric(N, omega, v=None):
    # N = e density, omega = signal freq, v = collision frequency
    w_p = plasma_freq(N)
    
    if not v:  # If we consider collision frequency to be negligible
        return 1 - w_p/omega**2

    return 1 - w_p / (omega**2 * (1-v/omega*1j))


