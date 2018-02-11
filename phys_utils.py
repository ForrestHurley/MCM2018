##################################################################
#
# This will be a collection of different physical equations
# for modeling the behavior of radio waves in the ionosphere
#
##################################################################

import numpy as np
import sys
from pyiri2016 import IRI2016Profile

e = 1.6021e-19  # The elementary charge
m_e = 9.109e-31  # Mass of electron in kg
eps_0 = 8.85418782e-12  # Epsilon naught
KRe = 8497  # Effective earth radius in km
mu_0 = 1.25663706e-6
k_b = 1.3806485  # The Boltzmann constant


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

def critical_freq(N_max):
    # The frequency for total internal reflection for a normal incident wave
    return 9*N_max**0.5

def MUF(N_max, theta_i):  # Calculates the MUF based on maximum electron density
    # If, at a given height, we have density N_max, then all waves with
    # incident angle theta_i won't be reflected
    return critical_freq(N_max) * (1/(np.cos(theta_i)))

def min_density(eps, freq, theta_i):
    # Calculates minimum electron density for wave to be totally
    # internally reflected
    return ((np.cos(theta_i))**2 * freq**2) / 81

def max_skip(virtual_height):
    return 2*(2*KRe*virtual_height)**0.5

def skip_dist(theta_i, virtual_height):
    # Calculates the horizontal distance traversed by radio wave with known
    # virtual height and angle of attack
    return np.tan(theta_i)*virtual_height*2

def free_space_wave_number(wavelength):
    return 2*np.pi / wavelength

def propagation_constant(eps, k_0):
    return eps**0.5 * k_0

def attenuation_constant(eps, k_0):
    return np.real(propagation_constant(eps, k_0))

def attenuated_power(distance, alpha, old_power):
    # Computes the attenuated power after traveling through a medium
    # alpha = attenuation constant
    return old_power**2 / (alpha * distance)

def index_reflectance(theta_i, index=1.33):
    # Using Fresnel's equations, we calculate the percentage of light
    # reflected off the surface of water to find attenuation
    # assumes that signal is unpolarized
    # Assumes that the permeability of the material is close to mu naught
    cos_theta_t = ( 1 - ( (1/index) * np.sin(theta_i))**2 )**0.5
    R_s = abs( (np.cos(theta_i) - index*cos_theta_t) / (np.cos(theta_i) + index*cos_theta_t) )**2
    R_p = abs( (cos_theta_t - index*np.cos(theta_i)) / (cos_theta_t + index*np.cos(theta_t)) )**2
    return .5*(R_s + R_p)

def perm_reflectance(theta_i, eps_2, mu_2, eps_1=1.0006*eps_0, mu_1=1.256637e-6):
    # Still assumes that signal is unpolarized
    # Considers the magnetic properties --> mu !~ mu_0
    Z2 = (mu_2 / eps_2)**0.5
    Z1 = (mu_1 / eps_1)**0.5

    n1 = ((mu_1/mu_0) * (eps_1/eps_0))**.5
    n2 = ((mu_2/mu_0) * (eps_2/eps_0))**.5

    cos_t = ( 1 - ( (n1/n2) * np.sin(theta_i))**2 )**0.5
    cos_i = np.cos(theta_i)

    R_s = abs( (Z2*cos_i - Z1*cos_t) / (Z2*cos_i + Z1*cos_t) )**2
    R_p = abs( (Z2*cos_t - Z1*cos_i) / (Z2*cos_t + Z1*cos_i) )**2
    return .5*(R_s + R_p)

def electron_density(altitude, lat, lon, year, month, hour):
    altlim = [100., 1000.]
    altstp = 1.
    altitude = round(altitude)
    iri2016Obj = IRI2016Profile(altlim=altlim, altstp=altstp, lat=lat, lon=lon, year=year, month=month, hour=hour, option=1, verbose=False)
    altbins = np.arange(100., 1001., altstp)
    nalt = len(altbins)
    index = range(nalt)

    ne = iri2016Obj.a[0, index]

    return ne[altitude-100]

def electron_density_profile(lat, lon, year=2016, month=12, hour=12):
    altlim = [100., 1000.]
    altstp = 1.
    iri2016Obj = IRI2016Profile(altlim=altlim, altstp=altstp, lat=lat, lon=lon, year=year, month=month, hour=hour, option=1, verbose=False)
    altbins = np.arange(100., 1001., altstp)
    nalt = len(altbins)
    index = range(nalt)

    ne = iri2016Obj.a[0, index]

    return ne

def virtual_height(lat=0, lon=0, frequency=30e-3, theta_i=1,year=2000, month=12, hour=0):
    iri_data = IRI2016Profile( lat=lat, lon=lon, year=year, month=month, hour=hour, option=1, verbose=False)
    f_c = frequency*np.cos(theta_i)
    e_densities = electron_density_profile(lat, lon, year, month, hour)

    if frequency > MUF(max(e_densities), theta_i):
        print("Frequency greater than MUF")
        return 1000

    for height in range(100, 1000):
        if e_densities[height-100] > f_c**2/81:
            return height-1

    for height in range(100, 1000):
        if (1 - (81*e_densities[height-100]/frequency**2))**0.5 < 0:
            print("Bug in virtual height. Returning true height.")
            return height    

    return 1000


def data_virtual_height(layer, lat, lon, year, month, hour):
    iri2016Obj = IRI2016Profile(altlim=altlim, altstp=altstp, lat=lat, lon=lon, year=year, month=month, hour=hour, option=1, verbose=False)
    pass


def pressure(altitude):
    # Very rough estimate of pressure at altitude
    # which we use for a very rough estimate of collision frequency
    pass

for frequency in range(1, 1000):
    print(virtual_height(frequency=frequency*1e5))
