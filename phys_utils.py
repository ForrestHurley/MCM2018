##################################################################
#
# This will be a collection of different physical equations
# for modeling the behavior of radio waves in the ionosphere
#
##################################################################

import numpy as np
import sys
from pyiri2016 import IRI2016Profile
import math
from PIL import Image
# import matplotlib.pyplot as plt

e = 1.6021e-19  # The elementary charge
m_e = 9.109e-31  # Mass of electron in kg
eps_0 = 8.85418782e-12  # Epsilon naught
Rem = 6371e3  # Effective earth radius in km
KRe=8497
temp= 6.4e4
bandwidth=27e6
mu_0 = 1.25663706e-6
k_b = 1.3806485  # The Boltzmann constant
k_br=1.3806485e-23

MODIS_DATA = Image.open('Datasets/Modis.tif')
MODIS_DATA = np.array(MODIS_DATA)

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

def propagation_constant(eps, wavelength):
    return eps**0.5 * free_space_wave_number(wavelength)

def attenuation_constant(eps, wavelength):
    return np.real(propagation_constant(eps, wavelength))

def attenuated_power(distance, alpha, old_power):
    # Computes the attenuated power after traveling through a medium
    # alpha = attenuation constant
    return old_power**2 / np.e**(alpha * distance)


def attenuated_power_db(distance, alpha):
    return alpha*distance*8.7


def index_reflectance(theta_i, index=1.33):
    # Using Fresnel's equations, we calculate the percentage of light
    # reflected off the surface of water to find attenuation
    # assumes that signal is unpolarized
    # Assumes that the permeability of the material is close to mu naught
    cos_theta_t = ( 1 - ( (1/index) * np.sin(theta_i))**2 )**0.5
    R_s = abs( (np.cos(theta_i) - index*cos_theta_t) / (np.cos(theta_i) + index*cos_theta_t) )**2
    R_p = abs( (cos_theta_t - index*np.cos(theta_i)) / (cos_theta_t + index*np.cos(theta_t)) )**2
    return .5*(R_s + R_p)


def index_reflectance_array(thetas, indices):
    length = len(thetas)
    one = np.ones(length)    

    cos_theta_t = ( one - ( (one/indices) * np.sin(thetas))**2 )**0.5
    R_s = abs( (np.cos(thetas) - indices*cos_theta_t) / (np.cos(thetas) + indices*cos_theta_t) )**2
    R_p = abs( (cos_theta_t - indices*np.cos(thetas)) / (cos_theta_t + indices*np.cos(thetas)) )**2
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


def water_empirical_index(salinity, temp_C=20, omega=1e6):
    # Pass in salinity (ppt) and temp (Celsius)
    conductivity = .18*salinity**.93*(1 + .02*(temp_C - 20))
    permittivity = 80  # Assumed value
    eps = permittivity - 1j * conductivity / (omega*eps_0)
    return eps**.5


def water_plasma_reflectance(salinity=35, theta_i=1, omega=1e6):
    # Provide salinity in ppt
    degree_of_ionization = salinity*2/(salinity*2 + 1000-salinity)
    
    # TODO Code below makes assumption that w_0 = 0, but if we observe magnetic effects, this isnt true
    
    # From from quasi neutrality, we know that average charge density ~ 1, so we define electron density:
    n_e = degree_of_ionization
    w_p = np.sqrt((n_e*e**2)/ (eps_0 * m_e))  # The plasma frequency
    eps = 1 - (w_p**2 / omega**2)
    ind = eps**.5
    return index_reflectance(theta_i, ind)

def water_plasma_index(salinity=35, omega=1e6):
    # Provide salinity in ppt
    degree_of_ionization = salinity*2/(salinity*2 + 1000-salinity)
    
    # TODO Code below makes assumption that w_0 = 0, but if we observe magnetic effects, this isnt true
    
    # From from quasi neutrality, we know that average charge density ~ 1, so we define electron density:
    n_e = degree_of_ionization
    w_p = np.sqrt((n_e*e**2)/ (eps_0 * m_e))  # The plasma frequency
    eps = 1 - (w_p**2 / omega**2)
    print(eps)
    return eps**.5

def earth_surface_reflectance(lat=0, lon=0, theta_i=1, omega=1e6):
    # Relative permeability assumed to be 1. Ground type is a number indexing surface types
    # Assume a positive time dependence
    eps = empirical_permittivity(lat, lon) - 1j * (empirical_conductivity(lat, lon)) / (omega*eps_0)
    ind = eps**.5
    return index_reflectance(theta_i, ind)


def earth_surface_index(lats, lons, omega=1e6):
    eps = empirical_permittivity(lats, lons) - 1j * (empirical_conductivity(lats, lons)) / (omega*eps_0)
    return eps**.5
    


def empirical_conductivity(lat=0, lon=0):
    # Cite the MODIS Data for this
    # Key-values: [0=Water, 1-5=Forest, 6-7=Shrublands, 8-9=Savannas, 10=Grasslands, 11=Wetlands, 12=Croplands, 13=Urban, 14=Cropland, 15=Snow, 16=Barren]    
    ground_type = get_ground_type(lat, lon)

    if ground_type == 254 or ground_type == 255:
        print("Error: Unclassified point in MODIS data set. Considering land to be barren.")
        ground_type = 16

    cond_estimates = [5, .02, .02, .02, .02, .02, .003, .003, .003, .003,  .003, .04, .01, .001, .01, .06, .001]

    return cond_estimates[ground_type]


def empirical_permittivity(lat=0, lon=0):
    # Cite the MODIS Data for this
    # Key-values: [0=Water, 1-5=Forest, 6-7=Shrublands, 8-9=Savannas, 10=Grasslands, 11=Wetlands, 12=Croplands, 13=Urban, 14=Cropland, 15=Snow, 16=Barren]    
    ground_type = get_ground_type(lat, lon)

    if ground_type == 254 or ground_type == 255:
        print("Error: Unclassified point in MODIS data set. Considering land to be barren.")
        ground_type = 16

    perm_estimates = [80, 30, 30, 30, 30, 30, 20, 20, 15, 15, 15, 15, 15, 3.5, 15, 4, 10]

    return perm_estimates[ground_type]


def get_ground_type(lat=0, lon=0):
    if lat < -64 or lat > 84:
        return 0
    
    im_x = math.floor((lon+180)*12)
    im_y = math.floor((lat-84)*-12)
    
    if im_y == 1776:
        im_y -= 1

    if im_y < 0:
        print("invalid lat. using a default")
        im_y = 0

    if im_y > 1775:
        print("invalid lat. using a default")
        im_y = 1775

    if im_x < 0:
        print("invalid lon. using a default")
        im_x = 0

    if im_x > 4319:
        print("invalid lon. using a default")
        im_x = 4319

    return MODIS_DATA[im_y, im_x]


def is_ground(lat, lon):
    return get_ground_type(lat, lon) != 0


def is_ground_array(lats, lons):
    rets = []
    for i in range(lats.size):
        print(lats)
        rets.append(is_ground_array(lats[i], lons[i]))

    return np.array(rets)


def loss_tangent(eps_r, eps_i, omega, conductivity):
    # Pass real and imaginary parts of eps and angular frequency and conductivity
    return np.atan((omega*eps_i + conductivity)/(omega*eps_r))


def loss_tangent_db(length, delta, wavelength):
    return delta*length*8.7*2*np.pi/wavelength


def penetration_depth(wavelength, eps_r, eps_i):
    coeff = wavelength/(2*np.pi)
    rest = ( 2/((eps_r**2 + eps_i**2)**.5 - eps_r) )**.5
    return coeff*rest


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
    altlim = [90., 1000.]
    altstp = 1.
    iri2016Obj = IRI2016Profile(altlim=altlim, altstp=altstp, lat=lat, lon=lon, year=year, month=month, hour=hour, option=1, verbose=False)
    altbins = np.arange(90., 1001., altstp)
    nalt = len(altbins)
    index = range(nalt)
    
    ne = iri2016Obj.a[0, index]
    
    return ne

def virtual_height(lat=0, lon=0, frequency=3e6, theta_i=1,year=2000, month=12, hour=0):
    iri_data = IRI2016Profile( lat=lat, lon=lon, year=year, month=month, hour=hour, option=1, verbose=False)
    f_c = frequency*np.cos(theta_i)
    e_densities = electron_density_profile(lat, lon, year, month, hour)
    muf = MUF(np.amax(e_densities), theta_i)
    
    if frequency > muf:
        print("Frequency greater than MUF")
        return 1000
    
    if frequency > 0.85 * muf:
        print("Frequency greater than OWF. Note that this can cause irregularities")
    
    for height in range(90, 1000):
        if e_densities[height-90] > f_c**2/81:
            return height-1

    for height in range(90, 1000):
        if (1 - (81*e_densities[height-100]/frequency**2))**0.5 < 0:
            print("Bug in virtual height. Returning true height.")
            return height

    return 1000


def refract_waves(rays, normals, theta_i, n_i=1, freq=1e6, altitude=90, lat=0, lon=0, month=12, year=2000, hour=12):
    # Add a function to refract waves for curved path model
    electron_densities = electron_density_profile(lat, lon, year, month, hour)  # Pull this out so we dont query every time
    electron_density = electron_densities[altitude-90]
    n = ( 1 - (81*electron_density/freq**2) )**.5
    theta_r = np.asin(n_i / n * np.sin(theta_i))
    # Rotate the vectors through an angle of theta_r - theta_i
    # TODO


def pressure(altitude):
    # Estimate for pressure at given altitude (altitude in km)
    # Returns P in mmHg
    
    #@source: Pressure data from US standard atmosphere on engineering toolbox
    heights = range(10, 85, 10)
    pressures = np.array([2.65, .5529, .1197, .0287, .007978, .002196, .00052, .00011])
    pressures = pressures*1e4/133.322
    
    for i in range(len(heights)):
        if heights[i] > altitude:
            return pressures[i-1]
    return pressures[7]

def free_space_loss(freq, distance):
    # Calculates the free space loss of waves as they travel a given distance through air
    # provide frequency in MHz and distance in meters
    # returns power loss in decibels
    return 27.6 + 20*np.log(freq)/np.log(10) + 20*np.log(distance)/np.log(10)


def D_layer_loss(freq=1e6, slice_sizes=10, thetas=1):
    # FOR TESTING PURPOSES, WE EXPECT FREQUENCIES BELOW 15MHz TO BE VIRTUALLY UNUSABLE
    # We can get around D layer absorption by sending signals with normal incidence
    estimated_N = [1, 10, 40, 300]
    eps_prev = 1
    total_db_loss = np.zeros(thetas.size)
    
    for altitude in range(60, 91, slice_sizes):
        # We need N, omega, v
        N_guess = estimated_N[(altitude-60)//10]
        v = 8.4e7 * pressure(altitude)
        eps = dielectric(N_guess, freq*2*np.pi, v)
        alpha = attenuation_constant(eps, 3e8/freq)
        thetas = np.arcsin(np.real(eps_prev)/np.real(eps) * np.sin(thetas))
        length = 1/(np.cos(thetas)) * slice_sizes * 1000
        total_db_loss += attenuated_power_db(length, alpha)
        eps_prev = eps
    return total_db_loss



def ionospheric_attenuation(frequency=1e6, theta_i=1, lat=0, lon=0, year=2000, month=12, hour=0, day=15):
    total_db_loss = 0
    if is_day(lat, lon, day, month, year, hour) == False:
        # Calculate attenuation in D-Layer
        total_db_loss += D_layer_loss(freq, 10, theta_i)
    
    # Calculate loss for other layers
    
    return total_db_loss


def calculate_time(in_day=1, in_month=12, in_year=2000, lat=0, long=0, is_rise=True):
    # @source: [http://williams.best.vwh.net/sunrise_sunset_algorithm.htm][2]
    # is_rise is a bool when it's true it indicates rise,
    # and if it's false it indicates setting time
    
    #set Zenith
    zenith = 96
    # offical      = 90 degrees 50'
    # civil        = 96 degrees
    # nautical     = 102 degrees
    # astronomical = 108 degrees
    
    
    #1- calculate the day of year
    n1 = math.floor( 275 * in_month / 9 )
    n2 = math.floor( ( in_month + 9 ) / 12 )
    n3 = ( 1 + math.floor( in_year - 4 * math.floor( in_year / 4 ) + 2 ) / 3 )
    
    new_day = n1 - ( n2 * n3 ) + in_day - 30
    
    #2- calculate rising / setting time
    if is_rise:
        rise_or_set_time = new_day + ( ( 6 - ( long / 15 ) ) / 24 )
    else:
        rise_or_set_time = new_day + ( ( 18 - ( long/ 15 ) ) / 24 )

    #3- calculate sun mean anamoly
    sun_mean_anomaly = ( 0.9856 * rise_or_set_time ) - 3.289
    
    #4 calculate true longitude
    true_long = (( sun_mean_anomaly + ( 1.916 * math.sin( math.radians( sun_mean_anomaly ) ) ) +( 0.020 * math.sin(  2 * math.radians( sun_mean_anomaly ) ) ) + 282.634 ) ) % 360
    
    #5 calculate s_r_a (sun_right_ascenstion)
    s_r_a = math.degrees( math.atan( 0.91764 * math.tan( math.radians( true_long ) ) ) ) % 360
    
    
    # s_r_a has to be in the same Quadrant as true_long
    true_long_quad = ( math.floor( true_long / 90 ) ) * 90
    s_r_a_quad = ( math.floor( s_r_a / 90 ) ) * 90
    s_r_a = s_r_a + ( true_long_quad - s_r_a_quad )
    
    # convert s_r_a to hours
    s_r_a = s_r_a / 15
    
    #6- calculate sun diclanation in terms of cos and sin
    sin_declanation = 0.39782 * math.sin( math.radians ( true_long ) )
    cos_declanation = math.cos( math.asin( sin_declanation ) )
    
    # sun local hour
    cos_hour = ( math.cos( math.radians( zenith ) ) - ( sin_declanation * math.sin( math.radians ( lat ) ) ) / ( cos_declanation * math.cos( math.radians ( lat ) ) ) )
        
    # extreme north / south
    if cos_hour > 1:
        #print("Sun never rises. Returning -1")
        return -1
    elif cos_hour < -1:
        #print("Sun never sets. Returning -2")
        return -2
    
    #7- sun/set local time calculations
    if is_rise:
        sun_local_hour =  ( 360 - math.degrees(math.acos( cos_hour ) ) ) / 15
    else:
        sun_local_hour = math.degrees( math.acos( cos_hour ) ) / 15

    sun_event_time = sun_local_hour + s_r_a - ( 0.06571 * rise_or_set_time ) - 6.622
    sun_event_time = sun_event_time % 24
    
    #final result
    time_in_utc =  (sun_event_time - ( long / 15 ))%24
    
    return time_in_utc


def is_day(lat=0, lon=0, day=1, month=12, year=2000, hour=12):
    sunrise = calculate_time(day, month, year, lat, lon, True)
    sunset = calculate_time(day, month, year, lat, lon, False) % 24

    if sunrise == -1 or sunset == -1:
        return False
    if sunset == -2 or sunrise == -2:
        return True

    if hour > sunrise and hour < sunset:
        return True
    elif sunrise > sunset and hour > sunrise:
        return True
    
    return False



if __name__ == "main":
    earth = np.zeros((180, 360))
    for lat in range(90, -90, -1):
        for lon in range(-180, 180):
            #print("Sunrise:", calculate_time(lat=lat, long=lon), "Sunset:",calculate_time(lat=lat, long=lon, is_rise=False))
            earth[-lat+90][lon+180] = is_day(lat, lon, hour=20, year=2018, month=2, day=11)
    #plt.imshow(earth)
