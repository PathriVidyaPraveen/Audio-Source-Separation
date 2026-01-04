import numpy as np
from scipy.ndimage import median_filter

def hpss_energy_maps(magnitude,harm_kernel=31,perc_kernel=31):
    # compute harmonic and percussive energy maps

    power = magnitude**2
    harmonic = median_filter(power,size=(1,harm_kernel))
    percussive = median_filter(power,size=(perc_kernel,1))

    return harmonic,percussive
