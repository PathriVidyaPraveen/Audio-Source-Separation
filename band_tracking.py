import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def smooth_spectrum(magnitude_spectrum,sigma=2.0):
    # smooth 1d magnitude spectrum using gaussian filter
    return gaussian_filter1d(magnitude_spectrum,sigma=sigma)

def find_2_dominant_peaks(spectrum):
    # find indices of 2 most dominant spectral peaks
    peaks,_ = find_peaks(spectrum)
    if len(peaks)<2:
        mid = spectrum.size//2
        low_peak = np.argmax(spectrum[:mid])
        high_peak = np.argmax(spectrum[mid:]) + mid
        return low_peak,high_peak
    
    sorted_peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
    peak1,peak2 = sorted_peaks[:2]
    return tuple(sorted((peak1,peak2)))

def estimate_energy_maps(magnitude_spectrogram,smoothing_sigma=2.0):
    # estimate time freq energy gaps for 2 sources using adaptive freq band tracking
    if magnitude_spectrogram.ndim !=2:
        raise ValueError("Input magnitude spectrogram must be 2-d")
    
    num_freqs,num_frames = magnitude_spectrogram.shape
    e1 = np.zeros_like(magnitude_spectrogram,dtype=np.float64)
    e2 = np.zeros_like(magnitude_spectrogram,dtype=np.float64)

    freq_indices = np.arange(num_freqs)
    for t in range(num_frames):
        spectrum = magnitude_spectrogram[:,t]
        smoothed = smooth_spectrum(spectrum,sigma=smoothing_sigma)
        peak_low,peak_high = find_2_dominant_peaks(smoothed)
        dist_to_low = np.abs(freq_indices - peak_low)
        dist_to_high = np.abs(freq_indices - peak_high)
        mask_low = (dist_to_low <= dist_to_high)
        mask_high = ~mask_low
        e1[mask_low,t] = spectrum[mask_low]
        e2[mask_high,t] = spectrum[mask_high]

    return e1,e2





