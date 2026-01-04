import numpy as np

def safe_energy(x,epsilon=1e-10):
    # compute signal energy with numerical; stability
    return np.sum(x**2) + epsilon

def project(target,estimate):
    # orthogonal projection of estimate onto target
    scale = np.dot(estimate,target)/safe_energy(target)
    return scale*target

def compute_sdr(reference,estimate):
    # compute signal to distortion ratio
    reference = reference[:len(estimate)]
    estimate = estimate[:len(reference)]
    error = reference - estimate
    sdr = 10.0*np.log10(safe_energy(reference)/safe_energy(error))
    return sdr

def compute_sir(reference,interference,estimate):
    # compute signal to interference ratio
    reference = reference[:len(estimate)]
    interference = interference[:len(estimate)]
    estimate = estimate[:len(reference)]

    target_project = project(reference,estimate)
    interference_project = project(interference,estimate)

    sir = 10.0*np.log10(safe_energy(target_project)/safe_energy(interference_project))
    return sir

def compute_sar(reference,interference,estimate):
    # compute signal to artifacts ratio
    reference = reference[:len(estimate)]
    interference = interference[:len(estimate)]
    estimate = estimate[:len(reference)]

    target_project = project(reference,estimate)
    interference_project = project(interference,estimate)

    artifact = estimate - target_project- interference_project
    sar = 10*np.log10(safe_energy(target_project+interference_project)/safe_energy(artifact))
    return sar



