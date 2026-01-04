import numpy as np
from stft_utils import compute_istft

def reconstruct_sources(stft_matrix,mask_1,mask_2,sample_rate,config):
    # reconstruct time domain source signals from soft time frequency masks

    if not np.iscomplexobj(stft_matrix):
        raise ValueError("Input STFT matrix must be complex-valued")

    if stft_matrix.shape !=mask_1.shape or stft_matrix.shape != mask_2.shape:
        raise ValueError("STFT matrix and masks must have same shape")
    
    stft_source_1 = mask_1*stft_matrix
    stft_source_2 = mask_2*stft_matrix

    source_1 = compute_istft(stft_source_1,sample_rate,config)
    source_2 = compute_istft(stft_source_2,sample_rate,config)

    return source_1,source_2




