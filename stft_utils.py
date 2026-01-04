import numpy as np
from scipy.signal import stft,istft,get_window


def compute_stft(signal,sample_rate,config):
    # compute short time fourier transform STFT of signal
    if signal.ndim!=1:
        raise ValueError("Audio signal must be a 1-D Numpy array")
    
    window = get_window(config.WINDOW_TYPE,config.FFT_SIZE,fftbins=True)
    frequencies,times,stft_matrix = stft(
        signal, 
        fs=sample_rate,  
        window=window,    
        nperseg=config.FFT_SIZE,
        noverlap=config.FFT_SIZE-config.HOP_LENGTH,
        boundary="zeros",
        padded=True
    )

    return stft_matrix,frequencies,times


def compute_istft(stft_matrix,sample_rate,config):
    # compute inverse short time fourier transform
    if not np.iscomplexobj(stft_matrix):
        raise ValueError("STFT matrix must be complex valued")

    window = get_window(config.WINDOW_TYPE,config.FFT_SIZE,fftbins=True)

    _,signal = istft(
        stft_matrix,
        fs=sample_rate,
        window=window,
        nperseg=config.FFT_SIZE,
        noverlap=config.FFT_SIZE-config.HOP_LENGTH,
        input_onesided=True,
        boundary=True
    )
    return signal
