import numpy as np
import soundfile as sf

def load_audio(path):
    # loads audio file and returns a mono signal
    signal,sample_rate = sf.read(path)
    if signal.ndim >1:
        signal = np.mean(signal,axis=1)
    signal = signal.astype(np.float32)
    return signal,sample_rate

def save_audio(path,signal,sample_rate):
    # saves mono audio signal to file

    if signal.ndim!=1:
        raise ValueError("Audio signal must be a 1-D Numpy array")
    
    sf.write(path,signal,sample_rate)


