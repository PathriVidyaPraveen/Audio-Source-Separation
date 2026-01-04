SAMPLE_RATE = 44100 # sampling rate of input audio file Hz
FFT_SIZE = 2048 # no. of FFT points - larger values improve freq resolution but time resolution is not that good
HOP_LENGTH = 512 # no. of samples between successive frames
WINDOW_TYPE = "hann" # window fucntion to be applied before FFT

MASK_EXPONENT = 1.0 # standard wiener filtering
EPSILON = 1e-10 # numerical stability to avoid divide by 0
