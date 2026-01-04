import numpy as np
import config
from io_utils import load_audio,save_audio
from stft_utils import compute_stft
from band_tracking import estimate_energy_maps
from masking import compute_soft_masks
from reconstruction import reconstruct_sources
from visualization import plot_spectrogram,plot_masks
from hpss import hpss_energy_maps




def main():
    print("Audio Source Separation using Digital Signal Processing")

    # load mixture audio
    print("Loading mixture audio...")
    mixture,sample_rate = load_audio("data/mixed.wav")

    if sample_rate !=config.SAMPLE_RATE:
        print(f"Input sample rate {sample_rate} Hz differs from config sample rate {config.SAMPLE_RATE} Hz")

    # compute STFT
    print("Computing STFT...")
    stft_matrix,freqs,times = compute_stft(mixture,sample_rate,config)
    magnitude = np.abs(stft_matrix)

    #adaptive freq band tracking
    print("Performing adaptive frequency band tracking...")
    # HPSS energy estimation
    E_harm,E_perc = hpss_energy_maps(magnitude)
    freqs = np.linspace(0,sample_rate/2,magnitude.shape[1])

    speech_band = (freqs >=80)&(freqs<=3500)

    E_perc[:,speech_band] *=1.3  
    E_harm[:,speech_band] *= 0.8 

    m1,m2 = compute_soft_masks(E_harm,E_perc,p=1.0,  epsilon=config.EPSILON)
    print("Mask means:", m1.mean(),m2.mean())
    print("Mask stds :", m1.std(),m2.std())


    # reconstruction
    source_1, source_2 = reconstruct_sources(stft_matrix,m1,m2,sample_rate,config)



    # compute wiener style soft masks
    # print("Computing Wiener style soft masks...")
    # m1,m2 = compute_soft_masks(e1,e2,p=config.MASK_EXPONENT,epsilon=config.EPSILON)

    # reconstruct separated signals
    print("Reconstructing time domain signals...")
    source_1,source_2 = reconstruct_sources(stft_matrix,m1,m2,sample_rate,config)

    #save audio outputs
    print("Saving separated audio files...")
    save_audio("outputs/clean_1.wav",source_1,sample_rate)
    save_audio("outputs/clean_2.wav",source_2,sample_rate)

    # generate visualizations
    print("Generating visualizations...")
    plot_spectrogram(magnitude,sample_rate,config.HOP_LENGTH,title="Mixture Log-Magnitude Spectrogram",output_path="outputs/mixture_spectrogram.png",)

    plot_masks(m1,m2,sample_rate,config.HOP_LENGTH,title_1="Soft mask of source 1",title_2="Soft Mask of source 2",output_path="outputs/soft_masks.png",)

    print("2 clean source signals separated successfully")



if __name__ == "__main__":
    main()
