# ablation study on wiener mask exponent p
import os
import numpy as np
import matplotlib.pyplot as plt
import config
from io_utils import load_audio
from stft_utils import compute_stft
from band_tracking import estimate_energy_maps
from masking import compute_soft_masks
from reconstruction import reconstruct_sources
from evaluation import compute_sdr,compute_sir,compute_sar

def signal_intensity(x):
    # compute RMS energy or intensity of signal
    return np.sqrt(np.mean(x**2))


def main():
    print("Mask Exponent (p) ablation study")
    mixture,sr = load_audio("data/mixed.wav")
    ref_1,_ = load_audio("data/clean1.wav")
    ref_2,_ = load_audio("data/clean2.wav")

    stft_matrix,_,_ = compute_stft(mixture,sr,config)
    magnitude = np.abs(stft_matrix)
    e1, e2 = estimate_energy_maps(magnitude)

    p_values = [0.5,1.0,1.5,2.0,2.5,3,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0]

    intensity_1 = []
    intensity_2 = []

    sdr_1,sir_1,sar_1 = [],[],[]
    sdr_2,sir_2,sar_2 = [],[],[]

    for p in p_values:
        print(f"Evaluating p = {p}")
        m1,m2 = compute_soft_masks(e1,e2,p=p,epsilon=config.EPSILON)
        src1,src2 = reconstruct_sources(stft_matrix,m1,m2,sr,config)

        intensity_1.append(signal_intensity(src1))
        intensity_2.append(signal_intensity(src2))

        sdr_1.append(compute_sdr(ref_1,src1))
        sir_1.append(compute_sir(ref_1,ref_2,src1))
        sar_1.append(compute_sar(ref_1,ref_2,src1))

        sdr_2.append(compute_sdr(ref_2,src2))
        sir_2.append(compute_sir(ref_2,ref_1,src2))
        sar_2.append(compute_sar(ref_2,ref_1,src2))

    # intensity vs p
    plt.figure(figsize=(8,5))
    plt.plot(p_values,intensity_1,marker="o",label="Source 1 Intensity")
    plt.plot(p_values,intensity_2,marker="o",label="Source 2 Intensity")
    plt.xscale("log")
    plt.xlabel("Mask Exponent p (log scale)")
    plt.ylabel("RMS Intensity")
    plt.title("Output Signal Intensity vs Mask Exponent")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/intensity_vs_p.png",dpi=300)
    plt.close()

    # metrics vs p for source 1
    plt.figure(figsize=(8,5))
    plt.plot(p_values,sdr_1,marker="o",label="SDR")
    plt.plot(p_values,sir_1,marker="o",label="SIR")
    plt.plot(p_values,sar_1,marker="o",label="SAR")
    plt.xscale("log")
    plt.xlabel("Mask Exponent p (log scale)")
    plt.ylabel("Metric Value (dB)")
    plt.title("Separation Metrics vs p (Source 1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/metrics_vs_p_source1.png",dpi=300)
    plt.close()

    # metrics vs p for source 2
    plt.figure(figsize=(8,5))
    plt.plot(p_values,sdr_2,marker="o",label="SDR")
    plt.plot(p_values,sir_2,marker="o",label="SIR")
    plt.plot(p_values,sar_2,marker="o",label="SAR")
    plt.xscale("log")
    plt.xlabel("Mask Exponent p (log scale)")
    plt.ylabel("Metric Value (dB)")
    plt.title("Separation Metrics vs p (Source 2)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/metrics_vs_p_source2.png",dpi=300)
    plt.close()

    print("Completed successfully")


if __name__ == "__main__":
    main()
