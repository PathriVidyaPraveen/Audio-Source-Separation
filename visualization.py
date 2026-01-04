import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(magnitude_spectrogram,sample_rate,hop_length,title,output_path):
    # plot and save log magnitude spectrogram

    log_magnitude = 20.0*np.log10(magnitude_spectrogram+1e-10)

    num_freqs,num_frames = magnitude_spectrogram.shape
    time_axis = np.arange(num_frames)*hop_length /sample_rate
    freq_axis = np.linspace(0,sample_rate/2, num_freqs)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        log_magnitude,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[time_axis[0],time_axis[-1],freq_axis[0],freq_axis[-1]],
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path,dpi=300)
    plt.close()


def plot_masks(mask_1,mask_2,sample_rate,hop_length,title_1,title_2,output_path):
    # plot ans save time frequency masks for 2 sources
    num_freqs,num_frames = mask_1.shape
    time_axis = np.arange(num_frames)*hop_length /sample_rate
    freq_axis = np.linspace(0,sample_rate/2, num_freqs)

    fig,axes = plt.subplots(2,1,figsize=(10,6),sharex=True)

    im1 = axes[0].imshow(
        mask_1,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[time_axis[0],time_axis[-1],freq_axis[0],freq_axis[-1]],
    )
    axes[0].set_title(title_1)
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1,ax=axes[0],label="Mask Value")

    im2 = axes[1].imshow(
        mask_2,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[time_axis[0],time_axis[-1],freq_axis[0],freq_axis[-1]],
    )
    axes[1].set_title(title_2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im2,ax=axes[1],label="Mask Value")
    plt.tight_layout()
    plt.savefig(output_path,dpi=300)
    plt.close()
