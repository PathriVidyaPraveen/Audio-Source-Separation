# Audio Source Separation using Digital Signal Processing

This project implements a purely signal-processing-based audio source separation system for separating two simultaneously active sources from a mono audio mixture. No machine learning models or pretrained networks are used.

---

## Project Structure

- **config.py**  
  Global configuration parameters (sampling rate, FFT size, hop length, etc.)

- **main.py**  
  Runs the complete source separation pipeline:
  - Loads mixture audio  
  - Computes STFT  
  - Estimates source energy maps  
  - Computes Wiener-style soft masks  
  - Reconstructs separated signals  
  - Saves outputs and visualizations  

- **io_utils.py**  
  Audio loading and saving utilities (mono conversion supported)

- **stft_utils.py**  
  Short-Time Fourier Transform (STFT) and inverse STFT utilities

- **band_tracking.py**  
  Adaptive frequency band tracking to estimate per-source energy maps

- **hpss.py**  
  Harmonic–Percussive energy estimation using median filtering

- **masking.py**  
  Wiener-style soft time–frequency masking

- **reconstruction.py**  
  Time-domain reconstruction from masked STFTs

- **visualization.py**  
  Visualization utilities for spectrograms and masks

- **p_sweep_analysis.py**  
  Ablation study for analyzing the effect of Wiener mask exponent *p*

- **evaluation.py**  
  Classical source separation evaluation metrics (SDR, SIR, SAR)

- **data/**  
  Input audio files:
  - `mixed.wav` — input mixture  
  - `clean1.wav` — reference source 1 (for evaluation)  
  - `clean2.wav` — reference source 2 (for evaluation)

- **outputs/**  
  Generated outputs:
  - `clean_1.wav`  
  - `clean_2.wav`  
  - `mixture_spectrogram.png`  
  - `soft_masks.png`  
  - `intensity_vs_p.png`  
  - `metrics_vs_p_source1.png`  
  - `metrics_vs_p_source2.png`

---

## Dependencies

- **Python**: 3.9 or higher  
- **Required libraries**:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `soundfile`

Install dependencies using:

```bash
pip install numpy scipy matplotlib soundfile

---

## How to Run

1. Place input audio files inside the `data/` directory.
2. Ensure parameters in `config.py` match the input sampling rate.
3. Run the main separation pipeline:

```bash
python main.py
How to Run

Place input audio files inside the data/ directory.

Ensure parameters in config.py match the input sampling rate.

Run the main separation pipeline:

python main.py


Separated audio files and visualizations will be saved to the outputs/ directory.

---

## Ablation Study

To analyze the effect of the Wiener mask exponent p, run:

python p_sweep_analysis.py


This generates plots showing:

Output signal intensity vs p

SDR vs p

SIR vs p

SAR vs p

---

## Final Notes

This project intentionally avoids machine learning to study the capabilities and limitations of classical DSP-based source separation.

The approach performs best when sources occupy partially distinct time–frequency regions.

Performance degrades for heavily overlapping mono speech–music mixtures, which is a known limitation of purely DSP-based methods.