import soundfile as sf

x,sr = sf.read("data/mixed.wav")
print("Shape:", x.shape)
