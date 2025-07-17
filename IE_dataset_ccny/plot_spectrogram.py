import numpy as np
import matplotlib.pyplot as plt

fs = 500000 # sampling rate
X = np.load('IE_dataset_ccny/time_amplitude_all.npy')
signal_id = 401 # just choosing a random signal from the datasets

powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(X[signal_id], Fs=fs)
plt.axis('off')
plt.savefig('example_spectrogram.png', bbox_inches='tight', pad_inches=0)