import numpy as np
import matplotlib.pyplot as plt


def fft(signal, sr=500000):
    fourierTransform = np.fft.fft(signal)/len(signal)          
    fourierTransform = fourierTransform[range(int(len(signal)/2))]
    tpCount     = len(signal)
    values      = np.arange(int(tpCount/2))
    timePeriod  = tpCount/sr
    frequencies = values/timePeriod
    dt = abs(fourierTransform)
    return np.array(dt), frequencies


X = np.load('IE_dataset_ccny/time_amplitude_all.npy')
fftsig, frequencies= fft(X[20])
plt.plot(frequencies[0:100]/1000.0, fftsig[0:100]/np.max(fftsig[0:100]))
plt.show()