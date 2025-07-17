import numpy as np
import matplotlib.pyplot as plt


def generate_fftbscan(data, sr = 200000, hp_cutoff=0, lp_cutoff=60000):
      def fft(signal):
          fourierTransform = np.fft.fft(signal)/len(signal)          
          fourierTransform = fourierTransform[range(int(len(signal)/2))]
          tpCount     = len(signal)
          values      = np.arange(int(tpCount/2))
          timePeriod  = tpCount/sr
          frequencies = values/timePeriod
          l = int(hp_cutoff/100)
          h = int(lp_cutoff/100)
          dt = abs(fourierTransform[l:h])
          dt_max = np.max(dt)
          dt_min = np.min(dt)
          return (dt-dt_min)/(dt_max-dt_min), frequencies[l:h]

      X = data.copy()
      X = (X-np.min(X))/(np.max(X)-np.min(X))

      # first signal so I can extend the matrix later on
      first_fft, _ = fft(X[0,:-1])
      fft_matrix = np.array(first_fft)
      for i in range(10-1):
        fft_matrix = np.column_stack((fft_matrix, first_fft))

      for i, sig in enumerate(X):
        if i == 0:
          continue
        normalized, _ = fft(sig[:-1])
        for i in range(10):
          fft_matrix = np.column_stack((fft_matrix, normalized))

      _, frequencies = fft(X[0,:-1])
      ax = plt.gca()
      yt = []
      yl = []
      for i in range(0, len(frequencies), 100):
          yt.append(i)
          yl.append(str(int(frequencies[i]/1000.0))+" kHz")
      ax.set_yticks(yt)
      ax.set_yticklabels(yl)
      plt.imshow(fft_matrix, interpolation='bilinear')
      plt.show()
      return


sr = 500000
X = np.load('IE_dataset_ccny/time_amplitude_all.npy')
# generate a B-scan of a line along horizontal data collection
# (high-pass cutoff 1kHz, low-pass cutoff 30kHz)

generate_fftbscan(X[0:38], sr = sr, hp_cutoff=1000, lp_cutoff=30000)