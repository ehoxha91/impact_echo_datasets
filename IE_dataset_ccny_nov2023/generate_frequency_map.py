import matplotlib.pyplot as plt
import numpy as np

def generate_fftbscan(data, sr = 500000, hp_cutoff=0, lp_cutoff=60000):
      def fft(signal):
          fourierTransform = np.fft.fft(signal)/len(signal)
          fourierTransform = fourierTransform[range(int(len(signal)/2))]
          tpCount     = len(signal)
          values      = np.arange(int(tpCount/2))
          timePeriod  = tpCount/sr
          frequencies = values/timePeriod
          # print(frequencies)
          l = int(hp_cutoff/frequencies[1])
          h = int(lp_cutoff/frequencies[1])
          dt = abs(fourierTransform[l:h])
          dt_max = np.max(dt)
          dt_min = np.min(dt)
          return (dt-dt_min)/(dt_max-dt_min), frequencies[l:h]

      X = data.copy()
      X = (X-np.min(X))/(np.max(X)-np.min(X))

      # first signal so I can extend the matrix later on

      first_fft, frequencies_unnormalized = fft(X[0,:-1])
      frequencies_all = np.array(frequencies_unnormalized)
      fft_matrix = np.array(first_fft)
      for i, sig in enumerate(X):
        if i == 0:
          continue
        normalized, un_normalized = fft(sig[:-1])
        frequencies_all = np.column_stack((frequencies_all, un_normalized))
        fft_matrix = np.column_stack((fft_matrix, normalized))

      return fft_matrix, frequencies_all

X_train = np.load('IE_dataset_ccny_nov2023/ie_signals_nov2023.npy')

fft_slab1, _ = generate_fftbscan(X_train, sr = 500000, hp_cutoff=5000, lp_cutoff=20000)
fft_slab1 = np.rot90(fft_slab1)
map_s1 = []
freq_x = []
# fft_slab1 = np.rot90(fft_slab1)
# flip fft_slab1 verticallya
fft_slab1 = np.flip(fft_slab1, axis=0)

print(fft_slab1.shape)
i=0
map = []
for f in fft_slab1:
  map.append(np.argmax(f[1:]))

map = np.array(map)
print(map.shape)
map = np.reshape(map, (44, 34))
map = np.flip(map, axis=1)
im = plt.imshow(map, cmap='Spectral', interpolation='hamming')
# plot imbar
# cbar = plt.colorbar(im)
plt.axis('off')
# plt.title("Nov 2023 - Freq Map".format(i))
plt.savefig("frequency_map.png".format(i), dpi=2000)
plt.show()