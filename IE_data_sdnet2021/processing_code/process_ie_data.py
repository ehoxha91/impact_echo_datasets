import pandas as pd 
import numpy as np


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


if not True: 
  dataset = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_sb/frsb_origin_5A.npy')
  print(dataset.shape)
  fft_slab1, _ = generate_fftbscan(dataset, sr = 500000, hp_cutoff=1000, lp_cutoff=20000)

  print(fft_slab1.shape)
  fft_slab1 = np.rot90(fft_slab1)
  # fft_slab1 = np.row_stack((fft_slab1, fft_slab1[-1])) # origin-B didn't have the last row

  import matplotlib.pyplot as plt


  map = []
  for f in fft_slab1:
    map.append(np.argmax(f))

  map = np.reshape(map, (11, 11))
  map = np.flip(map, axis=0)
  im = plt.imshow(map, cmap='Spectral', interpolation='hamming')
  plt.show()

  lbs = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_sb/labels_frsb_5A.npy')
  map = np.reshape(lbs, (11, 11))
  plt.imshow(map, cmap='Spectral', interpolation='hamming')
  plt.show()
else:
  dataset1 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_sb/frsb_origin_5A.npy')
  dataset2 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_sb/frsb_origin_5B.npy')
  dataset3 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_sb/frsb_origin_5C.npy')
  dataset4 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_nb/frnb_origin_2A.npy')
  dataset5 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_nb/frnb_origin_2B.npy')
  dataset6 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_nb/frnb_origin_2C.npy')
  dataset7 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_median/prm_origin_1A.npy')
  dataset8 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_median/prm_origin_1B.npy')
  dataset9 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_median/prm_origin_1C.npy')
  dataset10 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_nb/prnb_origin_3A.npy')
  dataset11 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_nb/prnb_origin_3B.npy')
  dataset12 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_nb/prnb_origin_3C.npy')
  dataset13 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_sb/prsb_origin_4A.npy')
  dataset14 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_sb/prsb_origin_4B.npy')
  dataset15 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_sb/prsb_origin_4C.npy')
  dataset16 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_sb/prsb_origin_4D.npy')

  labels1 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_sb/labels_frsb_5A.npy')
  labels2 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_sb/labels_frsb_5B.npy')
  labels3 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_sb/labels_frsb_5C.npy')
  labels4 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_nb/labels_frnb_2A.npy')
  labels5 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_nb/labels_frnb_2B.npy')
  labels6 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/forest_river_nb/labels_frnb_2C.npy')
  labels7 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_median/labels_prm_1A.npy')
  labels8 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_median/labels_prm_1B.npy')
  labels9 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_median/labels_prm_1C.npy')
  labels10 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_nb/labels_prnb_3A.npy')
  labels11 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_nb/labels_prnb_3B.npy')
  labels12 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_nb/labels_prnb_3C.npy')
  labels13 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_sb/labels_prsb_4A.npy')
  labels14 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_sb/labels_prsb_4B.npy')
  labels15 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_sb/labels_prsb_4C.npy')
  labels16 = np.load('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/park_river_sb/labels_prsb_4D.npy')

  # concatenate all matrices into one big matrix
  dataset = np.concatenate((dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11, dataset12, dataset13, dataset14, dataset15, dataset16), axis=0)
  labels = np.concatenate((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10, labels11, labels12, labels13, labels14, labels15, labels16), axis=0)
  print(dataset.shape, labels.shape)

  np.save('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/sdnet_dataset.npy', dataset)
  np.save('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/sdnet_labels.npy', labels)