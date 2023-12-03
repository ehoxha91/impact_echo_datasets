import numpy as np
import matplotlib.pyplot as plt

# plot first IE signal
dataset = np.load('IE_data_sdnet2021_original/sdnet_dataset.npy')
print(dataset.shape)
plt.plot(dataset[0])
plt.show()

# get the label for the first IE signal
labels = np.load('IE_data_sdnet2021_original/sdnet_labels.npy')[0:121]
labels = np.reshape(labels, (11, 11))
plt.imshow(labels, cmap='Spectral', interpolation='hamming')
plt.show()