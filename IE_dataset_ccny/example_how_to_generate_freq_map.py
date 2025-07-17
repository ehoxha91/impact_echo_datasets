import numpy as np
import matplotlib.pyplot as plt


fx = np.load('IE_dataset_ccny/freq.npy')
freq = np.load('IE_dataset_ccny/freq_amplitude.npy')
max_freq_map = []
max_freq_map2 = []
for i in range(1178):
  freq[i,0:8]=0
  max_freq_map.append(fx[i,np.argmax(freq[i])])

for i in range(1178, 1824):
  freq[i,0:8]=0
  max_freq_map2.append(fx[i,np.argmax(freq[i])])

map = np.reshape(max_freq_map,(31, 38))
map = np.flip(map, axis=1)
im = plt.imshow(map, cmap='Spectral', interpolation='hamming')
im = plt.colorbar(im)
plt.axis('off')
plt.savefig("part1_freq.png", bbox_inches='tight')
plt.show()

map1 = np.reshape(max_freq_map2,(19, 34))
map1 = np.flip(map1, axis=1)
im = plt.imshow(map1, cmap='Spectral', interpolation='hamming')
im = plt.colorbar(im)
plt.axis('off')
plt.savefig("part2_freq.png", bbox_inches='tight')
plt.show() 