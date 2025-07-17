import numpy as np

SR = 500000 # sampling rate

### Time domain
# Between each measurement in all direction there is a 2 inch distance

X_test = np.load('IE_dataset_ccny/time_amplitude_all.npy')
X_part1 = X_test[0:1178]    # part 1 is collected over a grid of 31x38 = 1178 signals, 
                            # with a distance of ~2.5 inch in between each measurement

X_part2 = X_test[1178:1824] # part 2 is collected over a grid of 19x34 = 646 signals
                            # with a distance of ~2.8 inch in between each measurement
                            # reason for difference in distance is different configuration
                            # during data collection


# load time axis 
t = np.load('IE_dataset_ccny/time_all.npy')[100] # its the same for all signals

### Frequency domain

# load frequency data
freq = np.load('IE_dataset_ccny/freq_amplitude.npy')
f = np.load('IE_dataset_ccny/freq.npy')[100] # its the same for all signals (0-60 kHz, if one needs higher frequency info please use fft...)

f_part1 = freq[0:1178]
f_part2 = freq[1178:1824]


### Plotting
import matplotlib.pyplot as plt

# plot a signal from part 1
plt.subplot(2,1,1)
plt.plot(t, X_part1[100])
plt.subplot(2,1,2)
plt.plot(f, f_part1[100])
plt.show()

# plot a signal from part 2
plt.subplot(2,1,1)
plt.plot(t, X_part2[100])
plt.subplot(2,1,2)
plt.plot(f, f_part2[100])
plt.show()