# impact_echo_datasets
This repository contains pre-processed impact-echo datasets that are very useful for learning-based Impact-Echo method.

### IE dataset CCNY

We own the authorship for this dataset. This dataset does not contain labels, because it was used only for qualitative evaluation, but one can easily generate the labels using schematics and other dataset information. This dataset contains 1824 IE datapoints, each signal contains `1024` samples, and sampling rate is `500 kHz`.

### IE data SDNET2021 Upsampled

Contains trimmed, and upsampled signals along their respective labels. For more information please check [SDNET2021](https://commons.und.edu/data/19/). We do not claim ownership for this dataset, neither claim that we collected them. However, we contributed to cleaning up, and trimming the data. This dataset is upsampled to 500kHz, to match one of our use-cases. 

- `IE_data_sdnet2021_500k_upsampled/sdnet_dataset.npy` contains all Impact-Echo signals, concatenated in the order:
 - `forest_river_sb (2A, 2B, 2C)`
 - `forest_river_nb (5A, 5B, 5C)`
 - `park_river_median (1A, 1B, 1C)`
 - `park_river_nb  (3A, 3B, 3C)`
 - `park_river_sb (4A, 4B, 4C, 4D)`

For more information about the meaning please read [SDNET2021](https://commons.und.edu/data/19/), plot individual ground truth for each dataset, check the `.dwg` schematics.

- `IE_data_sdnet2021_500k_upsampled/sdnet_labels.npy` contains the labels for `sdnet_dataset.npy`.


### IE data SDNET2021 Original

Contains trimmed IE signals along their respective labels. For more information please check [SDNET2021](https://commons.und.edu/data/19/). **IE data SDNET2021 Upsampled** has a different sampling rate (`102.4 kHz`) from upsampled dataset (`500 kHz`), additionally upsampled dataset contains only `2000` samples per signal, while this dataset has `10000` samples per signal. 

- Due to the size limit of github we did not generate the `sdnet_dataset.npy`, but we provide code to generate that. Run:
  - `python3 IE_data_sdnet2021_original/combine_dataset_partitions.py`
  - this will generate `IE_data_sdnet2021_original/sdnet_dataset.npy` and `IE_data_sdnet2021_original/sdnet_labels.npy`

### Code Examples 

#### CCNY 2022

```
import numpy as np

SR = 500000 # sampling rate

### Time domain
# Between each measurement in all direction there is a 2 inch distance

X_test = np.load('IE_dataset_ccny/time_amplitude.npy')
X_part1 = X_test[0:1178]    # part 1 is collected over a grid of 31x38 = 1178 signals, 
                            # with a distance of ~2.5 inch in between each measurement

X_part2 = X_test[1178:1824] # part 2 is collected over a grid of 19x34 = 646 signals
                            # with a distance of ~2.8 inch in between each measurement
                            # reason for difference in distance is different configuration
                            # during data collection


# load time axis 
t = np.load('IE_dataset_ccny/time.npy')[0] # its the same for all signals

### Frequency domain

# load frequency data
freq = np.load('IE_dataset_ccny/freq_amplitude.npy')
f = np.load('IE_dataset_ccny/freq.npy')[0] # its the same for all signals (0-60 kHz, if one needs higher frequency info please use fft...)

f_part1 = freq[0:1178]
f_part2 = freq[1178:1824]


### Plotting
import matplotlib.pyplot as plt

# plot a signal from part 1
plt.subplot(2,1,1)
plt.plot(t, X_part1[0])
plt.subplot(2,1,2)
plt.plot(f, f_part1[0])
plt.show()

# plot a signal from part 2
plt.subplot(2,1,1)
plt.plot(t[0], X_part2[0])
plt.subplot(2,1,2)
plt.plot(f[0], f_part2[0])
plt.show()
```

#### SDNET 2021

```
import numpy as np
import matplotlib.pyplot as plt

# plot first IE signal
dataset = np.load('IE_data_sdnet2021_original/sdnet_dataset.npy')
print(dataset.shape)
plt.plot(dataset[0])
plt.show()

# get the label for the first IE signal
labels = np.load('IE_data_sdnet2021_original/sdnet_labels.npy')
print(labels.shape)
print(f'label for first IE signal: {labels[0]}')

# load forest_river_nb/labels_frnb_2A.npy, reshape it to 11x11, flip it to match corresponding .dwg file and plot it
labels = np.load('IE_data_sdnet2021_original/forest_river_nb/labels_frnb_2A.npy')
labels = np.flip(labels)
labels = np.reshape(labels, (11, 11))
plt.imshow(labels, cmap='Spectral', interpolation='hamming')
plt.show()
```

### Cite

When using CCNY dataset, please cite the paper below, because this dataset is released with the paper. When using SDNET datasets, please cite the original [publication](https://commons.und.edu/data/19/) and our paper that is related to this work (pre-processing).

```
@ARTICLE{10168232,
  author={Hoxha, Ejup and Feng, Jinglun and Sanakov, Diar and Xiao, Jizhong},
  journal={IEEE Robotics and Automation Letters}, 
  title={Robotic Inspection and Subsurface Defect Mapping Using Impact-Echo and Ground Penetrating Radar}, 
  year={2023},
  volume={8},
  number={8},
  pages={4943-4950},
  doi={10.1109/LRA.2023.3290386}}
```

```
@article{HOXHA2025139829,
title = {Contrastive learning for robust defect mapping in concrete slabs using impact echo},
journal = {Construction and Building Materials},
volume = {461},
pages = {139829},
year = {2025},
issn = {0950-0618},
doi = {https://doi.org/10.1016/j.conbuildmat.2024.139829},
url = {https://www.sciencedirect.com/science/article/pii/S0950061824049717},
author = {Ejup Hoxha and Jinglun Feng and Agnimitra Sengupta and David Kirakosian and Yang He and Bo Shang and Ardian Gjinofci and Jizhong Xiao},
keywords = {Impact echo, Bridge decks, Contrastive learning, Concrete defects}
}
```
