import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# origin_A 0:121
# origin_B 121:241
# origin_C 241:362
# [0, 121], [121, 242],
list = [[0, 121], [121, 242], [242, 363]]

for range in list:

    df = pd.read_csv('/Users/evhoxha/Desktop/phd/IE-dataset/ie_dataset/annotation/forest_river_sb.csv', header=None)[range[0]:range[1]]
    labels = df[10]
    labels = labels.astype(int).to_numpy()
    # labels[labels==1] = 0
    # labels[labels==2] = 1
    # labels[labels==3] = 1

    np.save(f'/Users/evhoxha/Desktop/phd/IE-dataset/forest_river_nb/labels_frsb_{df[6].loc[range[0]]}.npy', labels)
    
    labels = np.flip(labels)
    img = np.reshape(labels, (11, 11))
    img = np.flip(img, axis=1)
    # print(img)
    plt.imshow(img, cmap='Spectral', interpolation='hamming')
    plt.title(df[6].loc[range[0]])
    plt.show()