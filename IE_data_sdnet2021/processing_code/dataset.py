import ie_sample
import numpy as np
import pandas as pd

class ie_dataset(object):

    def __init__(self, foler_path) -> None:
        self.foler_path = foler_path
        # create an ordered dictionary of ie_sample objects
        self.samples = {}
        self.ids = []
        self.read_folder()
        self.ids = np.sort(self.ids)
    
    def read_folder(self):
        import os
        for root, dirs, files in os.walk(self.foler_path):
            for file in files:
                if file.endswith(".lvm"):
                    ie_signal = ie_sample.ie_sample(float(1/float(9.765625E-6)), 
                                                    204800, 
                                                    'voltage',
                                                    os.path.join(root, file))
                    self.samples[ie_signal.id] = ie_signal
                    self.ids.append(ie_signal.id)  
            print('-------------------')
        return
    


if __name__ == "__main__":
    if True:
        dataset = ie_dataset('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/temp/')
        upsampled_dataset = np.array([])
        print(dataset.ids)
        for signal_id in dataset.ids:

            print(signal_id)
            dataset.samples.get(signal_id).plot_signal(time=False)
            # dataset.samples.get(signal_id).plot_upsampled_signal(time=False)
            upsampled_dataset = np.append(upsampled_dataset, dataset.samples.get(signal_id).signal_upsampled[0:2000])

        upsampled_dataset = upsampled_dataset.reshape((len(dataset.ids), 2000))
        np.save('/Users/evhoxha/Desktop/phd/IE_data_sdnet2021/prm_origin_1C.npy', upsampled_dataset)
    else:
        import matplotlib.pyplot as plt
        dataset = np.load('/Users/evhoxha/Desktop/phd/IE-dataset/forest_river_sb/frsb_origin_5C.npy')

        ie8 = ie_sample.ie_sample(float(1/float(9.765625E-6)), 
                                        204800, 
                                        'voltage',
                                        '/Users/evhoxha/Desktop/phd/IE-dataset/ie_dataset/raw_data/forest_river_sb/Origin_5C/Metal_116.lvm')
        ie8.plot_signal(time=False)
        dataset[95] = ie8.signal_upsampled[0:2000]
        plt.plot(dataset[95])
        plt.show()
        np.save('/Users/evhoxha/Desktop/phd/IE-dataset/forest_river_sb/frsb_origin_5C.npy', dataset)

