import numpy as np

class ie_sample(object):

    def __init__(self, sample_rate, sample_length, sample_type, filename, trim_signal=True):
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.sample_type = sample_type 
        self.signal = np.zeros((sample_length, 1))
        self.time = np.zeros((sample_length, 1))
        self.read_file(filename)
        self.id = int(filename.split('/')[-1].split('.')[0].replace('Metal_', ''))
        if trim_signal:
            self.time, self.signal = self.get_signal(0.5)
        self.time_upsampled, self.signal_upsampled = self.upsample_signal()
    
    def read_file(self, filename):
        with open(filename, 'r') as f:
            data = f.readlines()
            for i, line in enumerate(data[23:]):
                line = line.replace('\n', '')
                if line == '':
                    break
                arr = line.split('\t')
                self.time[i] = (float(arr[0]))
                self.signal[i] = (float(arr[1].replace('\n', '')))
        return
    
    def plot_signal(self, time=True):
        import matplotlib.pyplot as plt
        if time:
            plt.plot(self.time, self.signal)
        else:
            plt.plot(self.signal)
        plt.show()
        return
    
    def get_signal(self, threshold=1):
        for i in range(self.signal.shape[0]):
            if np.abs(self.signal[i]) > threshold:
                index = i-10
                if index < 0:
                    index = 0
                return self.time[index:], self.signal[index:]
        return self.time, self.signal
    
    def upsample_signal(self, sample_rate=500000):
        from scipy.interpolate import interp1d
        f = interp1d(self.time.flatten(), self.signal.flatten(), fill_value="extrapolate")
        new_time = np.arange(self.time[0], self.time[-1], 1/sample_rate)
        new_signal = f(new_time)
        return new_time, new_signal
    
    def plot_upsampled_signal(self, time=False):
        import matplotlib.pyplot as plt

        _, axs = plt.subplots(4)
        if time:
            axs[0].plot(self.time, self.signal)
            axs[0].set_title('Original signal')
            axs[1].plot(self.time_upsampled, self.signal_upsampled)
            axs[1].set_title('Upsampled signal')
        else:
            axs[0].plot(self.signal)
            axs[0].set_title('Original signal')
            amp, freq = self.fft(self.signal.flatten(), self.sample_rate)
            axs[2].plot(freq, amp)
            axs[2].set_title('FFT of original signal')

            axs[1].plot(self.signal_upsampled)
            axs[1].set_title('Upsampled signal')
            amp, freq = self.fft(self.signal_upsampled.flatten(), 500000)
            axs[3].plot(freq, amp)
            axs[3].set_title('FFT of upsampled signal')

        plt.show()
        return
    
    def downsample_signal(self):

        from scipy.interpolate import interp1d
        f = interp1d(self.time_upsampled.flatten(), self.signal_upsampled.flatten(), fill_value="extrapolate")
        new_time = np.arange(self.time_upsampled[0], self.time_upsampled[-1], 1/self.sample_rate)
        new_signal = f(new_time)
        return new_time, new_signal
    
    def plot_downsampled_signal(self, time=False):
        import matplotlib.pyplot as plt
        new_time, new_signal = self.downsample_signal()
        _, axs = plt.subplots(4)
        if time:
            axs[0].plot(self.time_upsampled, self.signal_upsampled)
            axs[0].set_title('Original signal')
            axs[1].plot(new_time, new_signal)
            axs[1].set_title('Downsampled signal')
        else:
            axs[0].plot(self.signal_upsampled)
            axs[0].set_title('Original signal')
            amp, freq = self.fft(self.signal_upsampled.flatten(), 500000)
            axs[2].plot(freq, amp)
            axs[2].set_title('FFT of original signal')

            axs[1].plot(new_signal)
            axs[1].set_title('Downsampled signal')
            amp, freq = self.fft(new_signal.flatten(), self.sample_rate)
            axs[3].plot(freq, amp)
            axs[3].set_title('FFT of downsampled signal')

        plt.show()
        return

    
    def fft(self, signal, sr):
        fourierTransform = np.fft.fft(signal)/len(signal)
        fourierTransform = fourierTransform[range(int(len(signal)/2))]
        tpCount     = len(signal)
        values      = np.arange(int(tpCount/2))
        timePeriod  = tpCount/sr
        frequencies = values/timePeriod
        dt = abs(fourierTransform)
        dt_max = np.max(dt)
        dt_min = np.min(dt)
        return (dt-dt_min)/(dt_max-dt_min), frequencies
    
