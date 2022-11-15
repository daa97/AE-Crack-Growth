import os
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np

class Dataset:
    def __init__(self):
        self.hits = []
    def add_hit(self, fname):
        hit = Hit()
        hit.read_file(fname)
        self.hits.append(hit)
        return self.hits[-1]
    def add_hits(self, folder):
        for file in os.listdir(folder):
            f = os.path.join(folder, file)
            self.add_hit(f)

class Hit:
    def __init__(self, fname=None, waveform=None):
        self.file_name = fname
        self.start_time = None
        self.waveform = waveform
        self.entropy = None
        self.ie_outlier = None
        self.sample_interval = None
        self.spectrogram = None


    def read_file(self, fname=None):
        if fname is not None:
            self.file_name = fname
        with open(self.file_name, mode='r') as f:
            txt = f.readlines()
        for ind, line in enumerate(txt):
            if "TIME OF TEST:" in line:
                self.start_time = float(line.split(": ")[1])
            if "SAMPLE INTERVAL (Seconds):" in line:
                self.sample_interval = float(line.split(": ")[1])
            if line=="\n":
                self.waveform = []
                break
        else:
            raise ValueError("Expected blank line not found in waveform datafile")
        for line in txt[ind+1:]:
            self.waveform.append(float(line))
def spect(wave, binsize, sample_rate=1, plot=False):
    #print(binsize)
    output = sig.spectrogram(np.array(wave), nperseg=binsize, fs=sample_rate)
    freq_axis = output[0]
    time_axis = output[1]
    intensity = output[2]
    #x=1
    #intensity = [[j*x for j in i] for i in intensity]
    #print(max([max(i) for i in intensity]))
    if plot:
        ext = (time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1])
        stretch = (time_axis[-1]-time_axis[0]) / (freq_axis[-1]-freq_axis[0])
        plt.imshow(intensity, extent=ext, aspect=0.8*stretch, origin="lower")
        plt.show()
    return (freq_axis, time_axis, intensity)




