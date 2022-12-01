# import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

def entropy(waveform):
    dist = PMF(waveform)
    ent = 0
    for prob in dist:
        # print("PROB:", prob)
        if prob != 0:
            info = math.log(prob, 2)
        else:
            info = 0
        # print("INFO:", info)
        ent += - prob * info
    return ent


def PMF(waveform):
    hist = histogram(waveform)
    return [count / sum(hist) for count in hist]


def histogram(waveform):
    heights = sorted([abs(pt) for pt in waveform])
    n = len(heights)
    (Q1, Q2, Q3) = quartiles(heights)
    IQR = Q3 - Q1
    data_range = heights[-1] - heights[0]
    bin_width = 2 * IQR / (n ** (1 / 3))
    num_bins = math.ceil(data_range / bin_width)
    bins = []
    # Loop through each bin in the histogram
    for i in range(num_bins):
        bins.append(0)
        # Loop through each data point to see if it should be added to the bin
        for h in heights:
            # Most bins are inclusive only at lower bound: [x0, x1)
            if bin_width * i <= h < bin_width * (i + 1):
                bins[-1] += 1
            # Last bin is inclusive at both bounds: [x1, x2]
            elif i == num_bins - 1 and h == bin_width * (i + 1):
                bins[-1] += 1
    return bins


def quartiles(data):
    d = sorted(data)
    L = len(d)
    if L % 4 == 0:
        Q1 = ind2(d, L // 4)
        Q2 = ind2(d, L // 2)
        Q3 = ind2(d, (L // 4) * 3)
    elif L % 4 == 1:
        Q1 = ind2(d, L // 4)
        Q2 = d[L // 2]
        Q3 = ind2(d, (L // 4) * 3 + 1)
    elif L % 4 == 2:
        Q1 = d[L // 4]
        Q2 = ind2(d, L // 2)
        Q3 = d[(L // 4) * 3 + 1]
    elif L % 4 == 3:
        Q1 = d[L // 4]
        Q2 = d[L // 2]
        Q3 = d[(L // 4) * 3 + 2]
    return Q1, Q2, Q3


# averages two elements from a list with indices of index2 and index2-1 (for taking medians when # of elements is even)
def ind2(lis, index2):
    return (lis[index2 - 1] + lis[index2]) / 2


def kurtosis(data):
    mu = avg(data)
    D = [point - mu for point in data]

    numerator = avg([d ** 4 for d in D])
    denominator = avg([d ** 2 for d in D]) ** 2
    return numerator / denominator


def avg(l):
    return sum(l) / len(l)


def outlier(data):
    Q1, _, Q3 = quartiles(data)
    threshold = Q3 + (Q3 - Q1)
    return [pt > threshold for pt in data]


waves = np.load("waves.npy")

print(entropy(waves[0,:]))