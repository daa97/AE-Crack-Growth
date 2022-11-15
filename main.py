import entropy
import read_file
import random


data = read_file.Dataset()
for j in range(10):
    data.add_hit([random.gauss(10, 1) for i in range(10)])

for hit in data.hits:
    hit.entropy = entropy.entropy(hit.waveform)

outliers_tf = entropy.outlier([h.entropy for h in data.hits])
for i in range(len(outliers_tf)):
    data.hits[i].ie_outlier = outliers_tf[i]

print([outliers_tf])




