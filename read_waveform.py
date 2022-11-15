import numpy as np
import entropy
import read_file as rf
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
print("starting")
folder = "Waveform1"
test = rf.Dataset()
test.add_hits(folder)
print("imported")
#plt.plot(test.hits[103].waveform)
#plt.plot(test.hits[104].waveform)
plt.plot(test.hits[105].waveform)
plt.show()


'''
for h in test.hits:
    (_,_,img) = rf.spect(h.waveform, 120, sample_rate=1)
    # print(len(img), "|", len(img[0]))
    h.spectrogram = img
    h.entropy = entropy.entropy(h.waveform)
print("spectrogrammed")
w = np.array([h.spectrogram for h in test.hits])
ie = np.array([h.entropy for h in test.hits])
y = np.array([np.array(o+1) for o in entropy.outlier(ie)])

print()
print(y.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(61, 58, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))


model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(x=w, y=y, epochs=10)


'''
