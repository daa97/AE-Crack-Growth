# MEMT680-Project
## Detection of Crack Growth Using Acoustic Emission Signals
David Austin, [daa97@drexel.edu](mailto:daa97@drexel.edu)

## Introduction
This project is a jupyter notebook framework to determine the occurence of crack growth through. This framework takes in acoustic emissions waveforms in text file format and converts them to spectrograms. These spectrograms are then used to train a convolutional autoencoder model. This model is then used to autoencode more waveforms. The resultant latent space is extracted and the dimensionality is reduced so that clustering patterns among waveforms can be observed and used to easily classify these waveforms.

## Implementation
### Environment Requirements
To run the code in this project, you will need the following installed (or equivalent versions):
1. Python 3.9.12
2. Jupyter 1.0.0
3. [SciPy 1.7.3](https://scipy.org/install/)
4. [UMAP 0.5.5](https://umap-learn.readthedocs.io/en/latest/index.html)
5. [Tensorflow 2.10.0](https://www.tensorflow.org/install)
6. [NumPy 1.21.5](https://numpy.org/install/)
7. [MatPlotLib 3.5.1](https://matplotlib.org/stable/users/getting_started/)

### Instructions
To run the project, open main.ipynb. Under the "Start" heading, add in the name(s) of the folders from which you would like to read text files. 

Currently, the script is configured to read from four folders named Waveform1, Waveform2, Waveform3, and Waveform4, respectively. Data files used in building this model are too large to share in this repository, and are not mine to share regardless. Ensure the waveforms contained in each file are 6144 datapoints long. 

Continue to run through each cell in the file through the end. A GPU with at least 4 GB of RAM is recommended for training the model.

