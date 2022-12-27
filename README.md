# Detection of Crack Growth Using Acoustic Emission Signals
Project for MEM T680: Machine Learning and Data Analysis.
By David Austin, [daa97@drexel.edu](mailto:daa97@drexel.edu)

## Introduction
This project is created to determine the occurence of crack growth in metal samples using data from attached piezoelectric acoustic emissions (AE) sensors. Data from this project was provided by Drexel Theoretical and Applied Mechanics Group, and consisted of waveforms recorded during compact tensile testing of four aluminum plates.

The jupyter notebook contained here provides three major utilities:
1. It sets up the data structure for individual AE waveforms ('hits') and datasets containing many waveforms. The structure is designed to allow for flexibility in accessing metadata for many items or retrieving subsets of data. Data processing steps are included in this framework and allow for waveform properties to be calculated, saved, and quickly loaded in the future. 
2. It creates and trains a convolutional neural network in the form of an autoencoder built using Tensorflow. This machine learning model takes in hit spectrograms and attempts to reproduce them from a reduced latent space vector.
3. It extracts the latent space and reduces the dimensionality using UMAP. Data visualizations are included with additional calculated metrics to contextualize the feature space shown. This information can be used to separate hit waveforms relevant to crack growth.

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

Currently, the script is configured to read from four folders named Waveform1, Waveform2, Waveform3, and Waveform4, respectively.

Continue to run through each cell in the file through the end. A GPU with at least 4 GB of RAM is recommended for training the model.

