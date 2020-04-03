## Breaking Wave Classification

### About
A PyTorch implementation of classification of breaking waves from infra-red (IR) images using pre-trained Convolutional Neural Networks (CNNs).

It was found that just using the IR images gave poor results on the plunge wave due to the large class imbalance.
Augmentation of the images helped somewhat but there is still underwhelming results regarding the missclassifications.

Using optical flow its possible to increase the F1 score of the plunge class significantly as this incorporates dynamical
information into the features that are detected by the CNN. 

The TV-L1 optical flow method was used to calculate the optical flow, this was compared against a CNN based optical flow method SpyNet but the variational method TV-L1 gave much more accurate results.

### Folders
* callbacks : contains functions used during training loop
* data : contains the infra-red images, optical flows and converted HDF5 files
* models : classes to load the pre-trained models and config-files
* scripts : bash scripts used for calculating optical flows and seperating data

### Run
To begin training the model, open the terminal and enter

`python main.py -c <config-file>`


Dataset source: https://github.com/dbuscombe-usgs/IR_waveclass.git
