# waveclass

## About
A PyTorch implementation of classification of breaking waves from infra-red images using pre-trained Convolutional Neural Networks. Improved classification by taking optical flow from pairs of sequential images.

## Folders
- callbacks : contains functions used during training loop
- data : contains the infra-red images, optical flows and converted HDF5 files
- models : classes to load the pre-trained models and config-files
- scripts : bash scripts used for calculating optical flows and seperating data

## Run
`<addr> python main.py -c <config-file>`


Dataset source: https://github.com/dbuscombe-usgs/IR_waveclass.git
