import os
import glob
import numpy as np
import h5py
from PIL import Image
import argparse

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='Give path to root directory containing images in class folder and path to store .h5 file')

    parser.add_argument('--i', metavar='DIR', type=str, nargs=1,
        help='Enter the path to the parent directory contraining the class folders. eg, IMGS/IR/train')

    parser.add_argument('--o', metavar='H5', type=str, nargs=1,
        help='Enter the path to the save the .h5 file in. eg. H5_files/train.h5')
    args = parser.parse_args()
    if args.i is None:
        raise Exception('ERROR : must give paths to [--i] image folder and [--h] .h5 save path')
    elif args.h is None:
        raise Exception('ERROR : Must give path to [--o] .h5 save path')


    root_dir = args.i[0]
    hdf5_dir   = args.h[0]

    image_size = (480, 640) # could change this to be from first image read
    print(f'\nImage size :\t{image_size}')
    
    class_paths = glob.glob(root_dir+'/*')
    classes = []
    for i in class_paths:
        label = i.split('/')[-1]
        classes.append(label)
    print(f'Classes found :\t{classes}')
    
    image_path = glob.glob(root_dir+"/*/*.jpg")
    print(f'Images found :\t{len(image_path)}')

    train_shape = (len(image_path), image_size[0], image_size[1])
    
    
    # open h5 datatset
    with h5py.File(hdf5_dir, 'w') as h5_images:
        h5_images.create_dataset('images', train_shape, dtype=np.uint8)
        h5_images.create_dataset('labels', (len(image_path), len(classes)), dtype=np.uint8)

        for i in range(len(image_path)):
            # get label
            label = image_path[i].split('/')[3]
            one_hot_label = (label==np.array(classes)).astype(int)
            # read img and label as uint8
            img = Image.open(image_path[i])
            img = np.array(img, dtype=np.uint8)
            # save to h5
            h5_images['labels'][i, ...] = one_hot_label[None]
            h5_images['images'][i, ...] = img[None]

    size_h5 = os.path.getsize(hdf5_dir)
    size_h5 *= 1./1e6 # convert to Mb
    print('H5 file size :\t\t%.3f Mb' % size_h5 )
