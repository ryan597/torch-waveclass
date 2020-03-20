"""Script to convert images in a root directory to h5 file
format with the respective labels. The root directory must
contain folders with the class names, these will act as the
labels
"""
import os
import glob
import argparse
import numpy as np
import h5py
from PIL import Image


def find_classes(directory):
    class_paths = glob.glob(directory+"/*")
    classes = []
    for i in class_paths:
        label = i.split('/')[-1]
        classes.append(label)
    print(f"classes found : \t{classes}")
    classes = np.array(classes)
    return classes
    
def average_shape(image_path):
    avg_w = 0
    avg_h = 0
    # Calculate the average sizes for resize
    for i in image_path:
        ith_image = Image.open(i)
        image_shape = np.shape(np.array(ith_image))
        #print(f"image shape :\t{image_shape}")
        avg_w += image_shape[0]
        avg_h += image_shape[1]

    avg_w = int(avg_w/len(image_path))
    avg_h = int(avg_h/len(image_path))

    print(f"average width :\t{avg_w}")
    print(f"average height:\t{avg_h}")
    image_shape = (avg_w, avg_h)
    return image_shape

def img_to_h5(directory, h5file, image_shape=None):
    print(f"search {directory} \nsave to {h5file}")
    classes = find_classes(directory)
    
    image_path = glob.glob(directory+'/*/*.jpeg')
    len_images = len(image_path)
    print(f"images found :\t{len_images}")

    if image_shape[0] is None:
        print("image shape not supplied, calculating average width and heigth")
        image_shape = average_shape(image_path)
   
    h5_shape = (len_images, (image_shape))
    
    with h5py.File(h5file, 'w') as h5_images:
        h5_images.create_dataset('images', h5_shape, dtype=np.uint8)
        h5_images.create_dataset('labels', (len_images, ), dtype=np.uint8)
          
        for i in range(len_images):
            # get label
            label = image_path[i].split('/')[-2]
            label = np.argmax(label==classes)
            # read img and label as uint8
            img = Image.open(image_path[i])
            img = img.resize(image_shape).convert('L')
            img = np.array(img, dtype=np.uint8)
            # save to h5
            h5_images['labels'][i] = label
            h5_images['images'][i, ...] = img[None]
    size_h5 = os.path.getsize(h5file)
    size_h5 *= 1./1e6
    print(f"H5 file size : \t\t{size_h5:.3f}")
          
    return image_shape


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Give path to root directory containing images in class folder and path to store .h5 file')

    parser.add_argument('--i', metavar='DIR', type=str, nargs=1,
        help='Enter the path to the parent directory contraining the class folders. eg, IMGS/IR/train')

    parser.add_argument('--o', metavar='H5', type=str, nargs=1,
        help='Enter the path to the save the .h5 file in. eg. H5_files/train.h5')
    
    parser.add_argument('--s', metavar='SHAPE', type=int, nargs=2,
        help='Enter image shape as <W H> or to calculate average shape <0 0>')

    args = parser.parse_args()
    
    if args.i is None:
        raise Exception('ERROR : must give paths to [--i] image folder and [--h] .h5 save path')
    elif args.o is None:
        raise Exception('ERROR : Must give path to [--o] .h5 save path')
    elif args.s is None:
        raise Exception("ERROR : Specify image size [--s]<W H> or to calculate average size [--s]<0 0>")

    root_dir = args.i[0]
    hdf5_dir   = args.o[0]
    if args.s[0]==0:
        image_shape==None
    else:
        image_shape = (args.s[0], args.s[1])

    img_to_h5(root_dir, hdf5_dir, image_shape)
