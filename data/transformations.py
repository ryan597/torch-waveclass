"""Module for preprocessing HDF5 input images for CNNs. Can be used for PIL images if
the ToPILImage() transform is commented out"""

import torchvision.transforms as T


def get_normalization(base):
    """"Return the appropriate normalisation parameters for the base CNN,
    in the form [mean[R,G,B], std[R,G,B]]."""
    pretrain_transform = {
        #"resnet18" :   [[,,], [,,]],
        #"alexnet" :    [[,,], [,,]],
        "vgg16" :      [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        #"squeezenet" : [[,,], [,,]],
        #"densenet" :   [[,,], [,,]],
        #"inception" :  [[,,], [,,]],
        #"googlenet" :  [[,,], [,,]],
        "mobilenet" :  [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    }
    return pretrain_transform[base]

def get_transform(base, augment=False, image_shape=(224, 224)):
    """Return the transformations/augmentations for a given base CNN model."""
    mean, std = get_normalization(base)

    if augment:
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=15, fill=(0,)),
            T.RandomResizedCrop(size=image_shape, scale=(0.8, 1.0)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size=image_shape),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    return transform
