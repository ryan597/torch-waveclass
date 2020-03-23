"""Module for preprocessing HDF5 input images for CNNs. Can be used for PIL images if
the ToPILImage() transform is commented out"""

import torchvision.transforms as T



# DEPRECATED : All pytorch pretrained models use the same normalisation
# See https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
#def get_normalization(base):
#    """"Return the appropriate normalisation parameters for the base CNN,
#    in the form [mean[R,G,B], std[R,G,B]]."""
#    pretrain_transform = {
#        #"resnet18" :   [[0.485, 0.456, 0.406], [0.299, 0.244, 0.255]],
#        #"alexnet" :    [[,,], [,,]],
#        "vgg16" :      [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
#        #"squeezenet" : [[,,], [,,]],
#        #"densenet" :   [[,,], [,,]],
#        #"inception" :  [[,,], [,,]],
#        #"googlenet" :  [[,,], [,,]],
#        "mobilenet" :  [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
#    }
#    return pretrain_transform[base]



def get_transform(augment=False, image_shape=(224, 224)):
    """Return the transformations/augmentations for a given base CNN model."""

    if augment:
        preprocess_transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=15, fill=(0,)),
            T.RandomResizedCrop(size=image_shape, scale=(0.8, 1.0))
        ])
    else:
        preprocess_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size=image_shape)
        ])

    pretrain_transform = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([preprocess_transform, pretrain_transform])
