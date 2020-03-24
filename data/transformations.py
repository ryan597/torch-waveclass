"""Module for preprocessing HDF5 input images for CNNs. Can be used for PIL images if
the ToPILImage() transform is commented out"""

import torchvision.transforms as T

def get_transform(augment=False, image_shape=(224, 224)):
    """Return the transformations/augmentations for a given base CNN model."""
    # image shape must be a tuple, JSON stores it as a list
    image_shape = tuple(image_shape)
    if augment:
        preprocess_transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=5, fill=(0,)),
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
