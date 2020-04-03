import torchvision.transforms as T

def get_transform(augment=False, image_shape=(224, 224)):
    # image shape must be a tuple, JSON stores as a list
    image_shape = tuple(image_shape)

    if augment:
        preprocess_transform = T.Compose([
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop(size=image_shape, scale=(0.8, 1.0))
            #T.ColorJitter(0.3, 0.2, 0.2, 0.2)
        ])
    else:
        preprocess_transform = T.Compose([
            #T.ToPILImage(),
            T.Resize(size=image_shape)
        ])

    pretrain_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([preprocess_transform, pretrain_transform])


def get_transform_HDF5(augment=False, image_shape=(224, 224)):
    # image shape must be a tuple, JSON stores as a list
    image_shape = tuple(image_shape)

    if augment:
        preprocess_transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=15, fill=(0,)),
            T.RandomResizedCrop(size=image_shape, scale=(0.8, 1.0)),
            T.ColorJitter(0.1, 0.1)
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
