################################################################################
# Written by Ryan Smith    |    ryan.smith@ucdconnect.ie    |   March 2020
# University College Dublin|    github.com:ryan597/waveclass.git

################################################################################
################################################################################
#                   TO-DO:
# Refactor code (functions to own file, call with params)
# Generalise to both IR and Flow
# Automatic calculation of class weights
# Testing of different arch. & hyper. & augmentations
# Adding plotting of training history (real time updates)
# Implememnt early stopping
# Have some coffee
# Refactor again...
################################################################################
import argparse
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import classification_report
################################################################################

class H5Dataset(Dataset):
    """Dataset of image and label data in HDF5 format.

    Args:
        path (string) : path to the HDF5 file for train or test
        transforms (callable, optional): A transforms.Compose([...]) callable
            to transform or augment the images with before the appropriate
            transforms for the pretrained base model
        pretrain_transforms ()
    """
    def __init__(self, path, transform=None):
        self.file_path = path
        self.images = None
        self.labels = None
        self.transform = transform

        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file['images'])

    def __getitem__(self, index):
        if self.images is None:
            self.images = h5py.File(self.file_path, 'r')['images']
            self.labels = h5py.File(self.file_path, 'r')['labels']

        image = self.images[index]
        image = self.transform(image).numpy()[0]
        # Grayscale to RGB for the pretrain normalize to work
        # Try: save H5 files as RGB, storage will be 3x, but test speed up in loading
        image = torch.from_numpy(np.array([image, image, image]))
        image = pretrain_transform(image)
        # loss expects type long
        class_label = np.long(self.labels[index])
        return (image, class_label)

    def __len__(self):
        return self.dataset_len

class Net(nn.Module):
    """Convolutional Neural Network model for the classification of input images
    Args:
        x (tensor): inputs into the CNN, must be a torch.Tensor with shape
            [B, 3, ...] with B the batch size, with 3 channels and arbitray image
            input size ...
        PLACEHOLDER
    Returns:
        x (tensor): outputs from the CNN forward pass as a torch.Tensor with
            dimension [B, C], where B is the batch size and C is the number of
            classes (3).
    """
    def __init__(self):
        super(Net, self).__init__()
        # load the pretrained model
        pretrain = models.mobilenet_v2(pretrained=True)
        for param in pretrain.features.parameters():
            param.requires_grad = False
        self.model = pretrain
        # remove the top from pretrained and replace with custom fully connected
        self.model.classifier = nn.Dropout(0.5)
        self.norm1 = nn.BatchNorm1d(1280)
        self.fc1 = nn.Linear(1280, 10)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu6(self.model(x))
        x = self.norm1(x)
        x = F.relu6(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x

def h5_dataloader(h5_filepath, transform=None, batch_size=32, shuffle=False, num_workers=2):
    """Function to return a torch.utils.DataLoader for the HDF5 files containing
    input images.
    Args:
        h5_filepath (string): Path to the HDF5 file to be read in eg. H5_files/train.h5
        transform (callable, optional): Transformation or augmentations to apply
            to the input dataset. Should be given as torch.transforms.Compose([...]).
            These transforms will be applied before the pretrained models appropriate
            transforms. Default : None
        batch_size (int, optional): Batch size for the dataset, Default : 32
        shuffle (boolean, optional): To shuffle the dataset or not. Recommended for
            train dataset. Default : False
        num_workers (int, optional): The number of processes which Dataloader will run
            to load the batches. Default : 4
    Returns:
        dataloader : an instance of torch.utils.Dataloader for loading the HDF5 dataset
            in batches.
    """
    dataset = H5Dataset(h5_filepath, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader

"""
def image_data(root_dir, image_shape=(96, 96), augmentations=None, batch_size=1, shuffle=False):
    ""
    Args:
        root_dir (string): path to the class folders
        image_size (int, int): Resize all the images to this.
        augmentations (callable, optional): transformations to randomly apply to images, default is None
        batch_size (int): number of images in each batch, default is 1
        shuffle (bool): to shuffle the dataset or not, default is false
    Returns:
        A batched labeled image dataloader from the root_dir with applied augmentations.
    ""
    if augmentations is None:
        data_transform = transforms.Compose([
            transforms.Resize(size=image_shape),
            transforms.ToTensor(),
            # normalize between 0 and 1
            transforms.Normalize((0, 0, 0), ((255, 255, 255))),
            # normalize for mobilenet_v2
            transforms.Normalize((0.485, 0.456, 0.406), ((0.229, 0.224, 0.225)))
        ])
    else:
        data_transform = augmentations

    dataset = datasets.ImageFolder(root=root_dir,
                                    transform=data_transform)
    data_loader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=4)

    # implement dataset._find_classes() and return
    # get number of files in each class and return
    return data_loader
"""

def view_img_batch(data_loader):
    """View 25 images sampled from a batch from a DataLoader
    Args:
        data_loader: a torch.utils.data.DataLoader for an image dataset
    Returns:
        matplotlib.pyplot figure with 25 sample images displayed with respective
        labels as the titles on the plots.
    """
    plt.figure(figsize=(10,10))
    #data = iter(data_loader)
    for batch in data_loader:
        img_batch, label_batch = batch
        for i, img in enumerate(img_batch):
            # channels last
            img = img.permute(1, 2, 0)
            label = label_batch[i]
            plt.subplot(3, 3, i+1)
            plt.imshow(img.numpy())
            plt.title(label.numpy())
            plt.axis('off')
            if i == 8:
                plt.show()
                break
        break
    plt.show()

def class_report(model, criterion, dataloader, batch_size):
    """Classification report generated by the model when predicting on the
    dataset supplied by dataloader. Prints this classification report to console, returns the
    loss calculated by criterion.
    Args:
        model (nn.Module): A torch.nn.Module which takes the inputs supplied by the
            dataloader and outputs label probabilities.
        criterion (torch.nn): A loss function for the outputs and labels.
        dataloader (torch.utils.data.DataLoader): The DataLoader for
            the dataset to predict on
        batch_size (int): Size of the batches supplied by the DataLoader
    Returns:
        loss (float): Loss calculated from the criterion supplied
    """
    print("\n\t||\tVALIDATING\t||\n")
    size_dataset = len(dataloader) * batch_size
    predictions = np.zeros(size_dataset)
    true_labels = np.zeros(size_dataset)
    indx = 0

    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            # drop leftover samples
            if len(batch[0]) != batch_size:
                predictions = predictions[:(indx-1)*batch_size]
                true_labels = true_labels[:(indx-1)*batch_size]
                break

            image_batch, label_batch = batch
            outputs = model(image_batch)

            loss = criterion(outputs, label_batch)
            loss = loss.item()
            # get the index of the max value
            _, predicted_batch = torch.max(outputs, 1)

            predictions[indx*batch_size:(indx+1)*batch_size] = predicted_batch.numpy()
            true_labels[indx*batch_size:(indx+1)*batch_size] = label_batch.numpy()
            indx += 1

        true_labels = true_labels.astype(int)
        predictions = predictions.astype(int)

        print(classification_report(true_labels, predictions))

    print("\nloss:\t\t%.3f" % (loss))
    return loss

################################################################################
################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Program to train a CNN for \
        classification of infra-red images of breaking waves.")

    parser.add_argument("-f", metavar="PREFIX", type=str, nargs=1,
                        help="A prefix to uniquely identify user settings")

    args = parser.parse_args()

    if args.f is not None:
        NAME_PREFIX = args.f[0]
    else:
        NAME_PREFIX = "base_"
    print("model_"+NAME_PREFIX+"epoch_loss_acc")


    # Data and parameter settings
    BATCH_SIZE = 300
    VAL_BATCH_SIZE = 150
    IMAGE_SIZE = 96
    IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
    BASE_MODEL = "mobilenet"


    # Once adding other models, move to function and return the pretrain_transform
    if BASE_MODEL == "mobilenet":
        pretrain_transform = transforms.Compose([
            # mobilenet_v2 normalize
            transforms.Normalize((0.485, 0.456, 0.406), ((0.229, 0.224, 0.225)))
            ])

    augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=15, fill=(0,)),
        transforms.RandomResizedCrop(size=IMAGE_SHAPE, scale=(0.8, 1.0)),
        transforms.ToTensor()
        ])

    noaugment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=IMAGE_SHAPE),
        transforms.ToTensor()
    ])


    # Change to function:
    train = h5_dataloader("H5_files/train.h5", transform=augment,
                        batch_size=BATCH_SIZE, shuffle=True)

    valid = h5_dataloader("H5_files/valid.h5", transform=noaugment,
                          batch_size=VAL_BATCH_SIZE, shuffle=False)

    test = h5_dataloader("H5_files/test.h5", transform=noaugment,
                        batch_size=VAL_BATCH_SIZE, shuffle=False)


    """ #Load from JPEGs
    train = image_data("IMGS/IR/train", batch_size=300, image_shape=image_shape,
                        shuffle=True, augmentations=augment)
    valid = image_data("IMGS/IR/valid", batch_size=val_batch_size, image_shape=image_shape)
    test  = image_data("IMGS/IR/test" , batch_size=val_batch_size, image_shape=image_shape)
    """

    # augment / noaugment
    # increase channels with : x = x.numpy()[0], y = np.array([x, x, x])
    # pretrained_transform

    model = Net()

    # 1/(number of samples for each class)
    # Change to function: calc_class_weights()
    class_weights = np.array([1./5172, 1./166, 1./1652])
    class_weights = torch.FloatTensor(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train))
    ################################################################################

    # Change to function: train_model(model, train_dataloader, criterion, scheduler,
    # epochs, class_weights, verbose, model_save_prefixf)
    print("Starting training...")
    best_loss = np.inf
    for epoch in range(10):
        print("\n*****************\n\tTRAINING...")
        model.train()
        train_loss=0.0
        class_correct = np.zeros(3)
        class_total = np.zeros(3)

        print("\nLearning rate :\t %0.9f" % scheduler.get_lr()[0])
        for i, data in enumerate(train, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward, backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #print statistics
            train_loss += loss.item()

            # print every 5 mini-batch steps
            if (i+1)%5 == 0:
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels)

                for j, correct in enumerate(c):
                    label = labels[j]
                    class_correct[int(label)] += correct
                    class_total[int(label)] += 1

                train_acc = 100 * np.sum(class_correct) / np.sum(class_total)
                train_acc_w = 100./3 * np.sum((class_correct/class_total))

                print('Epoch %d\t| Step %d\t| Training loss : %.3f\t| Training acc : %.3f || %.3f' %
                      (epoch + 1, (i+1), train_loss / 5, train_acc, train_acc_w))
                train_loss = 0.0

        scheduler.step()
        print("\n\t*** TRAINING REPORT ***")
        _ = class_report(model, criterion, train, BATCH_SIZE)

        print("\n\t*** VALIDATION REPORT ***")
        validation_loss = class_report(model, criterion, valid, VAL_BATCH_SIZE)

        # save on the end of epoch if valid_loss improves
        if validation_loss < best_loss:
            print("Validation loss decreased :\t %.3f to %.3f" % (best_loss, validation_loss))
            torch.save(model.state_dict(), "SAVED_MODELS/model_%s_%d_%.2f.pth" % (
                NAME_PREFIX, epoch, validation_loss))
            print("Saved model \tmodel_%s_%d_%.2f.pth" % (
                NAME_PREFIX, epoch, validation_loss))
            best_loss = validation_loss
    print("Finished Training")

    ################################################################################

    # Loading model
    #model = TheModelClass(*args, **kwargs)
    #model.load_state_dict(torch.load(PATH))

    # Fine tuning
    #if weighted_acc > 70:
