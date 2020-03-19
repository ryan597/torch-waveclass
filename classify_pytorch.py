################################################################################
# Written by Ryan Smith    |    ryan.smith@ucdconnect.ie    |   March 2020
# University College Dublin|    github.com:ryan597/waveclass.git

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

parser = argparse.ArgumentParser(description="Model save file prefix")
parser.add_argument("-f", metavar="PREFIX", type=str, nargs=1,
        help="A prefix to uniquely identify user settings")
args = parser.parse_args()
if args.f is not None:
    model_prefix = args.f[0]
else:
    model_prefix = "base_"
print("model_"+model_prefix+"epoch_loss_acc")
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

def h5_dataloader(h5path, transform=None, batch_size=32, shuffle=False, num_workers=4):
    """Function to return a torch.utils.DataLoader for the HDF5 files containing
    input images.
    Args:
        h5path (string): Path to the HDF5 file to be read in eg. H5_files/train.h5
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
    dataset = H5Dataset(h5path, transform=transform)
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

def validation_report(model, criterion, valid, val_batch_size):
    """Classification report generated by the model when predicting from the 
    validation dataset. Prints this classification report to console, returns the
    loss and an accuracy averaged over the classes.
    Args:
        model (nn.Module): A torch.nn.Module which takes the inputs supplied by the 
            validation dataloader and outputs label probabilities.
        criterion (torch.nn): A loss function valid for the outputs and labels.
        valid (DataLoader): The validation DataLoader for the testing dataset
        val_batch_size (int): Size of the batches supplied by the validation 
            DataLoader
    Returns:
        valid_loss (float): Loss calculated from the criterion supplied
        weighted_acc (float): Accuracy of model averaged over the classes on the
            validation dataset
    """
    print("\ngenerating report...")
    classes = ('nonbreaking', 'plunge', 'spill')
    size_valid = len(valid) * val_batch_size
    preds = np.zeros(size_valid)
    labels = np.zeros(size_valid)
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    weighted_acc = 0.0
    indx = 0

    with torch.no_grad():
        model.eval()
        for batch in valid:
            # drop leftover samples
            if len(batch[0]) != val_batch_size:
                break
            img_batch, label_batch = batch
            outputs = model(img_batch)
            # validation loss
            valid_loss = criterion(outputs, label_batch)
            valid_loss = valid_loss.item()
            # get the index of the max value
            _, predicted = torch.max(outputs, 1)
            c = (predicted == label_batch)

            preds[indx*val_batch_size:(indx+1)*val_batch_size] = predicted.numpy()
            labels[indx*val_batch_size:(indx+1)*val_batch_size] = label_batch.numpy()
            # Accuracy on each class
            for i in range(val_batch_size):
                label = label_batch.numpy()[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            indx += 1
        labels = labels.astype(int)
        preds = preds.astype(int)
        print(classification_report(labels, preds))

    for i, _ in enumerate(classes):
        print("Accuracy of %5s\t:\t %.2f %%" % (
            classes[i], 100*class_correct[i]/class_total[i]))
        weighted_acc += 100./3 * (class_correct[i]/class_total[i])

    print("\nValidation loss:\t%.3f" % (valid_loss))
    print("Validation Weighted Accuracy:\t%.2f" % (weighted_acc))
    return valid_loss, weighted_acc

################################################################################

# Data and parameter settings
batch_size = 300
val_batch_size = 150
image_size = 96
image_shape = (image_size, image_size)
base_model = "mobilenet"


# Once adding other models, move to function and return the pretrain_transform
if base_model == "mobilenet":
    pretrain_transform = transforms.Compose([
        # mobilenet_v2 normalize
        transforms.Normalize((0.485, 0.456, 0.406), ((0.229, 0.224, 0.225)))
        ])

augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=15, fill=(0,)),
    transforms.RandomResizedCrop(size=image_shape, scale=(0.8, 1.0)),
    transforms.ToTensor()
    ])

noaugment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=image_shape),
    transforms.ToTensor()
])


# augment / noaugment
# increase channels with : x = x.numpy()[0], y = np.array([x, x, x])
# pretrained_transform


train = h5_dataloader("H5_files/train.h5", transform=augment,
                    batch_size=batch_size, shuffle=True)
valid = h5_dataloader("H5_files/valid.h5", transform=noaugment, batch_size=batch_size,
                    shuffle=False)
test = h5_dataloader("H5_files/test.h5", transform=noaugment, batch_size=batch_size,
                   shuffle=False)


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
class_weights = np.array([1./5172, 1./166, 1./1652])
class_weights = torch.FloatTensor(class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train))
################################################################################

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
    print("\n*****************\n\tVALIDATING...")
    print("\n\t*** TRAINING REPORT ***")
    _, _ = validation_report(model, criterion, train, batch_size)
    print("\n\t*** VALIDATION REPORT ***")
    valid_loss, weighted_acc = validation_report(model, criterion, valid, val_batch_size)
    # save on the end of epoch if valid_loss improves
    if valid_loss < best_loss:
        print("Validation loss decreased :\t %.3f to %.3f" % (best_loss, valid_loss))
        torch.save(model.state_dict(), "SAVED_MODELS/model_%s_%d_%.2f_%.2f.pth" % (
            model_prefix, epoch, valid_loss, weighted_acc))
        print("Saved model \tmodel_%s_%d_%.2f_%.2f.pth" % (
            model_prefix, epoch, valid_loss, weighted_acc))
        best_loss = valid_loss
print("Finished Training")

################################################################################

# Loading model
#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))

# Fine tuning
#if weighted_acc > 70:
