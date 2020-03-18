import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import PIL
from sklearn.metrics import classification_report
################################################################################

class Net(nn.Module):
    # implement for other pretrained models
    # play with the arch. and hyper.
    def __init__(self):
        super(Net, self).__init__()
        # load the pretrained model
        pretrain = models.mobilenet_v2(pretrained=True)
        for param in pretrain.features.parameters():
            param.requires_grad = False
        self.model = pretrain
        self.drop = nn.Dropout(0.5)
        # remove the top from pretrained and replace with custom fully connected
        self.model.fc = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.model(x)
        x = self.drop(x)
        x = F.softmax(x)
        return x

def image_data(root_dir, image_size=(96, 96), augmentations=None, batch_size=1, shuffle=False):
    """
    Args:
        root_dir (string): path to the class folders
        image_size (int, int): Resize all the images to this.
        augmentations (callable, optional): transformations to randomly apply to images, default is None
        batch_size (int): number of images in each batch, default is 1
        shuffle (bool): to shuffle the dataset or not, default is false
    Returns:
        A batched labeled image dataloader from the root_dir with applied augmentations.
    """
    if augmentations is None:
        data_transform = transforms.Compose([
            transforms.Resize(size=image_size),
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
                            num_workers=6)

    # implement dataset._find_classes() and return
    # get number of files in each class and return
    return data_loader

def view_img_batch(data_loader):
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
    print("\ngenerating validation report...")
    classes = ('nonbreaking', 'plunge', 'spill')
    size_valid = len(valid) * val_batch_size
    preds = np.zeros(size_valid)
    labels = np.zeros(size_valid)
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    weighted_acc = 0.0
    indx = 0
    
    model.eval()
    with torch.no_grad():
        for batch in valid:
            img_batch, label_batch = batch
            outputs = model(img_batch)
            # validation loss
            valid_loss = criterion(outputs, label_batch)
            valid_loss = valid_loss.item()
            # get the index of the max value
            _, predicted = torch.max(outputs, 1)
            preds[indx*val_batch_size:(indx+1)*val_batch_size] = predicted.numpy()
            labels[indx*val_batch_size:(indx+1)*val_batch_size] = label_batch.numpy()
            c = (predicted == labels).squeeze()
            # Accuracy on each class
            for i in range(val_batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            indx+=1

        print(classification_report(labels, preds, labels=classes))

    for i in range(len(classes)):
        print("Accuracy of %5s : %2d %%" % (
            classes[i], 100*class_correct[i]/class_total[i]))
        weighted_acc += 100./3 * (class_correct[i]/class_total[i])

    print(f"Validation loss:\t{valid_loss}")
    print(f"Validation Weighted Accuracy:\t{weighted_acc}")
    return valid_loss, weighted_acc

################################################################################

# Data and parameter settings
batch_size = 300
val_batch_size = 15
image_size = (96, 96)

augment = transforms.Compose([
    transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0), ratio=(1, 1)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    # rescale to be in (0, 1)
    transforms.Normalize((0, 0, 0), ((255, 255, 255)))
    # mobilenet_v2 normalize
])

train = image_data("IMGS/IR/train", batch_size=300, shuffle=True)
valid = image_data("IMGS/IR/valid", batch_size=15)
test  = image_data("IMGS/IR/test" , batch_size=15)

model = Net()
print(model)


#class_weights = torch.from_numpy(np.array([1./5172, 1./166, 1/1652]))
criterion = nn.CrossEntropyLoss()#weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

################################################################################

# Training 
print("Starting training...")
valid_loss = np.inf
for epoch in range(10):
    model.train()
    train_loss=0.0
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
            print('Epoch %d\t| Step %d\t| Training loss : %.3f' % 
                    (epoch + 1, (i+1), train_loss / 5))
            train_loss=0.0

    print("End of Epoch...\t Running validation")
    valid_loss, weighted_acc = validation_report(model, criterion, valid, val_batch_size)
    # save on the end of epoch if valid_loss improves
    if valid_loss < best_loss:
        print(f"Validation loss decreased : {best_loss} to {valid_loss}")
        torch.save(model.state_dict(), f'SAVED_MODELS/model_{epoch}_{valid_loss}_{weighted_acc}.pth')
        print(f"Saved model \tmodel{epoch}_{valid_loss}_{weighted_acc}")
        best_loss = valid_loss
print("Finished Training")

################################################################################