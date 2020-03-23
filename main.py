################################################################################
# Written by Ryan Smith    |    ryan.smith@ucdconnect.ie    |   March 2020
# University College Dublin|    github.com:ryan597/waveclass.git

import argparse
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
import time

import numpy as np
import torch
import torch.optim as optim
from comet_ml import Experiment
from torchvision import transforms as T

import my_utils
from callbacks import callbacks
from data import transformations
from models import my_CNNs


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(f"Using {device}")
################################################################################


def train_model(model,
                train_dataloader,
                validation_dataloader,
                criterion,
                learning_rate=0.001,
                epochs=10,
                scheduler=None,
                class_weights=None,
                model_save_prefix=None,
                verbose=1):
    """Train the torch.nn.Module for the specified number of epochs, perfoms optimization
    with respect to the supplied criterion. Saves the model each epoch if the validation
    loss improves.
    """
    if verbose:
        print(f"Training model for {epochs} epochs")
        print(f"Class weights :\t{class_weights.numpy()}")
        print(f"Model save file prefix :\t{model_save_prefix}")
        print("\nStarting training...")
        start = time.perf_counter()
    best_loss = np.inf

    for epoch in range(epochs):
        if verbose:
            if scheduler is not None:
                lr = scheduler.get_lr()[0]
            else:
                lr = learning_rate
            print(f"Learning rate :\t {lr}")

        model.train()
        train_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            #inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward, backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # accumulate loss for epoch
            train_loss += loss.item()

            #print_statistics(outputs, labels)
            # print every 5 mini-batch steps
            if verbose & (i+1)%5==0 :
                if (i+1)%5 == 0:
                    train_acc, train_acc_w = print_statistics(outputs, labels)

                    print('Epoch %d\t| Step %d\t| Training loss : %.3f\t| Training acc : %.3f || %.3f' %
                        (epoch + 1, (i+1), train_loss / 5, train_acc, train_acc_w))
                    train_loss = 0.0

        if scheduler is not None:
            scheduler.step()

        if verbose == 2:
            print("\n\t*** TRAINING REPORT ***")
            _ = class_report(model, criterion, train, BATCH_SIZE)
        if verbose:
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
    if verbose:
        end = time.perf_counter()
        print(f"Training time : \t {end - start} seconds")
    #return history

################################################################################
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Program to train a CNN for \
    classification of infra-red images of breaking waves.")

    parser.add_argument("-m", metavar="PREFIX", type=str, nargs=1,
                        help="base CNN model")

    args = parser.parse_args()

    if args.m is not None:
        base_model = args.m[0]
    else:
        base_model = "resnet18"
    print(f"saving as :\tmodel_{base_model}_epoch_loss_acc")

    return base_model


if __name__ == "__main__":


    # Data and parameter settings
    BATCH_SIZE = 300
    VAL_BATCH_SIZE = 150
    IMAGE_SIZE = 96
    IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
    INITIAL_EPOCHS = 10
    BASE_MODEL = parse_args()
    VERBOSE = 1

    augment = get_transform(BASE_MODEL, augemnt=True, image_shape=IMAGE_SHAPE)
    no_augs = get_transform(BASE_MODEL, image_shape=IMAGE_SHAPE)

    # import the image transformations (augmentations and preprocessing pretrained)
    model = my_CNNs.Net(BASE_MODEL)

    # Change to function:
    train = h5_dataloader("H5_files/train.h5", transform=augment,
                        batch_size=BATCH_SIZE, shuffle=True)

    valid = h5_dataloader("H5_files/valid.h5", transform=no_augs,
                          batch_size=VAL_BATCH_SIZE, shuffle=False)

    test = h5_dataloader("H5_files/test.h5", transform=no_augs,
                        batch_size=VAL_BATCH_SIZE, shuffle=False)

    model = Net(BASE_MODEL)

    # Change to function: calc_class_weights()
    class_weights = np.array([1./166, 1./1652, 1./5172])
    class_weights = torch.FloatTensor(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=INITIAL_EPOCHS)
    ################################################################################

    #### COMET
    experiment = Experiment(project_name='waveclass', workspace='ryan597')

    experiment.display()

    train_model(model, train, valid, criterion, epochs=10, scheduler=scheduler, class_weights=class_weights,
            model_save_prefix=BASE_MODEL, verbose=VERBOSE)
    
    experiment.end()
    ################################################################################
