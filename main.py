################################################################################
# Written by Ryan Smith    |    ryan.smith@ucdconnect.ie    |   March 2020
# University College Dublin|    github.com:ryan597/waveclass.git

################################################################################
################################################################################
#                   TO-DO:
# Generalise to both IR and Flow
# Automatic calculation of class weights
# Testing of different arch. & hyper. & augmentations
# Adding plotting of training history (real time updates)
# Implememnt early stopping
# Have some coffee
# Refactor again...
################################################################################
import argparse
import time
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from comet_ml import Experiment

import my_utils
from callbacks import callbacks
from data import transformations
from models import my_CNNs


if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

print(f"Using {DEVICE}")
################################################################################


def train_model(model,
                train_dataloader,
                validation_dataloader,
                criterion,
                scheduler=None,
                epochs=10,
                verbose=1):
    """Train the torch.nn.Module for the specified number of epochs, perfoms optimization
    with respect to the supplied criterion. Saves the model each epoch if the validation
    loss improves.
    """

    if verbose:
        print(f"Training model for {epochs} epochs")
        print(f"Class weights :\t{criterion.weight}")
        print(f"Base model :\t{model.base}")
        print("\nStarting training...")
        start = time.perf_counter()
    best_loss = np.inf

    for epoch in range(epochs):
        if verbose:
            print(f"Learning rate :\t {scheduler.get_lr()}")

        model.train()
        train_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            #inputs, labels = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            scheduler.optimizer.zero_grad()

            # forward, backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            scheduler.optimizer.step()
            # accumulate loss for epoch
            train_loss += loss.item()

            # print every 5 mini-batch steps
            if verbose & (i+1) % 5 == 0:
                train_acc, train_acc_w = callbacks.print_statistics(outputs, labels)

                print(f"Epoch {epoch+1}\t| Step {i+1}\t| " + \
                      f"Training loss : {train_loss / 5 :.4f}\t| " + \
                      f"Training acc : {train_acc:.3f} || {train_acc_w:.3f}")
                train_loss = 0.0

        scheduler.step()

        if verbose == 2:
            print("\n\t*** TRAINING REPORT ***")
            _ = callbacks.class_report(model, criterion, train_dataloader)
        if verbose:
            print("\n\t*** VALIDATION REPORT ***")
            validation_loss = callbacks.class_report(model, criterion, validation_dataloader)

        # save on the end of epoch if valid_loss improves
        if validation_loss < best_loss:
            print("Validation loss decreased :\t %.3f to %.3f" % (best_loss, validation_loss))
            torch.save(model.state_dict(), "SAVED_MODELS/model_%s_%d_%.2f.pth" % (
                       model.base, epoch, validation_loss))
            print("Saved model \tmodel_%s_%d_%.2f.pth" % (
                  model.base, epoch, validation_loss))
            best_loss = validation_loss

    print("Finished Training")
    if verbose:
        end = time.perf_counter()
        print(f"Training time : \t {end - start} seconds")
    #return history


def parse_args():
    """Fetch config_file to use from command line input, defaults to resnet18 config"""

    parser = argparse.ArgumentParser(description="Program to train a CNN for \
    classification of infra-red images of breaking waves.")

    parser.add_argument("-c", metavar="CONFIG", type=str, nargs=1,
                        help="specify the config_file to use")

    args = parser.parse_args()

    if args.c is None:
        config_filename = "resnet18"
    else:
        config_filename = args.m[0]

    return config_filename


def main(config):
    """Fetch program settings from supplied config dictionary, get transforms for
    preprocessing images and load the datasets.  Then loads the CNN model specified
    in the config and begins training."""
    try:
        batch_size = config['batch_size']
        val_batch_size = config['val_batch_size']
        image_shape = config['image_shape']
        initial_epochs = config['initial_epochs']
        base_model = config['base_model']
        verbose = config['verbose']
    except:
        raise Exception("CONFIG PARAMETERS NOT FOUND...")

    # import the image transformations (augmentations and preprocessing pretrained)
    augment = transformations.get_transform(augment=True, image_shape=image_shape)
    no_augs = transformations.get_transform(image_shape=image_shape)

    # Change to function:
    train = my_utils.h5_dataloader("H5_files/train.h5", transform=augment,
                                   batch_size=batch_size, shuffle=True)

    valid = my_utils.h5_dataloader("H5_files/valid.h5", transform=no_augs,
                                   batch_size=val_batch_size, shuffle=False)

    #test = my_utils.h5_dataloader("H5_files/test.h5", transform=no_augs,
    #                              batch_size=val_batch_size shuffle=False)

    model = my_CNNs.Net(base_model)

    # Change to function: calc_class_weights() in my_utils
    class_weights = np.array([1./166, 1./1652, 1./5172])
    class_weights = torch.FloatTensor(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                               T_0=5,
                                                               T_mult=2)

    #### COMET
    experiment = Experiment(project_name='waveclass', workspace='ryan597')

    experiment.display()

    train_model(model, train, valid, criterion, epochs=initial_epochs, scheduler=scheduler,
                verbose=verbose)

    experiment.end()


################################################################################
################################################################################

if __name__ == "__main__":

    CONFIG_FILENAME = parse_args()
    S = os.sep
    with open(f"{os.getcwd()+S}models{S}configs{S+CONFIG_FILENAME}.json") as f:
        CONFIG = json.load(f)

    main(CONFIG)

################################################################################
################################################################################
