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

from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import my_utils
from callbacks import callbacks
from data import transformations
from models import my_CNNs


if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

print(f"Using :\t{DEVICE}\n")
################################################################################


def train_model(model,
                train_dataloader,
                validation_dataloader,
                criterion,
                epochs=10,
                comet_expt=None,
                scheduler=None):
    """Train the torch.nn.Module for the specified number of epochs, perfoms optimization
    with respect to the supplied criterion. Saves the model each epoch if the validation
    loss improves.
    """

    print(f"Training model for {epochs} epochs")
    print(f"Class weights :\t{criterion.weight}")
    print(f"Base model :\t{model.base}")
    print("\nStarting training...")
    start = time.perf_counter()
    best_loss = np.inf

    for epoch in range(epochs):
        print(f"Learning rate :\t {scheduler.get_lr()[0]}")

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
            if (i+1) % 5 == 0:
                train_acc, train_acc_w = callbacks.print_statistics(outputs, labels)

                # LOGGING
                callbacks.comet_logging(comet_expt,
                                        ('train_loss', train_loss),
                                        ('train_acc', train_acc),
                                        ('train_acc_w', train_acc_w),
                                        step=i+1,
                                        epoch=epoch+1)

                print(f"Epoch {epoch+1}\t| Step {i+1}\t| " + \
                      f"Training loss : {train_loss / 5 :.4f}\t| " + \
                      f"Training acc : {train_acc:.3f} || {train_acc_w:.3f}")
                train_loss = 0.0

        scheduler.step()

        # check for overfitting train
        print("\n\t*** TRAINING REPORT ***")
        _ = callbacks.class_report(model, criterion, train_dataloader)
        print("\n\t*** VALIDATION REPORT ***")
        val_loss, val_auc = callbacks.class_report(model, criterion, validation_dataloader)

        # LOGGING
        callbacks.comet_logging(comet_expt,
                                ('val_loss', val_loss),
                                ('val_AUC', val_auc),
                                ('learning_rate', scheduler.get_last_lr()[0]),
                                epoch=epoch+1)

        # save on the end of epoch if valid_loss improves
        if val_loss < best_loss:
            torch.save(model.state_dict(),
                       f"SAVED_MODELS/model_{model.base}_{epoch}_{val_loss:.4f}_{val_auc:.3f}.pth")

            print(f"Validation loss decreased :\t {best_loss:.4f} to {val_loss:.4f}")
            print(f"Saved model \tmodel_{model.base}_{epoch}_{val_loss:.4f}_{val_auc:.3f}.pth")
            # save model as model_{model.base}_best.pth then reload immediately
            best_loss = val_loss

    print("Finished Training")
    end = time.perf_counter()
    print(f"Training time : \t {end - start} seconds")



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


def main(config, comet_expt=None):
    """Fetch program settings from supplied config dictionary, get transforms for
    preprocessing images and load the datasets.  Then loads the CNN model specified
    in the config and begins training."""

    # import the image transformations (augmentations and preprocessing pretrained)
    augment = transformations.get_transform(augment=True, image_shape=config['image_shape'])
    no_augs = transformations.get_transform(image_shape=config['image_shape'])

    # Change to function:
    train = my_utils.h5_dataloader("data/H5_files/train.h5", transform=augment,
                                   batch_size=config['batch_size'], shuffle=True)

    valid = my_utils.h5_dataloader("data/H5_files/valid.h5", transform=no_augs,
                                   batch_size=config['val_batch_size'], shuffle=False)

    #test = my_utils.h5_dataloader("H5_files/test.h5", transform=no_augs,
    #                              batch_size=config['val_batch_size'], shuffle=False)

    model = my_CNNs.Net(config['base_model'])
    model = model.to(DEVICE)

    class_weights = my_utils.class_weight(train.dataset).to(DEVICE)
    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                               T_0=5,
                                                               T_mult=2)


    train_model(model, train, valid, criterion, epochs=config['initial_epochs'],
                comet_expt=comet_expt, scheduler=scheduler)



################################################################################
################################################################################

if __name__ == "__main__":

    CONFIG_FILENAME = parse_args()
    S = os.sep
    with open(f"{os.getcwd()+S}models{S}configs{S+CONFIG_FILENAME}.json") as f:
        CONFIG = json.load(f)

    for value in CONFIG:
        print(f"{value} \t: {CONFIG[value]}")

    COMET_EXPT = Experiment(project_name="waveclass", workspace="ryan597")
    COMET_EXPT.log_parameters(CONFIG)

    main(CONFIG, COMET_EXPT)
