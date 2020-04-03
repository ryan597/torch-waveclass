################################################################################
# Written by Ryan Smith    |    ryan.smith@ucdconnect.ie    |   March 2020
# University College Dublin|    github.com:ryan597/waveclass.git

################################################################################
################################################################################
#                   TO-DO:
# Generalise to both IR and Flow
# Test out RAW + RAW, RAW + FLOW
# Testing of different arch. & hyper. & augmentations
################################################################################
import argparse
import time
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import my_utils
from callbacks import callbacks
from data import transformations
from models import ClassifyNet

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
                config,
                scheduler=None):

    print("\nStarting training...")
    start = time.perf_counter()
    best_loss = np.inf
    # Arrays to record stats
    train_loss = []
    train_auc = []
    valid_loss = []
    valid_auc = []

    for epoch in range(config['epochs']):
        print(f"Learning rate :\t {scheduler.get_lr()[0]}")

        model.train()
        tmp_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader, 0):

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            scheduler.optimizer.zero_grad()

            # forward, backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # accumulate loss for epoch
            tmp_loss += loss.item()

            # print every 5 mini-batch steps
            if (i+1) % 5 == 0:
                scheduler.optimizer.step()
                
                train_auc.append(callbacks.print_class_stat(outputs.detach(), labels,
                                                    epoch+1, i+1, tmp_loss/5))
                train_loss.append(tmp_loss/5)
                tmp_loss = 0.0

        scheduler.step()

        print("\n\t*** VALIDATION REPORT ***")
        val_loss, val_auc = callbacks.class_report(model, criterion, validation_dataloader)
        valid_loss.append(val_loss)
        valid_auc.append(val_auc)

        # save on the end of epoch if valid_loss improves
        if val_loss < best_loss:
            torch.save(model.state_dict(),
                       f"SAVED_MODELS/{model.base}_{config['data']}.pth")

            print(f"Validation loss decreased :\t {best_loss:.4f} to {val_loss:.4f}")
            best_loss = val_loss

    print("Finished Training")
    end = time.perf_counter()
    print(f"Training time : \t {end - start} seconds")
    history = {
        "train_loss" : train_loss,
        "train_auc" : train_auc,
        "val_loss" : valid_loss,
        "val_auc" : valid_auc,
        "epochs" : epoch,
        "batches" : len(train_dataloader)
    }
    return history

def plot_history(history):
    x1 = range(history['epochs'] * history['batches']) / history['batches']
    x2 = range(history['epochs'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.plot(x1, history['train_loss'], 'b-')
    ax1.plot(x2, history['valid_loss'], 'r-')
    ax1.title('Training and Validation loss')
    ax1.xlabel('Epoch')
    ax1.ylabel('Loss')
    ax1.grid(True)

    ax2.plot(x1, history['train_auc'], 'b-')
    ax2.plot(x2, history['valid_auc'], 'r-')
    ax2.title('Train and Validation AUC')
    ax2.xlabel('Epoch')
    ax2.ylabel('AUC')
    ax2.grid(True)

    fig.show()

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
        config_filename = args.c[0]

    return config_filename


def main(config):

    # import the image transformations (augmentations and preprocessing pretrained)
    augment = transformations.get_transform_HDF5(augment=True, image_shape=config['image_shape'])
    no_augs = transformations.get_transform_HDF5(image_shape=config['image_shape'])

    train = my_utils.h5_dataloader(config['train'], transform=augment,
                                      batch_size=config['batch_size'], shuffle=True)

    valid = my_utils.h5_dataloader(config['valid'], transform=no_augs,
                                      batch_size=config['val_batch_size'], shuffle=False)

    #test = my_utils.h5_dataloader(config['test'], transform=no_augs,
    #                              batch_size=config['val_batch_size'], shuffle=False)

    model = ClassifyNet.ClassifyNet(config)
    model = model.to(DEVICE)

    class_weights = my_utils.class_weight(train.dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=config['epochs'])

    history = train_model(model, train, valid, criterion,
                config, scheduler=scheduler)

    plot_history(history)


################################################################################
################################################################################

if __name__ == "__main__":

    CONFIG_FILENAME = parse_args()
    S = os.sep
    with open(f"{os.getcwd()+S}models{S}configs{S+CONFIG_FILENAME}.json") as f:
        CONFIG = json.load(f)

    for value in CONFIG:
        print(f"{value} \t: {CONFIG[value]}")

    main(CONFIG)
