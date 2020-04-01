"""Functions to call during the training loop."""

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

def class_report(model, criterion, dataloader):
    """Classification report generated by the model when predicting on the
    dataset supplied by dataloader. Prints this classification report to console, returns the
    loss calculated by criterion.
    """
    batch_size = dataloader.batch_size
    size_dataset = len(dataloader.dataset)
    num_classes = len(dataloader.dataset.classes)
    true_labels = torch.zeros(size_dataset).to(DEVICE)
    output_full = torch.zeros((size_dataset, num_classes)).to(DEVICE)

    model.eval()
    with torch.no_grad():
        for i, (image_batch, label_batch) in enumerate(dataloader):
            # check size of batch in case of excess samples
            len_batch = len(label_batch)

            image_batch = image_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            output_full[i*batch_size:i*batch_size+len_batch] = model(image_batch)
            true_labels[i*batch_size:i*batch_size+len_batch] = label_batch

        loss = criterion(output_full, true_labels.to(torch.long)).item()

    true_labels = true_labels.to('cpu').numpy().astype(int)
    output_full = output_full.to('cpu').numpy().astype(int)

    # one hot encode for roc_auc_score
    one_hot_labels = np.zeros((size_dataset, num_classes))
    for i, value in enumerate(true_labels):
        one_hot_labels[i, value] = 1
    # tensors to numpy arrays

    auc = roc_auc_score(one_hot_labels, output_full, multi_class='ovo')

    print(classification_report(true_labels, np.argmax(output_full, -1)))
    print(f"Area under curve :\t{auc}")
    print(f"\nloss:\t\t{loss:.4f}")

    return loss, auc

def print_statistics(outputs, labels):
    """Calculates the accuracy of output predictions to the given labels"""
    class_correct = np.zeros(3)
    class_total = np.zeros(3)

    _, predicted = torch.max(outputs, 1)
    binary_correct = (predicted == labels)

    for j, correct in enumerate(binary_correct):
        label = labels[j]
        class_correct[int(label)] += correct
        class_total[int(label)] += 1

    overall_accuracy = 100 * np.sum(class_correct) / np.sum(class_total)
    average_accuracy = 100./3 * np.sum((class_correct/class_total))

    return overall_accuracy, average_accuracy

#def save_model():

#def verbose_print():


def comet_logging(comet_expt, *args, **kwargs):
    """Function for logging metrics to comet_ml experiments.
    Pass the metric with associated value as a tuple along with the step and epoch
    as follows:
    comet_logging(EXPT, (m1, v1), (m2, v2), step=5, epoch=2)."""
    if comet_expt is not None:
        for metric, value in args:
            comet_expt.log_metric(metric, value, **kwargs)
