import numpy as np
import torch

from sklearn.metrics import classification_report, roc_auc_score

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

def class_report(model, criterion, dataloader):

    batch_size = dataloader.batch_size
    size_dataset = len(dataloader.dataset)
    num_classes = 3
    true_labels = torch.zeros(size_dataset).to(DEVICE)
    output_full = torch.zeros((size_dataset, num_classes)).to(DEVICE)

    model.eval()
    with torch.no_grad():
        for i, (image_batch, label_batch) in enumerate(dataloader):
            # check size of batch
            len_batch = len(label_batch)

            image_batch = image_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            output_full[i*batch_size:i*batch_size+len_batch] = model(image_batch)
            true_labels[i*batch_size:i*batch_size+len_batch] = label_batch

        loss = criterion(output_full, true_labels.to(torch.long)).item()

    true_labels = true_labels.to('cpu').numpy().astype(int)
    output_full = output_full.to('cpu').numpy()

    # one hot encode for roc_auc_score
    one_hot_labels = np.zeros((size_dataset, num_classes))
    for i, value in enumerate(true_labels):
        one_hot_labels[i, value] = 1

    auc = roc_auc_score(one_hot_labels, output_full, multi_class='ovo')

    print(classification_report(true_labels, np.argmax(output_full, -1)))
    print(f"Area under curve :\t{auc}")
    print(f"\nloss:\t\t{loss:.4f}")

    return loss, auc

def print_class_stat(outputs, labels, epoch, step, loss):

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

    # one hot encode for roc_auc_score
    one_hot_labels = np.zeros((len(labels), 3))
    for i, value in enumerate(labels):
        one_hot_labels[i, value] = 1

    auc = roc_auc_score(one_hot_labels, outputs, multi_class='ovo')
    print(f"Epoch {epoch}\t| Step {step}\t | loss {loss}\t" + \
          f"Acc : {overall_accuracy:.3f} || {average_accuracy:.3f}\t AUC : {auc}")

    return auc
