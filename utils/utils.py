import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation


def random_point_sampler(points, labels, size=5000):
    """
    Random Point Sampler

    Parameters:
        points (list): List of pointclouds
        labels (list): List containing the labels of the pointclouds
        size (int): Numuber of points to sample
    """

    assert points.shape[0] == labels.size
    assert size < labels.size

    indices = np.random.choice(labels.size, size)
    return points[indices], labels[indices]

def training_process_plot_save(train_loss_arr, val_loss_arr, train_accuracy_arr, val_accuracy_arr, save_dir='images/training.png'):
    """
    Loss & Accuracy of training set and validation set plot during the training

    Parameters:
        train_loss_arr (list): Training loss list
        val_loss_arr (list): Validation loss list
        train_accuracy_arr (list): Training set accuracy list
        val_accuracy_arr (list): Validation set accuracy list
        save_dir (str): Directory to save result
    """

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1).set_title("Loss / Epoch")
    plt.plot(train_loss_arr, label='Train')
    plt.plot(val_loss_arr, label='Validation')
    plt.legend()
    plt.subplot(1, 2, 2).set_title("Accuracy / Epoch")
    plt.plot(train_accuracy_arr, label='Train')
    plt.plot(val_accuracy_arr, label='Validation')
    plt.legend()
    plt.savefig(save_dir)

def test_accuracy(model, test_dataloader, device):
    """
    Calculates the accuracy of the model on test set

    Parameters:
        model (Type[nn.Module]): Model
        test_dataloader (DataLoader): Test dataloader
        device (str): cpu or cuda
    """

    model.eval()
    test_acc = .0
    with torch.no_grad():
        for input, labels in test_dataloader:
            input, labels = input.to(device).squeeze().float(), labels.to(device)
            outputs, _, _ = model(input)
            test_acc += (labels == outputs.argmax(1)).sum().item() / np.prod(labels.shape)
        test_acc /= len(test_dataloader)
    
    return test_acc