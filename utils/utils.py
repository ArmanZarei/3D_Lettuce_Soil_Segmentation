from models.dgcnn import DGCNN
from models.pointnet2 import PointNet2
from models.pointnet import PointNet, pointnet_loss
from models.randlanet import RandLANet
from models.dgcnn import DGCNN
from models.simplified_dgcnn import SimplifiedDGCNN
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
import torch.nn as nn


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
            outputs, _ = get_model_output_and_loss(model, input, labels, calculate_loss=False)
            test_acc += (labels == outputs.argmax(1)).sum().item() / np.prod(labels.shape)
        test_acc /= len(test_dataloader)
    
    return test_acc

def get_model_output_and_loss(model, input, labels, calculate_loss=True):
    """
    Returns the output and loss of model according to model type

    Parameters:
        model (Type[nn.Module]): Model
        input (Tensor): input
    """
    if isinstance(model, PointNet):
        outputs, mat_3x3, mat_64x64 = model(input)
        if not calculate_loss:
            return outputs, None
        return outputs, pointnet_loss(outputs, labels, mat_3x3, mat_64x64)
    elif isinstance(model, (RandLANet, PointNet2, DGCNN, SimplifiedDGCNN)):
        outputs = model(input)
        if not calculate_loss:
            return outputs, None
        return outputs, nn.CrossEntropyLoss()(outputs, labels)
    
    raise Exception("Model should be of type PointNet, RandLANet, PointNet++ (PointNet2), DGCNN or SimplifiedDGCNN")

def get_model_optimizer_and_scheduler(model, num_epochs):
    """
    Returns optimizer and scheduler according to model's type

    Parameters:
        model (Type[nn.Module]): Model
        num_epochs (Tensor): Number of epochs
    """
    if isinstance(model, (PointNet, RandLANet, PointNet2)):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
    elif isinstance(model, (DGCNN, SimplifiedDGCNN)):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-3)
    else:
        raise Exception("Model should be of type PointNet, RandLANet, PointNet++ (PointNet2), DGCNN or SimplifiedDGCNN")
    
    return optimizer, scheduler