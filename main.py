from dataset import LettucePointCloudDataset
from transformers import RandomRotation
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataset import random_split
import torch
import torch.nn as nn
from models.pointnet import PointNet
from models.randlanet import RandLANet
from models.pointnet2 import PointNet2
from models.dgcnn import DGCNN
from models.simplified_dgcnn import SimplifiedDGCNN
from utils.utils import training_process_plot_save, test_accuracy, get_model_output_and_loss, get_model_optimizer_and_scheduler
from utils.visualizer import PointCloudVisualizer, labels_to_soil_and_lettuce_colors
import numpy as np


# -------------------------------- Dataset & DataLoader -------------------------------- #
dataset = LettucePointCloudDataset(
    root_dir='', 
    is_train=False,
    transform=transforms.Compose([
        RandomRotation()
    ])
)

train_dataset, val_dataset, test_dataset = random_split(dataset, [70, 13, 10])
train_dataset.is_train = True

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)


# -------------------------------- Training -------------------------------- #
num_epochs = 150

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n{"-"*30}')


# model = PointNet().to(device)
# model = RandLANet(d_in=3, num_classes=2, num_neighbors=16, decimation=4, device=device).to(device)
# model = PointNet2(2).to(device)
model = DGCNN(num_classes=2).to(device)
# model = SimplifiedDGCNN(num_classes=2).to(device)


model_name = type(model).__name__


optimizer, scheduler = get_model_optimizer_and_scheduler(model, num_epochs)


train_loss_arr, val_loss_arr = [], []
train_accuracy_arr, val_accuracy_arr = [], []

for epoch in range(num_epochs):
    train_loss, val_loss = .0, .0
    train_acc, val_acc = .0, .0
  
    model.train()
    for input, labels in train_dataloader:
        input, labels = input.to(device).squeeze().float(), labels.to(device)
    
        optimizer.zero_grad()
        outputs, loss = get_model_output_and_loss(model, input, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (labels == outputs.argmax(1)).sum().item() / np.prod(labels.shape)
    if scheduler is not None:
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        for input, labels in val_dataloader:
            input, labels = input.to(device).squeeze().float(), labels.to(device)
            outputs, loss = get_model_output_and_loss(model, input, labels)
            val_loss += loss.item()
            val_acc += (labels == outputs.argmax(1)).sum().item() / np.prod(labels.shape)

    train_loss_arr.append(train_loss/len(train_dataloader))
    val_loss_arr.append(val_loss/len(val_dataloader))
    train_accuracy_arr.append(train_acc/len(train_dataloader))
    val_accuracy_arr.append(val_acc/len(val_dataloader))
  
    print(f'Epoch: {"{:2d}".format(epoch)} -> \t Train Loss: {"%.10f"%train_loss_arr[-1]} \t Validation Loss: {"%.10f"%val_loss_arr[-1]} | Train Accuracy: {"%.4f"%train_accuracy_arr[-1]} \t Validation Accuracy: {"%.4f"%val_accuracy_arr[-1]} \t ')


torch.save(model.state_dict(), f'pretrained_models/{model_name}.pth')
training_process_plot_save(train_loss_arr, val_loss_arr, train_accuracy_arr, val_accuracy_arr, save_dir=f'images/training_{model_name}.png')


# # -------------------------------- Testing -------------------------------- #
print("-"*30)
print(f"Test Accuracy: {test_accuracy(model, test_dataloader, device)}")


# # -------------------------------- Visualization -------------------------------- #
print("-"*30)
input, labels = next(iter(test_dataloader))
input, labels = input.squeeze().float().to(device), labels.to(device)

outputs, _ = get_model_output_and_loss(model, input, None, calculate_loss=False)
outputs = outputs.argmax(1)

visualizer = PointCloudVisualizer()
num_visualizations = 3
for i in range(num_visualizations):
    print(f"Visualization {i+1}/{num_visualizations}")
    curr_input, curr_label, curr_output = input[i].cpu(), labels[i].cpu(), outputs[i].cpu()
    visualizer.save_visualization(curr_input, labels_to_soil_and_lettuce_colors(curr_label), f'images/{model_name}_labeled_{i}.gif')
    visualizer.save_visualization(curr_input, labels_to_soil_and_lettuce_colors(curr_output), f'images/{model_name}_predicted_{i}.gif')
    colors = np.full(curr_label.shape[0], '#2ecc71', dtype=object)
    colors[curr_output != curr_label] = '#e74c3c'
    visualizer.save_visualization(curr_input, colors, f'images/{model_name}_diff_{i}.gif')