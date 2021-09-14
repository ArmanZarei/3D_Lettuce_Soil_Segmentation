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
from utils.utils import training_process_plot_save, test_accuracy, get_model_output_and_loss
from utils.visualizer import PointCloudVisualizer, labels_to_soil_and_lettuce_colors
import numpy as np


train_dataset = LettucePointCloudDataset(
    root_dir='', 
    is_train=True,
    transform=transforms.Compose([
        RandomRotation()
    ])
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# model = PointNet().to(device)
# model = RandLANet(d_in=3, num_classes=2, num_neighbors=16, decimation=4, device=device).to(device)
model = PointNet2(2).to(device)

model.train()
model_name = type(model).__name__
print(f'Model: {model_name}\n{"-"*30}')

num_epoches = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epoches):
    train_loss, train_acc = .0, .0
    for input, labels in train_dataloader:
        input, labels = input.to(device).squeeze().float(), labels.to(device)
    
        optimizer.zero_grad()
        outputs, loss = get_model_output_and_loss(model, input, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (labels == outputs.argmax(1)).sum().item() / np.prod(labels.shape)
    
    train_loss, train_acc = train_loss/len(train_dataloader), train_acc/len(train_dataloader)
    print(f'Epoch: {"{:2d}".format(epoch)} -> \t Train Loss: {"%.10f"%train_loss} \t Train Accuracy: {"%.4f"%train_acc}')

torch.save(model.state_dict(), f'pretrained_models/{model_name}.pth')
