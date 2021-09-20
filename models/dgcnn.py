import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
	inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
	xx = torch.sum(x ** 2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

	idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
	return idx


def get_graph_feature(x, k=20):
	# x = x.squeeze()
	idx = knn(x, k=k)  # (batch_size, num_points, k)
	batch_size, num_points, _ = idx.size()
	
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

	idx = idx + idx_base

	idx = idx.view(-1)

	_, num_dims, _ = x.size()

	# (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
	x = x.transpose(2, 1).contiguous()  

	feature = x.view(batch_size * num_points, -1)[idx, :]
	feature = feature.view(batch_size, num_points, k, num_dims)
	x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

	feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

	return feature


class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()

        self.conv1 = nn.Sequential(
			nn.Conv2d(6, 64, kernel_size=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2)
		)
        self.conv2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(negative_slope=0.2)
		)
        self.conv3 = nn.Sequential(
			nn.Conv1d(128, 1024, kernel_size=1, bias=False),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(negative_slope=0.2)
		)

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN(torch.nn.Module):
	def __init__(self, num_classes=2):
		super(DGCNN, self).__init__()

		self.transform_net = Transform_Net()
		self.conv0 = nn.Sequential(
			nn.Conv2d(6, 64, kernel_size=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv1 = nn.Sequential(
			nn.Conv2d(6, 64, kernel_size=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv6 = nn.Sequential(
			nn.Conv1d(64*3, 1024, kernel_size=1, bias=False),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv7 = nn.Sequential(
			nn.Conv1d(1024 + 64*3, 512, kernel_size=1, bias=False),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv8 = nn.Sequential(
			nn.Conv1d(512, 256, kernel_size=1, bias=False),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv9 = nn.Sequential(
			nn.Conv1d(256, 128, kernel_size=1, bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.dp1 = nn.Dropout(0.5)
		self.dp2 = nn.Dropout(0.5)
		self.conv10 = nn.Conv1d(128, num_classes, kernel_size=1, bias=False)


	def forward(self, x):
		x = x.permute(0, 2, 1)
		if x.shape[1] != 3:
			raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")
		
		num_points = x.size(2)

		x0 = get_graph_feature(x)
		t = self.transform_net(x0)
		x = x.transpose(2, 1)
		x = torch.bmm(x, t)
		x = x.transpose(2, 1)

		x = get_graph_feature(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x1 = x.max(dim=-1, keepdim=False)[0]

		x = get_graph_feature(x1)
		x = self.conv3(x)
		x = self.conv4(x)
		x2 = x.max(dim=-1, keepdim=False)[0]

		x = get_graph_feature(x2)
		x = self.conv5(x)
		x3 = x.max(dim=-1, keepdim=False)[0]

		x = torch.cat((x1, x2, x3), dim=1)
		
		x = self.conv6(x)
		x = x.max(dim=-1, keepdim=True)[0]

		x = x.repeat(1, 1, num_points)
		x = torch.cat((x, x1, x2, x3), dim=1)

		x = self.conv7(x)
		x = self.conv8(x)
		x = self.dp1(x)
		x = self.conv9(x)
		x = self.dp2(x)
		x = self.conv10(x)

		return x