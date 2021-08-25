from torch.utils.data import Dataset
import os
from transformers import PointSampler
import open3d as o3d
import numpy as np


class LettucePointCloudDataset(Dataset):
    def __init__(self, root_dir, is_train=False, transform=None):
        self.is_train = is_train
        self.transform = transform
        self.files = []

        dataset_dir = root_dir + 'PhytoOracle_Dataset/'
        for batch_dir_name in os.listdir(dataset_dir):
            pcd_dir = f'{dataset_dir}{batch_dir_name}/{batch_dir_name}_norm/'
            seg_dir = f'{dataset_dir}{batch_dir_name}/{batch_dir_name}_norm_annot_seg/'
            for f in os.listdir(seg_dir):
                self.files.append({
                    'pcd_path': pcd_dir + f.replace('.npy', '.pcd'),
                    'seg_path': seg_dir + f, 
                })

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.files[idx]['pcd_path'])
        points = np.array(pcd.points)
        labels = np.load(self.files[idx]['seg_path']).astype(np.long)

        points, labels = PointSampler(1500)((points, labels))
        if self.is_train:
            points, labels = self.transform((points, labels))

        return points, labels
