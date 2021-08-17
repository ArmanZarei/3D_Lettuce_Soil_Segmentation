import argparse
import os
import open3d as o3d
import json
import numpy as np


def normalize(source_dir, dest_dir):
    """
    - Normalize pointclouds in format .ply
    - Save in .pcd format (to use in supervisely)
    - save needed information for transforming the pointcloud to the original one

    Parameters:
        source_dir (str): The directory of origin pointclouds to normalize
        dest_dir (str): The directory to save results in
    """

    norm_dict = {}

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file_name in os.listdir(source_dir):
        if not file_name.endswith('.ply'):
            continue

        file_path = os.path.join(source_dir, file_name)

        pcd = o3d.io.read_point_cloud(file_path)
        points = np.array(pcd.points)
    
        mean = points.mean(axis=0)
        points = points - mean
        norm_dict[file_name] = {
            'mean': list(mean),
            'std': np.linalg.norm(points, axis=1).max()
        }
        points /= norm_dict[file_name]['std']

        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(dest_dir, file_name.replace(".ply", ".pcd")), pcd)

    with open(os.path.join(dest_dir, "normalization.json"), "w") as outfile: 
        json.dump(norm_dict, outfile)


def denormalize():
    pass # points*std + mean  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    args = parser.parse_args()

    normalize(source_dir=args.source, dest_dir=args.dest)