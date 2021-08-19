import open3d as o3d
import os
import numpy as np
import argparse
from jinja2 import Template
from visualizer import PointCloudVisualizer
from utils.utils import random_point_sampler


    
def generate_preview(pcd_dir, annot_dir, dest_dir):
    """
    Generate a HTML containing annotated gifs

    Parameters:
        pcd_dir (str): The directory containing pointclouds (.pcd)
        annot_dir (str): The directory containing annotation files (.npy)
        dest_dir (str): The directory to save result in
    """

    if not os.path.isdir(pcd_dir) or not os.path.isdir(annot_dir):
        raise Exception('Please Enter path that exists!')  

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print("Destination directory created!")
    
    visualizer = PointCloudVisualizer()
    images = []
    
    for file_name in os.listdir(annot_dir):
        if not file_name.endswith('.npy'):
            continue
            
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, file_name.replace('.npy', '.pcd')))
        points = np.array(pcd.points)
        labels = np.load(os.path.join(annot_dir, file_name))

        points, labels = random_point_sampler(points, labels)

        images.append(os.path.join(dest_dir, file_name.replace('.npy', '.gif')))
        visualizer.save_visualization(points, labels, images[-1])

    generate_html(images, dest_dir)


def generate_html(images, dest_dir):
    with open('templates/preview_pointclouds.html') as f:
        template = Template(f.read())
    
    dest_file_path = os.path.join(dest_dir, 'preview.html')
    with open(dest_file_path, 'w') as f:
        f.write(template.render(images=images, num_rows=int(np.ceil(len(images)/4))))
    
    print(f"You can see the preview in {dest_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_dir', type=str, required=True)
    parser.add_argument('--annot_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    args = parser.parse_args()
    generate_preview(args.pcd_dir, args.annot_dir, args.dest_dir)