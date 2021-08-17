import argparse
import os
import json
import numpy as np


object_encode = {'plant': 1, 'soil': 0}


def convert_supervisely_format(source_dir, dest_dir):
    """
    Converts supervisely annotations formats to .npy format

    Parameters:
        source_dir (str): The directory containing supervisely annotation files
        dest_dir (str): The directory to save results in
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for file_name in os.listdir(source_dir):
        if not file_name.endswith('.pcd.json'):
            continue

        with open(os.path.join(source_dir, file_name)) as f:
            dic = json.load(f)
        objects = {}
        for obj in dic['objects']:
            objects[obj['id']] = obj['classTitle'].lower()
        
        classes = np.zeros(max([max(fig['geometry']['indices']) for fig in dic['figures']]) + 1)
        for fig in dic['figures']:
            obj_id = fig['objectId']
            indices = fig['geometry']['indices']
            classes[indices] = object_encode[objects[obj_id].lower()]
        
        with open(os.path.join(dest_dir, file_name.replace('.pcd.json', '.npy')), 'wb') as f:
            np.save(f, np.array([1, 2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    args = parser.parse_args()

    convert_supervisely_format(args.source, args.dest)