import numpy as np
import random


class PointSampler(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, input):
        assert input[0].shape[0] >= self.output_size

        indices = np.random.choice(input[0].shape[0], self.output_size)
        return (input[0][indices], input[1][indices])


class RandomRotation(object):
    def __call__(self, input):
        pointcloud, categories = input[0], input[1]
        theta = random.random() * 2 * np.pi # Rotation angle
        rotation_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        rotated_pointcloud = rotation_mat.dot(pointcloud.T).T

        return (rotated_pointcloud, categories)