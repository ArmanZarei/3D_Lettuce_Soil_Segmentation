import numpy as np


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