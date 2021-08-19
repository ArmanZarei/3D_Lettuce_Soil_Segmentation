import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PointCloudVisualizer:
    def __R_x(self, theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    
    def __R_y(self, theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    def __R_z(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    def save_visualization(self, points, labels, file_path):
        colors = np.array(['#4E342E' for i in range(labels.size)])
        colors[labels == 1] = '#2E7D32'
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.grid(True)
          
        ims = []
        for rot in [self.__R_z]:
            for theta in np.linspace(0, 2*np.pi, 40):
                rp = rot(theta).dot(points.T).T
                ims.append([ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], c=colors, s=2)])
        
        ani = animation.ArtistAnimation(fig, ims, blit=True)
        ani.save(file_path, writer='pillow', fps=10)