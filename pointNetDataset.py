import pandas as pd
from torch.utils.data import Dataset
import torch
#import pyvista as pv
import numpy as np
import torch.nn.functional as F

import torch.utils.data as torch_data

import open3d as o3d


class ModelNet(Dataset):
    classes = {
        'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4,
        'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9,
    }

    def __init__(self, metadata_path: str, split=None, file_format='ply', num_points=10000, transform=None):
        super().__init__()

        # Load the metadata from the specified path
        self.metadata = pd.read_parquet(metadata_path)

        # Make sure the expected fields exist in the metadata
        expected_fields = ['path', 'split', 'label', 'label_id', 
                           'orientation_class', 'orientation_class_id', 'rot_x', 'rot_y', 'rot_z']
        for expected_field in expected_fields:
            assert expected_field in self.metadata.columns

        # Subset on file format (e.g., 'npy')
        self.file_format = file_format
        assert file_format == 'ply'
        self.metadata = self.metadata[self.metadata.path.str.contains(file_format)]

        # Subset on split (e.g., 'test' or 'train')
        if split is None:
            split = 'test|train'  # Use regex or
        else:
            assert split == 'test' or split == 'train'
        self.metadata = self.metadata[self.metadata.split.str.contains(split)]

        # Get the number of unique labels
        self.n_classes = self.metadata.label_id.unique().size
        self.num_points = num_points
        self.transform = transform

    def __getitem__(self, idx):
        file = self.metadata.iloc[idx]
        mesh = o3d.io.read_triangle_mesh(file.path)  # Load the mesh using Open3D

        # Use mesh_to_point_cloud to convert the mesh to a point cloud
        pc = mesh_to_point_cloud(mesh, num_points=self.num_points)

        if self.transform is not None:
            pc = self.transform(pc)

        # Ensure all point clouds have the same number of points
        if pc.shape[0] < self.num_points:
            # Pad with zeros if there are fewer points
            padding = np.zeros((self.num_points - pc.shape[0], 3), dtype=np.float32)
            pc = np.concatenate([pc, padding], axis=0)
        elif pc.shape[0] > self.num_points:
            # Randomly subsample if there are more points
            choice_indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
            pc = pc[choice_indices]

        return pc, file.label_id  # Return the point cloud and label_id

    def __len__(self):
        # Metadata was already subset during initialization
        return len(self.metadata)

def mesh_to_point_cloud(mesh, num_points=None, random_sampling=True):
    vertices = np.array(mesh.vertices)

    if num_points is not None:
        if random_sampling:
            num_vertices = vertices.shape[0]
            point_indices = np.random.choice(num_vertices, min(num_points, num_vertices), replace=False)
            sampled_points = vertices[point_indices]
        else:
            pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
            return np.array(pcd.points)
    else:
        sampled_points = vertices

    return sampled_points
    


# if we rotate a point class we have the point cloud

class RandomJitterTransform(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        """ Randomly jitter points. Jittering is per point.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        N, C = data.shape
        assert (self.clip > 0)
        jittered_data = np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip)
        jittered_data += data  # Add jitter without changing the number of points
        return np.float32(jittered_data)

class RandomRotateTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        """ Randomly rotate the point clouds to augment the dataset
            Rotation is per shape based along ANY direction
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """

        # Generate a random rotation matrix
        rotation_angle = np.random.uniform() * 2 * np.pi

        # Create the rotation matrix around the y-axis
        rotation_matrix_y = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                                      [0, 1, 0],
                                      [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]])

        # Apply the rotation to the input data
        rotated_data = np.dot(data, rotation_matrix_y)

        return np.float32(rotated_data)

class ScaleTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        data = (data - data.min(  axis=0)) / (data.max(axis=0) - data.min(axis=0))
        return np.float32(data)
