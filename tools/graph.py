import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix


def build_edge_index(positions, n_closest=5):
    """
    Constructs a distance map using physical montage coordinates and converts 
    it into an adjacency matrix with `n_closest` neighbors. 
    Returns the edge index and weights for PyTorch Geometric models.
    """
    # 1. Coordinate array & initialize distance matrix
    coordinates = np.array(list(positions.values()))
    n_channels = len(positions)
    distance_matrix = np.zeros((n_channels, n_channels))

    # 2. Compute distance matrix
    for row in range(coordinates.shape[0]):
        for column in range(coordinates.shape[0]):
            displacement = coordinates[row] - coordinates[column]
            distance = np.sqrt(np.dot(displacement, displacement))
            distance_matrix[row, column] = distance

    # 3. Create adjacency matrix mask based on closest N neighbors
    adjacency_matrix_indices = np.argsort(distance_matrix)[:, 1:n_closest + 1]
    adjacency_matrix_mask = np.zeros((n_channels, n_channels), dtype=int)

    # Set 1 for row-wise neighbor matches, 0 otherwise
    for row in range(n_channels):
        adjacency_matrix_mask[row, adjacency_matrix_indices[row]] = 1

    # 4. Filter and normalize adjacency matrix
    adjacency_matrix = distance_matrix * adjacency_matrix_mask
    with np.errstate(divide='ignore'):
        adjacency_matrix = np.where(
            adjacency_matrix > 0, 1 / adjacency_matrix, 0)

    if np.max(adjacency_matrix) > 0:
        adjacency_matrix /= np.max(adjacency_matrix)

    # 5. Convert to PyTorch Geometric COO format
    sparse_adj = csr_matrix(adjacency_matrix)
    edge_index, edge_weight = from_scipy_sparse_matrix(sparse_adj)

    # Ensure correct tensor format
    edge_index = edge_index.to(torch.long)
    edge_weight = edge_weight.to(torch.float)

    return edge_index, edge_weight
