"Helpers for graph construction and manipulation taken and adapted from TorchDrug"

import torch

from torch_cluster import knn_graph, radius_graph
from torch_geometric.data import Batch


def find_isolated_nodes(data: Batch, edge_index):
    """
    Find isolated nodes in the graph.

    Parameters:
        data (Batch): n graph(s) with node positions
        edge_index (Tensor): edge index of shape :math:`(2, |E|)`

    Returns:
        Tensor: indices of isolated nodes
    """
    num_nodes = data.num_nodes
    connected_nodes = edge_index.unique().long()
    return torch.tensor(list(set(range(num_nodes)) - set(connected_nodes.tolist())))


def union(edge_index1, edge_index2):
    """
    Compute the union of two edge index tensors.

    Parameters:
        edge_index1 (Tensor): edge index of shape (2, |E1|)
        edge_index2 (Tensor): edge index of shape (2, |E2|)

    Returns:
        Tensor: union of edges of shape (2, |E_union|)
    """
    combined_edges = torch.cat([edge_index1, edge_index2], dim=1)
    unique_edges = torch.unique(combined_edges, dim=1)

    return unique_edges


class KNNGraph(torch.nn.Module):
    """
    Construct edges between each node and its nearest neighbors.

    Parameters:
        k (int, optional): number of neighbors
        min_distance (int, optional): minimum distance between the residues of two nodes
        max_distance (int, optional): maximum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, k=10, min_distance=5, max_distance=None):
        super().__init__()
        self.k = k
        self.min_distance = min_distance
        self.max_distance = max_distance

    def forward(self, data: Batch):
        """
        Return KNN edges constructed from the input graph.

        Parameters:
            data (Batch): n graph(s) with node positions

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`
        """
        edge_list = knn_graph(data.pos, k=self.k, batch=data.batch).t()

        if self.min_distance > 0:
            dest_idx, src_idx = edge_list.t()
            mask = (data.res_idx[dest_idx] - data.res_idx[src_idx]).abs() < self.min_distance
            edge_list = edge_list[~mask]

        if self.max_distance:
            dest_idx, src_idx = edge_list.t()
            mask = (data.res_idx[dest_idx] - data.res_idx[src_idx]).abs() > self.max_distance
            edge_list = edge_list[~mask]

        dest_idx, src_idx = edge_list.t()
        mask = (data.pos[dest_idx] - data.pos[src_idx]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list


class SpatialGraph(torch.nn.Module):
    """
    Construct edges between nodes within a specified radius.

    Parameters:
        r (float, optional): spatial radius
        min_distance (int, optional): minimum distance between the residues of two nodes
        max_distance (int, optional): maximum distance between the residues of two nodes
        max_num_neighbors (int, optional): maximum number of neighbors to connect
    """

    eps = 1e-10

    def __init__(self, r=5, min_distance=5, max_distance=None, max_num_neighbors=32):
        super().__init__()
        self.r = r
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_num_neighbors = max_num_neighbors

    def forward(self, data: Batch):
        """
        Return spatial radius edges constructed based on the input graph.

        Parameters:
            data (Batch): n graph(s) with node positions

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`
        """

        edge_list = radius_graph(data.pos, r=self.r, batch=data.batch, max_num_neighbors=self.max_num_neighbors).t()

        if self.min_distance > 0:
            dest_idx, src_idx = edge_list.t()
            mask = (data.res_idx[dest_idx] - data.res_idx[src_idx]).abs() < self.min_distance
            edge_list = edge_list[~mask]

        if self.max_distance:
            dest_idx, src_idx = edge_list.t()
            mask = (data.res_idx[dest_idx] - data.res_idx[src_idx]).abs() > self.max_distance
            edge_list = edge_list[~mask]

        dest_idx, src_idx = edge_list.t()
        mask = (data.pos[dest_idx] - data.pos[src_idx]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list
