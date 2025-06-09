"Helpers for graph construction and manipulation taken and adapted from TorchDrug"

import torch

from torch_geometric.data import Batch
from torch_cluster import radius_graph, knn_graph


def find_isolated_nodes(num_nodes, edge_index):
    deg = torch.bincount(edge_index[1], minlength=num_nodes)
    return (deg == 0).nonzero(as_tuple=True)[0]


def build_graph(data: Batch, max_edges: int, min_residue_distance: int, radius: float, k: int):
    num_nodes = data.num_nodes
    device = data.pos.device

    # Create initial graph using radius and kNN with residue distance filtering
    edge_index_radius = radius_graph(data.pos, r=radius, batch=data.batch)
    edge_index_knn = knn_graph(data.pos, k=k, batch=data.batch)

    edge_index = torch.cat([edge_index_radius, edge_index_knn], dim=1)
    src, dst = edge_index
    residue_distances = (data.res_idx[src] - data.res_idx[dst]).abs()
    mask = residue_distances >= min_residue_distance
    edge_index = edge_index[:, mask]

    edge_index = torch.unique(edge_index, dim=1)

    # Clip to max_edges
    if edge_index.size(1) > max_edges:
        src, dst = edge_index
        edge_dist = (data.pos[src] - data.pos[dst]).norm(dim=-1)
        edge_list = list(zip(src.tolist(), dst.tolist(), edge_dist.tolist()))

        # Sort edges by distance
        edge_list.sort(key=lambda x: x[2])
        edge_list = edge_list[: int(max_edges)]

        edge_index = torch.tensor(
            [[s for s, _, _ in edge_list], [d for _, d, _ in edge_list]], dtype=torch.long, device=device
        )

    # If there are isolated nodes, connect them to their nearest neighbor (ignoring residue constraint)
    isolated_nodes = find_isolated_nodes(num_nodes=num_nodes, edge_index=edge_index)
    if len(isolated_nodes) > 0:
        extra_edges = knn_graph(data.pos, k=1, batch=data.batch)

        for node in isolated_nodes:
            candidates = extra_edges[:, extra_edges[1] == node]
            for i in range(candidates.size(1)):
                edge = candidates[:, i : i + 1]
                edge_index = torch.cat([edge_index, edge], dim=1)
                break

    return edge_index
