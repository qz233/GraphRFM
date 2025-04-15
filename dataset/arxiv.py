import torch

from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset

# Compatible for pytorch 2.6 load
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


def get_arxiv_dataset(kind, path="./cache"):
    dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = path)
    idx_split = dataset.get_idx_split()

    dataset.edge_index = to_undirected(dataset.edge_index)
    dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
    dataset.edge_index, _ = add_self_loops(dataset.edge_index, num_nodes=dataset.x.shape[0])
    dataset.y = dataset.y[:, 0]
    if kind == "arxiv-year":
        q1 = torch.quantile(dataset.y[:, 0].float(), 0.8)
        q2 = torch.quantile(dataset.y[:, 0].float(), 0.9)

        train_idx = torch.argwhere(dataset.y[:, 0] < q1)[:, 0]
        val_idx = torch.argwhere((dataset.y[:, 0] >= q1) & (dataset.y[:, 0] < q2))[:, 0]
        test_idx = torch.argwhere(dataset.y[:, 0] >= q2)[:, 0]
        idx_split = {"train": train_idx, "valid": val_idx, "test": test_idx}
        print(f"train: [1, {q1}), valid: [{q1}, {q2}), test: [{q2})")

        dataset.y = torch.bucketize(dataset.node_year, torch.LongTensor([2014, 2016, 2018, 2019]), right=True)
        dataset.__num_classes__ = 5
    return dataset, idx_split