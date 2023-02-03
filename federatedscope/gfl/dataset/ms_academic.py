import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data

def build_graph(path, filename):
    file_name=osp.join(path, filename)
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])#.toarray()

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape']).toarray()
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if attr_matrix.all() is None:
            raise RuntimeError()
        else:
            min_values=np.min(attr_matrix,axis=0)
            max_values=np.max(attr_matrix,axis=0)+1e-15
            for i in range(len(attr_matrix)):
                attr_matrix[i]-=min_values
                attr_matrix[i]/=max_values

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape']).toarray()
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        edge_index=adj_matrix.nonzero()

        data = Data(x=torch.tensor(attr_matrix, dtype=torch.float32),
                edge_index=torch.tensor(edge_index,dtype=torch.long),
                y=torch.tensor(labels, dtype=torch.int64))

        # order edge list and remove duplicates if any.
        data.coalesce()




        return data


class MSAcademic(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self,
                 root,
                 splits=[0.5, 0.2, 0.3],
                 transform=None,
                 pre_transform=None):
        self.name = 'ms_academic'
        self._customized_splits = splits
        super(MSAcademic, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['ms_academic_cs.npz']
        return names

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    def download(self):
        # Download to `self.raw_dir`.
        url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz'
        download_url(f'{url}', self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data = build_graph(self.raw_dir, self.raw_file_names[0])

        data_list_w_masks = []
        indices = torch.randperm(data.num_nodes)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[indices[:round(self._customized_splits[0] *
                                       len(data.y))]] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[
            indices[round(self._customized_splits[0] *
                          len(data.y)):round((self._customized_splits[0] +
                                              self._customized_splits[1]) *
                                             len(data.y))]] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[indices[round((self._customized_splits[0] +
                                      self._customized_splits[1]) *
                                     len(data.y)):]] = True
        data_list_w_masks.append(data)
        data_list = data_list_w_masks

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
