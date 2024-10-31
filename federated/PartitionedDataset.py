import os
import torch
import numpy as np
from ..utils.data import get_data, get_extra_samples, ClientDataset
from torch.utils.data import DataLoader
from fedlab.contrib.dataset.basic_dataset import FedDataset
from fedartml import SplitAsFederatedData
from fedartml.function_base import hellinger_distance
from .utils import set_fair_order


class PartitionedDataset(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.


    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        num_clients (int): Number of clients.
        seed (int): Random seed.
        partition (str): Partition scheme name.
        alpha (float): Dirichlet distribution parameter.
        device (torch.device): The available device.
        download (bool): Whether to download the raw dataset.
        preprocess (bool): Whether to preprocess the dataset.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self,
                 root,
                 path,
                 num_clients,
                 seed,
                 partition,
                 alpha,
                 device,
                 download=True,
                 preprocess=False,
                 verbose=True,
                 transform=None,
                 target_transform=None) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.seed = seed
        self.partition = partition
        self.alpha = alpha
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

        self.data_distribution = None
        self.hd = None

        if preprocess:
            self.preprocess(partition=partition,
                            alpha=alpha,
                            device=device,
                            verbose=verbose,
                            seed=seed,
                            download=download,
                            transform=transform,
                            target_transform=target_transform)

    def preprocess(self,
                   partition,
                   alpha,
                   device,
                   verbose=True,
                   seed=None,
                   download=True,
                   transform=None,
                   target_transform=None):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        self.download = download

        if os.path.exists(self.path) is not True:
            os.makedirs(self.path, exist_ok=True)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        # Instantiate a SplitAsFederatedData object
        my_federater = SplitAsFederatedData(random_state = self.seed)

        # Get training global features and labels
        data = get_data()
        X_train_glob, y_train_glob = data[:2]
        X_extra, y_extra = get_extra_samples(X_train_glob, y_train_glob)

        if (self.partition == "attr"):

            # Get the array relative to the protected attribute
            attr_array = X_train_glob[:, 0]

            # Get feature federated dataset skewed on sex feature from centralized dataset
            clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances, clients_glob_spatemp_dic = my_federater.create_clients(
                image_list = X_train_glob,
                label_list = y_train_glob,
                num_clients = self.num_clients,
                prefix_cli='Local_node',
                method = "no-label-skew",
                spa_temp_skew_method = "st-dirichlet",
                alpha_spa_temp = self.alpha,
                spa_temp_var = attr_array
            )

        # Default method: dirichlet
        else:
            # Get label skewed federated dataset from centralized dataset
            clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = my_federater.create_clients(
                image_list = X_train_glob,
                label_list = y_train_glob,
                num_clients = self.num_clients,
                prefix_cli='Local_node',
                method = "dirichlet",
                alpha = self.alpha
            )
            self.hd = distances['without_class_completion']['hellinger']

        clients_train = clients_glob_dic['without_class_completion']
        client_data_sizes = []

        for cid, client_name in enumerate(clients_train.keys()):
          client_train = clients_train[client_name]
          client_data_sizes.append(len(client_train))

          client_features = np.array([item[0] for item in client_train])
          client_features = np.concatenate((client_features, X_extra))

          client_labels = np.array([item[1] for item in client_train])
          client_labels = np.concatenate((client_labels, y_extra))

          client_dataset = ClientDataset(client_features, client_labels, self.device)

          torch.save(
              client_dataset,
              os.path.join(self.path, "train", "data{}.pkl".format(cid))
          )

        self.data_distribution = client_data_sizes / np.sum(client_data_sizes)

        if (self.partition == "attr"):

          pctg_distr = []

          for cid in range(self.num_clients):

            # Get the dataset of each client
            dataset =  self.get_dataset(cid)

            # Count the number of data points per group
            data_size = len(dataset.data)
            count_0 = np.sum(np.array(dataset.data[:, 0].cpu()) == 0)
            count_1 = np.sum(np.array(dataset.data[:, 0].cpu()) == 1)

            # Calculate the distribution
            client_distr = [count_0 / data_size, count_1 / data_size]
            pctg_distr.append(client_distr)

          self.hd = hellinger_distance(pctg_distr)


    def get_dataset(self, cid, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
             cid (int): client id
             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(cid)), weights_only=False)
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train", fair_order=False):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
            rr_seed (int, optional): Random Reshuffling seed, if employed. Default as ``"None"``.
        """
        dataset = self.get_dataset(cid, type)
        is_train = (type == "train")

        if (fair_order):
          dataset = set_fair_order(dataset)
          data_loader = DataLoader(dataset, batch_size, shuffle=False)
        else:
          data_loader = DataLoader(dataset, batch_size, shuffle=is_train)

        return data_loader

    def get_data_distribution(self):
      """Return distribution of federated dataset."""

      return self.data_distribution

    def get_hd(self):
      """Return Hellinger distance of federated dataset."""

      return self.hd