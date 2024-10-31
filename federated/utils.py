import torch
import numpy as np
from munch import Munch
from sklearn.metrics import f1_score

from client import *
from server import *
from ..utils.model import get_model
from ..utils.evaluation import AverageMeter, average_odds, equal_opportunity, demographic_parity
from ..utils.plot import plot_data_distribution
from .data import PartitionedDataset


# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
args = Munch # To correctly instantiate

def set_fair_order(client_dataset):
  # Extract the indices of protected group and unprotected group instances
  attr = np.array(client_dataset.data[:, 0].cpu())
  indices_1 = np.where(attr == 0)[0]
  indices_2 = np.where(attr == 1)[0]

  # Define the array of permutation indices
  perm_indices = []

  idx_1 = len(indices_1) - 1
  idx_2 = len(indices_2) - 1

  while (idx_1 >= 0) and (idx_2 >= 0):
    perm_indices.insert(0, indices_1[idx_1])
    perm_indices.insert(0, indices_2[idx_2])
    idx_1 -= 1
    idx_2 -= 1

  if (idx_1 >= 0):
    perm_indices = np.concatenate((indices_1[:idx_1 + 1], perm_indices))
  if (idx_2 >= 0):
    perm_indices = np.concatenate((indices_2[:idx_2 + 1], perm_indices))

  # Only in case of categorical protected attribute (non-binary)
  if (len(np.unique(attr)) > 2):
    all_indices = set(range(len(attr)))
    indices_1_2 = set(indices_1).union(set(indices_2))
    other_indices = np.array(list(all_indices - indices_1_2))
    perm_indices = np.concatenate((other_indices, perm_indices))

  client_dataset.data = client_dataset.data[perm_indices]
  client_dataset.targets = client_dataset.targets[perm_indices]

  return client_dataset


def evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy. """
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()

    labels_array = []
    predicted_array = []
    attr_array = []

    with torch.no_grad():
        for inputs, labels in test_loader:

            batch_size = len(labels)
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item(), batch_size)
            acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)
            attr = torch.tensor([features[0].item() for features in inputs.cpu()])

            labels_array.extend(labels.cpu())
            predicted_array.extend(predicted.cpu())
            attr_array.extend(attr.cpu())

        labels_array = torch.tensor(labels_array)
        predicted_array = torch.tensor(predicted_array)
        attr_array = torch.tensor(attr_array)

        f1 = f1_score(labels_array, predicted_array, average='macro')
        ao = average_odds(labels_array, predicted_array, attr_array)
        eod = equal_opportunity(labels_array, predicted_array, attr_array)
        dp = demographic_parity(predicted_array, attr_array)

    return loss_.avg, acc_.avg, f1, ao, eod, dp


def client_evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy. """
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()

    labels_array = []
    predicted_array = []
    attr_array = []

    with torch.no_grad():
        for inputs, labels in test_loader:

            batch_size = len(labels)
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item(), batch_size)
            attr = torch.tensor([features[0].item() for features in inputs.cpu()])

            labels_array.extend(labels.cpu())
            predicted_array.extend(predicted.cpu())
            attr_array.extend(attr.cpu())

        labels_array = torch.tensor(labels_array)
        predicted_array = torch.tensor(predicted_array)
        attr_array = torch.tensor(attr_array)

        ao = average_odds(labels_array, predicted_array, attr_array)

    return loss_.avg, ao



# Prepare federated data
def partition_dataset(args, plot_dist=True):
  # Download raw dataset and partition them according to given configuration
  fed_data = PartitionedDataset(root=args.root,
                        path=args.path,
                        num_clients=args.num_clients,
                        partition=args.partition,
                        alpha=args.alpha,
                        device=device,
                        seed=args.part_seed,
                        preprocess=args.preprocess,
                        download=True,
                        verbose=True,
                        transform=None)

  # Set the federated dataset
  args.fed_dataset = fed_data

  if (plot_dist):

    # Plot data distribution
    plot_data_distribution(args)

# Get client local training pipeline
def get_client_trainer(model, num_clients, cuda, algorithm):

  if (algorithm == "fedfair"):
    trainer = FedFairSerialClientTrainer(model=model, num_clients=num_clients, cuda=cuda)
  elif (algorithm == "fedfairuniform"):
    trainer = FedFairUniformSerialClientTrainer(model=model, num_clients=num_clients, cuda=cuda)
  elif (algorithm == "fairorder"):
    trainer = FairOrderSerialClientTrainer(model=model, num_clients=num_clients, cuda=cuda)
  else:
    trainer = FedAvgSerialClientTrainer(model=model, num_clients=num_clients, cuda=cuda)

  trainer.setup_dataset(args.fed_dataset)
  trainer.setup_optim(args.epochs, args.batch_size, args.lr)

  return trainer

# Get server global aggregation pipeline
def get_server_handler(model, global_round, sample_ratio, cuda, algorithm):

  if (algorithm == "fedfair"):
    handler = FedFairServerHandler(model=model, global_round=global_round, sample_ratio=sample_ratio, cuda=cuda, beta = args.beta)
    handler.setup_optim(args.d)
  elif (algorithm == "fedfairuniform"):
    handler = FedFairUniformServerHandler(model=model, global_round=global_round, sample_ratio=sample_ratio, cuda=cuda, beta = args.beta)
    handler.setup_optim(args.d)
  else:
    handler = FedAvgServerHandler(model=model, global_round=global_round, sample_ratio=sample_ratio, cuda=cuda)

  return handler


def fed_setting(args):

  # Initialize the model
  args.model = get_model(init_seed=args.init_seed).to(device)

  # Define client local training
  trainer = get_client_trainer(model=args.model, num_clients=args.num_clients, cuda=args.cuda, algorithm=args.algorithm)

  # Define server global aggregation
  handler = get_server_handler(model=args.model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda, algorithm=args.algorithm)

  return trainer, handler