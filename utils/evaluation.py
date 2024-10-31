import torch
import numpy as np
from munch import Munch
from fedartml.function_base import hellinger_distance
from .model import get_model
from ..federated.utils import evaluate
from ..federated import PartitionedDataset

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
args = Munch # To correctly instantiate

# Define the AverageMeter class
class AverageMeter(object):
    """Record metrics information"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Function for the Average Odds computation
def average_odds(y, yhat, attr):

  # Convert to floating point tensors
  y, yhat = y.float(), yhat.float()

  # Compute TRP and FPR for the first group
  tpr_0 = torch.mean(yhat[(y == 1) & (attr == 0)])
  fpr_0 = torch.mean(yhat[(y == 0) & (attr == 0)])

  # Compute TPR and FPR for the second group
  tpr_1 = torch.mean(yhat[(y == 1) & (attr == 1)])
  fpr_1 = torch.mean(yhat[(y == 0) & (attr == 1)])

  # Compute Average Odds
  average_odds = 0.5 * (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1))

  return average_odds.item()

# Function for the Equal Opportunity computation
def equal_opportunity(y, yhat, attr):

    # Convert to floating point tensors
    y, yhat = y.float(), yhat.float()

    # Compute TPR for the first group
    tpr_0 = torch.mean(yhat[(y == 1) & (attr == 0)])

    # Compute TPR for the second group
    tpr_1 = torch.mean(yhat[(y == 1) & (attr == 1)])

    # Compute Equal Opportunity Difference
    eopp = abs(tpr_0 - tpr_1)

    return eopp.item()

# Function for the Demographic Parity computation
def demographic_parity(yhat, attr):

    # Convert to floating point tensor
    yhat = yhat.float()

    # Compute the true prediction probability for each group
    true_prob_0 = torch.mean(yhat[attr == 0])
    true_prob_1 = torch.mean(yhat[attr == 1])

    # Compute Demographic Parity Difference
    dp = abs(true_prob_0 - true_prob_1)

    return dp.item()

# Function to evaluate the final global model in different non-IIDness conditions
def evaluate_global_model(alpha_values, save_folder, device):

  loss_values = []
  acc_values = []
  f1_values = []
  ao_values = []
  eopp_values = []
  dp_values = []

  for alpha_value in alpha_values:
    save_path = save_folder + f"alpha_{alpha_value}.pth"
    model = get_model(ckpt=save_path)

    loss, acc, f1, ao, eopp, dp = evaluate(model, torch.nn.CrossEntropyLoss(), args.test_loader)
    loss_values.append(loss)
    acc_values.append(acc)
    f1_values.append(f1)
    ao_values.append(ao)
    eopp_values.append(eopp)
    dp_values.append(dp)

  return loss_values, acc_values, f1_values, ao_values, eopp_values, dp_values


# Sample num_seeds random seeds from the full int64 range
def sample_seeds(num_seeds):

  np.random.seed(42)

  seeds = set()
  while len(seeds) < num_seeds:
      seeds.update(np.random.randint(0, 2**32, size=num_seeds - len(seeds)))

  seeds = np.array(list(seeds))

  return seeds

# Returns a dictionary with alpha values as dictionary keys and HDs as dictionary values
def get_alpha_dict(alpha_values):

  alpha_dict = {}

  for alpha_value in alpha_values:

    # Get the federated dataset for the current alpha value
    fed_acs = PartitionedDataset(root=args.root,
                        path=args.path,
                        num_clients=args.num_clients,
                        partition=args.partition,
                        alpha=alpha_value,
                        device=device,
                        seed=args.part_seed,
                        preprocess=args.preprocess,
                        download=True,
                        verbose=True,
                        transform=None)

    # Get Hellinger distance
    hd = fed_acs.get_hd()

    # Associate the rounded HD to the alpha value in the dictionary
    alpha_dict[alpha_value] = round(hd, 2)

  return alpha_dict


# Returns an array containing an array of four category counts for each client
def get_client_counts():

  client_counts = []

  for cid in range(args.num_clients):
      client_dataset = args.fed_dataset.get_dataset(cid)
      groups, labels = client_dataset.data[:, 0], client_dataset.targets

      # Category order (group, label): (0,0), (0,1), (1,0), (1,1)
      client_count = np.zeros(4)
      for x, y in zip(groups, labels):
          index = int(x) * 2 + int(y)
          client_count[index] += 1
      client_counts.append(client_count)

  client_counts = np.array(client_counts)

  return client_counts

# Function extracting counts according to protected attribute and label
def extract_counts(client_counts):
  female_counts = []
  male_counts = []
  low_income_counts = []
  high_income_counts = []

  for client_count in client_counts:
      female_counts.append(client_count[0] + client_count[1])
      male_counts.append(client_count[2] + client_count[3])
      low_income_counts.append(client_count[0] + client_count[2])
      high_income_counts.append(client_count[1] + client_count[3])

  return female_counts, male_counts, low_income_counts, high_income_counts

# Function to get HD between labels in the protected attribute skew case
def get_label_hd(args):

  pctg_distr = []

  for cid in range(args.num_clients):

    # Get the dataset of each client
    dataset =  args.fed_dataset.get_dataset(cid)

    # Count the number of data points per label
    data_size = len(dataset.targets)
    count_0 = np.sum(np.array(dataset.targets.cpu()) == 0)
    count_1 = np.sum(np.array(dataset.targets.cpu()) == 1)

    # Calculate the distribution
    client_distr = [count_0 / data_size, count_1 / data_size]
    pctg_distr.append(client_distr)

  hd = hellinger_distance(pctg_distr)

  return hd

# Function to get HD between protected attributes in the label skew case
def get_attr_hd(args):

  pctg_distr = []

  for cid in range(args.num_clients):

    # Get the dataset of each client
    dataset =  args.fed_dataset.get_dataset(cid)

    # Count the number of data points per label
    data_size = len(dataset.data)
    count_0 = np.sum(np.array(dataset.data[:, 0].cpu()) == 0)
    count_1 = np.sum(np.array(dataset.data[:, 0].cpu()) == 1)

    # Calculate the distribution
    client_distr = [count_0 / data_size, count_1 / data_size]
    pctg_distr.append(client_distr)

  hd = hellinger_distance(pctg_distr)

  return hd