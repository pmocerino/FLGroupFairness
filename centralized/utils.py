import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import CentralizedTrainer
from ..utils.model import get_model
from ..utils.data import ClientDataset
from ..utils.evaluation import sample_seeds
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

# Global variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Function defining the training of a specific model instance
def centralized_train(data, device, init_seed=42, rr_seed=42, num_epochs=300):

  # Get the centralized model
  c_model = get_model(init_seed=init_seed).to(device)

  # Define the centralized trainer
  c_trainer = CentralizedTrainer(c_model, device=device)

  # Get training features and labels
  X_train, y_train = data[:2]
  trainset = ClientDataset(X_train, y_train, device)
  if (rr_seed is not None):
    trainset.data, trainset.targets = shuffle(trainset.data, trainset.targets, random_state=rr_seed)
  train_loader = DataLoader(trainset, batch_size=128, shuffle=True)

  # Train on whole train dataset
  c_model = c_trainer.train(train_loader, epochs=num_epochs)

  return c_trainer

# Function defining different runs of model training
def centralized_baseline(data, device, models_path, num_runs=50):

  # Sample num_runs random initialization seeds from the full int64 range
  seeds = sample_seeds(num_runs)

  for run_idx in range(num_runs):

    # Allow initialization and Random Reshuffling variablity
    c_trainer = centralized_train(data, device, init_seed=seeds[run_idx], rr_seed=seeds[run_idx])

    # Save the final model
    torch.save(c_trainer.model, os.path.join(models_path, f'model_{run_idx}'))


# Function evaluating the performance of the models obtained from the runs
def baseline_evaluation(test_loader, models_path, values_path, num_runs=50):

  # Record metrics across runs
  loss_values, acc_values, f1_values, ao_values, eopp_values, dp_values = [], [], [], [], [], []

  for run_idx in range(num_runs):

    # Load the model and define the trainer
    c_model = get_model(ckpt= os.path.join(models_path, f"model_{run_idx}"))
    cuda = device.type == 'cuda'
    c_trainer = CentralizedTrainer(c_model, cuda=cuda, device=device)

    # Evaluate the trained model
    loss, acc, f1, ao, eopp, dp = c_trainer.evaluate(test_loader)

    # Record the obtained values
    loss_values.append(loss)
    acc_values.append(acc)
    f1_values.append(f1)
    ao_values.append(ao)
    eopp_values.append(eopp)
    dp_values.append(dp)

  # Save arrays to .npy files
  np.save(os.path.join(values_path, 'loss_values.npy'), np.array(loss_values))
  np.save(os.path.join(values_path, 'acc_values.npy'), np.array(acc_values))
  np.save(os.path.join(values_path, 'f1_values.npy'), np.array(f1_values))
  np.save(os.path.join(values_path, 'ao_values.npy'), np.array(ao_values))
  np.save(os.path.join(values_path, 'eopp_values.npy'), np.array(eopp_values))
  np.save(os.path.join(values_path, 'dp_values.npy'), np.array(dp_values))


# Function to obtain the metrics values of centralized baseline
def get_baseline_metrics(baseline_path, mean=True):

  if os.path.exists(baseline_path):
    loss_baseline = np.load(os.path.join(baseline_path, 'loss_values.npy'))
    acc_baseline = np.load(os.path.join(baseline_path, 'acc_values.npy'))
    f1_baseline = np.load(os.path.join(baseline_path, 'f1_values.npy'))
    ao_baseline = np.load(os.path.join(baseline_path, 'ao_values.npy'))
    eopp_baseline = np.load(os.path.join(baseline_path, 'eopp_values.npy'))
    dp_baseline = np.load(os.path.join(baseline_path, 'dp_values.npy'))

    if (mean):
      loss_baseline = np.mean(loss_baseline)
      acc_baseline = np.mean(acc_baseline)
      f1_baseline = np.mean(f1_baseline)
      ao_baseline = np.mean(ao_baseline)
      eopp_baseline = np.mean(eopp_baseline)
      dp_baseline = np.mean(dp_baseline)

    return loss_baseline, acc_baseline, f1_baseline, ao_baseline, eopp_baseline, dp_baseline

  else:
    print("Invalid baseline path.")
    return

# Function to plot the metrics range over different runs
def plot_centralized_baseline(baseline_metrics):
    metrics = ['Loss', 'Accuracy', 'F1 Score', 'Average Odds', 'Equal Opportunity', 'Demographic Parity']

    # Use default style
    plt.style.use('default')

    # Create subplots and add a title
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Centralized Baseline', fontsize=24, fontweight='bold', y=1.01) #, y=1.02)

    # Iterate over the axes and metrics to create the boxplots
    for idx, ax in enumerate(axs.flat):
        ax.set_title(metrics[idx], fontsize=18, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Boxplot customization
        ax.boxplot(baseline_metrics[idx], showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='skyblue', color='navy'),
                   medianprops=dict(color='darkorange', linewidth=2))

        # Add grid lines with light color and dashed style
        ax.grid(True, axis='y', linestyle='--', color='grey', alpha=0.6)

        # Remove x-axis ticks
        ax.set_xticks([])

        # Customize y-ticks and labels
        yticks = ax.get_yticks()
        ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))

        if metrics[idx] == 'Loss':
            ax.set_yticklabels([f'{y:.3f}' for y in yticks], fontsize=14)
        else:
            ax.set_yticklabels([f'{y*100:.1f}%' for y in yticks], fontsize=14)

    # Display the plot
    plt.tight_layout()
    plt.show()