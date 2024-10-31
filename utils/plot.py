import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from evaluation import extract_counts

# Function to print federated execution configuration
def print_config(args):

  execution_config = f"""
  Algorithm: {args.algorithm}

  [SIMULATION CONFIGURATION]
  Number of clients: {args.num_clients}
  Partition: {args.partition}
  Alpha: {args.alpha}
  Partition seed: {args.part_seed}
  Preprocess: {args.preprocess}
  Cuda: {args.cuda}

  [CLIENT CONFIGURATION]
  Epochs: {args.epochs}
  Batch size: {args.batch_size}
  Learning rate: {args.lr}

  [SERVER CONFIGURATION]
  Communication rounds: {args.com_round}
  Sample ratio: {args.sample_ratio}
  """

  print(execution_config)


# Function for federated data distribution plot
def plot_data_distribution(args):

  # Get federated dataset distribution
  data_distribution = args.fed_dataset.get_data_distribution()

  # Calculate positions for the bar plot
  positions = np.arange(len(data_distribution))

  # Create the plot with a specific figure size
  plt.figure(figsize=(10, 6))

  # Plot the bars with the colormap
  bars = plt.bar(positions, data_distribution, width=1.0, align='center', edgecolor='k', color='#1f77b4')

  # Customize the plot
  plt.title(f'Distribution of data among {args.num_clients} federated clients ({args.partition}~Dir(alpha = {args.alpha}))')
  plt.xlabel('Client ID')
  plt.ylabel('Data Fraction')

  # Display the plot
  plt.tight_layout()
  plt.show()


# Function plotting the data distribution of clients
def plot_distributed_data(args, client_counts):

    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=False)
    titles = [
        f'Client Data Distributions ({args.partition}~Dir(alpha = {args.alpha}))',
        f'Client Label Distributions ({args.partition}~Dir(alpha = {args.alpha}))',
        f'Client Protected Attribute Distributions ({args.partition}~Dir(alpha = {args.alpha}))'
    ]

    # Set up bar positions
    bar_width = 0.7
    bar_positions = np.arange(args.num_clients)

    for idx, ax in enumerate(axes):
        if idx == 0:
            categories = ['Female Salary < 50K$', 'Female Salary ≥ 50K$', 'Male Salary < 50K$', 'Male Salary ≥ 50K$']
            colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
            count_category = [client_counts[:, i] for i in range(len(categories))]

        elif idx == 1:
            categories = ['Salary < 50K$', 'Salary ≥ 50K$']
            colors = ['#d62728', '#1f77b4']
            count_category_0 = client_counts[:, 0] + client_counts[:, 2]
            count_category_1 = client_counts[:, 1] + client_counts[:, 3]
            count_category = [count_category_0, count_category_1]

        else:
            categories = ['Female', 'Male']
            colors = ['#d62728', '#1f77b4']
            count_category_0 = client_counts[:, 0] + client_counts[:, 1]
            count_category_1 = client_counts[:, 2] + client_counts[:, 3]
            count_category = [count_category_0, count_category_1]

        # Plot each category
        bottoms = np.zeros(args.num_clients)
        for i, category in enumerate(categories):
            ax.bar(bar_positions, count_category[i], bar_width, label=category, color=colors[i], bottom=bottoms, edgecolor='black')
            bottoms += count_category[i]

        # Set labels and title
        ax.set_ylabel('Number Of Data Points', fontsize=12)
        ax.set_title(titles[idx], fontsize=14)

        # Set x-ticks
        tick_positions = np.arange(args.num_clients*args.sample_ratio, args.num_clients, args.num_clients*args.sample_ratio)
        ax.set_xticks(tick_positions)
        ax.set_xlabel('Clients', fontsize=12)

        # Add legend
        ax.legend(loc='upper right', fontsize=10)

    # Adjust layout for better spacing
    plt.tight_layout(pad=3.0)
    plt.show()

# Function displaying a scatter plot of client datasets
def scatter_plot_grid(args, client_counts):

    # Extract protected group/label counts: female, male, low_income, high_income
    counts = extract_counts(client_counts)
    labels = ["Female", "Male", "Low Income", "High Income"]

    # Define an index mapping dictionary
    index_mapping = {
        0: (0, 1),
        1: (0, 2),
        2: (0, 3),
        3: (1, 2),
        4: (1, 3),
        5: (2, 3)
    }

    # Define the grid with subplots and the title
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    fig.suptitle(f"Scatter Plots ({args.partition}~Dir(alpha = {args.alpha}))", fontsize=16)

    # Flatten the axes array
    axes = axes.flatten()

    # Loop through each subplot
    for i, ax in enumerate(axes):
        x_idx, y_idx = index_mapping[i]
        x_data, y_data = counts[x_idx], counts[y_idx]
        x_label, y_label = labels[x_idx], labels[y_idx]

        # Scatter plot on each subplot
        ax.scatter(x_data, y_data, color="blue", marker="o")

        # Set title and labels
        ax.set_xlabel(f"Number Of {x_label} Samples", fontsize=12)
        ax.set_ylabel(f"Number Of {y_label} Samples", fontsize=12)

        # Show grid
        ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

# Function to plot the relationship between non-IIDness and metrics
def plot_non_iid_metrics(alpha_dict, metrics, baseline_metrics):

  hd_values = [alpha_dict[alpha_value] for alpha_value in alpha_dict.keys()]
  ylabels = ['Loss', 'Accuracy', 'F1 Score', 'Average Odds', 'Equal Opportunity', 'Demographic Parity']
  colors = ['r', 'b', 'y', 'g', 'm', 'c']

  fig, axs = plt.subplots(2, 3, figsize=(20, 10))

  for idx, ax in enumerate(axs.flat):
    ax.set_xlabel("Hellinger Distance")
    ax.set_ylabel(ylabels[idx])
    ax.plot(hd_values, metrics[idx], color = colors[idx])

    ax.axhline(y=baseline_metrics[idx], color='k', linestyle='--')

    if ylabels[idx] != 'Loss':
      ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True)

  plt.tight_layout()
  plt.show()