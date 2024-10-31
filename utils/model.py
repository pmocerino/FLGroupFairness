import torch
import torch.nn as nn

# Global variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class MLP(nn.Module):
  def __init__(self):
      super(MLP, self).__init__()

      self.fc1 = nn.Linear(in_features=10, out_features=64)
      self.fc2 = nn.Linear(in_features=64, out_features=2)

      self.relu = nn.ReLU()

  def forward(self, x):

     x = self.relu(self.fc1(x))
     x = self.fc2(x)

     return x

def get_model(init_seed=42, ckpt=None):

  # Set weight initialization seed
  torch.manual_seed(init_seed)

  if ckpt is not None:
    # Load model from checkpoint
    model = torch.load(ckpt, map_location=device, weights_only=False)
  else:
    model = MLP()

  return model