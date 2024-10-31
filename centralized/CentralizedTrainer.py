import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from ..utils.evaluation import AverageMeter, average_odds, equal_opportunity, demographic_parity

class CentralizedTrainer():
  """Centralized training handler.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool, optional): use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
    """
  def __init__(self,
               model:torch.nn.Module,
               cuda:bool=False,
               device:str=None):

    self.model = model
    self.device = device
    if self.device is not None:
      self.cuda = self.device.type == 'cuda'


  def train(self,
            train_loader,
            rr=True,
            epochs=300,
            lr=0.001):

    self.train_loader = train_loader
    self.rr = rr
    self.epochs = epochs
    self.lr = lr

    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)

    self.model.train()

    for epoch in tqdm(range(self.epochs)):

      # Seed fixing the shuffling at each epoch if RR is used
      if (self.rr):
        torch.manual_seed(epoch)

      for batch_idx, (data, target) in enumerate(train_loader):
        if self.cuda:
          data = data.cuda(self.device)
          target = target.cuda(self.device)

        # Zero the gradients for every batch
        self.optimizer.zero_grad()

        # Compute the outputs of the model
        output = self.model(data)

        # Compute the loss
        loss = self.criterion(output, target)

        # Compute loss gradient
        loss.backward()

        # Adjust learning weights
        self.optimizer.step()

    return self.model

  def evaluate(self,
            data_loader):

    self.criterion = nn.CrossEntropyLoss()

    self.model.eval()
    gpu = next(self.model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()

    labels_array = []
    predicted_array = []
    attr_array = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            batch_size = len(labels)
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

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
        eopp = equal_opportunity(labels_array, predicted_array, attr_array)
        dp = demographic_parity(predicted_array, attr_array)

    return loss_.avg, acc_.avg, f1, ao, eopp, dp