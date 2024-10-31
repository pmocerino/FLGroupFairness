import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import client_evaluate
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer


# ----------- FedAvg ----------- #

class FedAvgSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()

        data_size = 0
        for _ in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters, data_size]

# ----------- FairOrder ----------- #

class FairOrderSerialClientTrainer(FedAvgSerialClientTrainer):

  def local_process(self, payload, id_list):
    model_parameters = payload[0]
    for id in (progress_bar := tqdm(id_list)):
        progress_bar.set_description(f"Training on client {id}", refresh=False)
        data_loader = self.dataset.get_dataloader(id, self.batch_size, fair_order=True)
        pack = self.train(model_parameters, data_loader)
        self.cache.append(pack)

# ----------- FedFair ----------- #

class FedFairSerialClientTrainer(FedAvgSerialClientTrainer):

    def local_process(self, payload, id_list):
      model_parameters = payload[0]
      for id in (progress_bar := tqdm(id_list)):
          progress_bar.set_description(f"Training on client {id}", refresh=True)
          data_loader = self.dataset.get_dataloader(id, self.batch_size, fair_order=True)
          pack = self.train(model_parameters, data_loader)
          self.cache.append(pack)

    def evaluate(self, payload, id_list):
        model_parameters = payload[0]
        self.set_model(model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        losses, aos, sizes = [], [], []
        for id in id_list:
            dataset = self.dataset.get_dataset(id)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            loss, ao = client_evaluate(self._model, criterion, dataloader)
            losses.append(loss)
            aos.append(ao)
            sizes.append(len(dataset))
        return losses, aos, sizes

# ----------- FedFairUniform ----------- #

class FedFairUniformSerialClientTrainer(SGDSerialClientTrainer):

    def local_process(self, payload, id_list):
      model_parameters = payload[0]
      for id in (progress_bar := tqdm(id_list)):
          progress_bar.set_description(f"Training on client {id}", refresh=True)
          data_loader = self.dataset.get_dataloader(id, self.batch_size, fair_order=True)
          pack = self.train(model_parameters, data_loader)
          self.cache.append(pack)

    def evaluate(self, payload, id_list):
        model_parameters = payload[0]
        self.set_model(model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        losses, aos, sizes = [], [], []
        for id in id_list:
            dataset = self.dataset.get_dataset(id)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            loss, ao = client_evaluate(self._model, criterion, dataloader)
            losses.append(loss)
            aos.append(ao)
            sizes.append(len(dataset))
        return losses, aos, sizes