import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate
from fedlab.core.standalone import StandalonePipeline


# ----------- FedAvg ----------- #

class EvalPipeline(StandalonePipeline):
  def __init__(self, handler, trainer, test_loader):
      super().__init__(handler, trainer)
      self.test_loader = test_loader
      self.loss = []
      self.acc = []
      self.f1 = []
      self.ao = []
      self.eod = []
      self.dp = []

  def main(self):
      t = 0
      while self.handler.if_stop is False:
          # server side
          sampled_clients = self.handler.sample_clients()
          broadcast = self.handler.downlink_package

          # client side
          self.trainer.local_process(broadcast, sampled_clients)
          uploads = self.trainer.uplink_package

          # server side
          for pack in uploads:
              self.handler.load(pack)
          criterion = torch.nn.CrossEntropyLoss()
          loss, acc, f1, ao, eod, dp = evaluate(self.handler.model, criterion, self.test_loader)
          msg = "Round {}, Loss {:.4f}, Test Accuracy {:.4f}, F1 Score {:.4f}, Average Odds {:.4f}, Equal Opportunity {:.4f}, Demographic Parity {:.4f}".format(t, loss, acc, f1, ao, eod, dp)
          print(msg)
          t += 1

          self.loss.append(loss)
          self.acc.append(acc)
          self.f1.append(f1)
          self.ao.append(ao)
          self.eod.append(eod)
          self.dp.append(dp)

  def show(self):
      fig, axs = plt.subplots(2, 3, figsize=(20, 10))

      metrics = {
        'Loss': self.loss,
        'Accuracy': self.acc,
        'F1 Score': self.f1,
        'Average Odds': self.ao,
        'Equal Opportunity': self.eod,
        'Demographic Parity': self.dp
      }

      for ax, (label, values) in zip(axs.flat, list(metrics.items())):
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(label)
        ax.plot(np.arange(len(values)), values)

      plt.tight_layout()
      plt.show()

# ----------- FedFair ----------- #

class FedFairPipeline(EvalPipeline):

  def main(self):
      t = 0

      while self.handler.if_stop is False:

          # server side
          all_clients = np.array(range(self.handler.num_clients))
          broadcast = self.handler.downlink_package

          # client side
          losses, aos, sizes = self.trainer.evaluate(broadcast, all_clients)

          # server side
          candidates = self.handler.sample_candidates(sizes)
          values = self.handler.compute_values(losses, aos, sizes, candidates)
          sampled_clients = self.handler.sample_clients(candidates, values)
          broadcast = self.handler.downlink_package

          # client side
          self.trainer.local_process(broadcast, sampled_clients)
          uploads = self.trainer.uplink_package

          # server side
          for pack in uploads:
              self.handler.load(pack)
          criterion = torch.nn.CrossEntropyLoss()
          loss, acc, f1, ao, eod, dp = evaluate(self.handler.model, criterion, self.test_loader)
          msg = "Round {}, Loss {:.4f}, Test Accuracy {:.4f}, F1 Score {:.4f}, Average Odds {:.4f}, Equal Opportunity {:.4f}, Demographic Parity {:.4f}".format(t, loss, acc, f1, ao, eod, dp)
          print(msg)
          t += 1

          self.loss.append(loss)
          self.acc.append(acc)
          self.f1.append(f1)
          self.ao.append(ao)
          self.eod.append(eod)
          self.dp.append(dp)