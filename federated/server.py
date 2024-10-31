import torch
import numpy as np
from copy import deepcopy
from typing import List
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

# ----------- FedAvg ----------- #

class FedAvgServerHandler(SyncServerHandler):
    """FedAvg server handler."""
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        weights = [ele[1] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


# ----------- FedFair ----------- #

class FedFairServerHandler(FedAvgServerHandler):
    def __init__(self, beta, *args, **kwargs):
        super(FedFairServerHandler, self).__init__(*args, **kwargs)

        self.beta = beta
        self.selected_clients = []
        self.ao_devs = []

    def setup_optim(self, d):
        self.d = d

    def sample_candidates(self, sizes):
        probs = np.array(sizes) / np.array(sizes).sum()
        selected_clients = np.random.choice(self.num_clients,
                                            size=self.d,
                                            replace=False,
                                            p=probs)

        selection = sorted(selected_clients)
        return selection

    def sample_clients(self, candidates, values):
        sort = np.array(values).argsort().tolist()
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]
        selected_clients = sorted(selected_clients.tolist())
        self.selected_clients = selected_clients
        return selected_clients

    def compute_values(self, losses, aos, sizes, candidates):

        loss_min = np.min(losses)
        loss_max = np.max(losses)
        if loss_max > 1 and loss_max > loss_min:
            losses = np.array((losses - loss_min) / (loss_max - loss_min))

        weights = np.array(sizes) / np.sum(np.array(sizes))
        assert np.all(weights >= 0)

        global_ao = np.average(aos, weights=weights)
        ao_deltas = np.array([abs(ao - global_ao) for ao in aos])
        ao_delta_mean = np.average(ao_deltas)

        ao_devs = np.array([ao_delta - ao_delta_mean for ao_delta in ao_deltas])
        ao_devs_min = np.min(ao_devs)
        ao_devs_max = np.max(ao_devs)
        if ao_devs_max > ao_devs_min:
          ao_devs = (ao_devs - ao_devs_min) / (ao_devs_max - ao_devs_min)

        self.ao_devs = ao_devs

        values = np.array([self.beta * ao_devs[cid] + (1 - self.beta) * (1 - losses[cid]) for cid in candidates])

        return values

    def load(self, payload: List[torch.Tensor]) -> bool:

      assert len(payload) > 0
      self.client_buffer_cache.append(deepcopy(payload))

      assert len(self.client_buffer_cache) <= self.num_clients_per_round

      if len(self.client_buffer_cache) == self.num_clients_per_round:

          sizes_selected = np.array([size for _, size in self.client_buffer_cache])
          weights_selected = sizes_selected / sizes_selected.sum()
          assert np.all(weights_selected >= 0)

          adj_weights = weights_selected * (1 - self.beta * self.ao_devs[self.selected_clients])
          adj_weights = adj_weights / np.sum(adj_weights)
          assert np.all(adj_weights >= 0)

          self.client_buffer_cache = [[model_params, adj_weight] for (model_params, _), adj_weight in zip(self.client_buffer_cache, adj_weights)]
          self.global_update(self.client_buffer_cache)
          self.round += 1

          # reset cache
          self.client_buffer_cache = []

          return True  # return True to end this round.
      else:
          return False 

# ----------- FedFairUniform ----------- #

class FedFairUniformServerHandler(SyncServerHandler):
    def __init__(self, beta, *args, **kwargs):
        super(FedFairUniformServerHandler, self).__init__(*args, **kwargs)

        self.beta = beta
        self.selected_clients = []
        self.ao_devs = []

    def setup_optim(self, d):
        self.d = d

    def sample_candidates(self, sizes):
        probs = np.array(sizes) / np.array(sizes).sum()
        selected_clients = np.random.choice(self.num_clients,
                                            size=self.d,
                                            replace=False,
                                            p=probs)

        selection = sorted(selected_clients)
        return selection

    def sample_clients(self, candidates, values):
        sort = np.array(values).argsort().tolist()
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]
        selected_clients = sorted(selected_clients.tolist())
        self.selected_clients = selected_clients
        return selected_clients

    def compute_values(self, losses, aos, sizes, candidates):

        loss_min = np.min(losses)
        loss_max = np.max(losses)
        if loss_max > 1 and loss_max > loss_min:
            losses = np.array((losses - loss_min) / (loss_max - loss_min))

        weights = np.array(sizes) / np.sum(np.array(sizes))
        assert np.all(weights >= 0)

        global_ao = np.average(aos, weights=weights)
        ao_deltas = np.array([abs(ao - global_ao) for ao in aos])
        ao_delta_mean = np.average(ao_deltas)

        ao_devs = np.array([ao_delta - ao_delta_mean for ao_delta in ao_deltas])
        ao_devs_min = np.min(ao_devs)
        ao_devs_max = np.max(ao_devs)
        if ao_devs_max > ao_devs_min:
          ao_devs = (ao_devs - ao_devs_min) / (ao_devs_max - ao_devs_min)

        self.ao_devs = ao_devs

        values = np.array([self.beta * ao_devs[cid] + (1 - self.beta) * (1 - losses[cid]) for cid in candidates])

        return values

    def global_update(self, buffer, weights):
        parameters_list = [ele[0] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights=weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    def load(self, payload: List[torch.Tensor]) -> bool:

          assert len(payload) > 0
          self.client_buffer_cache.append(deepcopy(payload))

          assert len(self.client_buffer_cache) <= self.num_clients_per_round

          if len(self.client_buffer_cache) == self.num_clients_per_round:

              adj_weights = 1 - self.beta * self.ao_devs[self.selected_clients]
              adj_weights = adj_weights / np.sum(adj_weights)
              assert np.all(adj_weights >= 0)

              self.global_update(self.client_buffer_cache, weights=adj_weights)
              self.round += 1

              # reset cache
              self.client_buffer_cache = []

              return True  # return True to end this round.
          else:
              return False