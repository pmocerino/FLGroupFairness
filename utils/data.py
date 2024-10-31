import torch 
import numpy as np
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset



def get_data(survey_year="2018", horizon="1-Year", states=["CA"]):

  # Get data downloaded from data source
  data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person')
  acs_data = data_source.get_data(states=states, download=True)
  X, y, _ = ACSIncome.df_to_numpy(acs_data)

  # Reorder the features to move the protected attribute at the front
  X = X[:, [8, *range(8), 9]]

  # Modify the protected attribute values according to tha advantaged/disadvantaged notation
  X[X[:, 0] == 2, 0] = 0

  # Train-Test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Define a data scaler
  datascaler = MinMaxScaler()
  datascaler.fit(X_train)
  X_train, X_test = datascaler.transform(X_train), datascaler.transform(X_test)

  return X_train, y_train, X_test, y_test


def get_extra_samples(X_train, y_train):

  # Initialize a dictionary to track whether each sample group has been added
  samples_needed = {(0, 0): False, (0, 1): False, (1, 0): False, (1, 1): False}
  X_extra, y_extra = [], []

  i = 0
  while not all(samples_needed.values()):
      X, y = X_train[i], y_train[i]
      pair = (X[0], y)

      # Add the sample if it hasn't been added yet
      if not samples_needed[pair]:
        X_extra.append(X)
        y_extra.append(y)
        samples_needed[pair] = True

      i += 1

  X_extra, y_extra = np.array(X_extra), np.array(y_extra)

  return X_extra, y_extra


class ClientDataset(Dataset):
    def __init__(self, data, targets, device):
        self.device = device

        self.data = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.targets = torch.tensor(targets, dtype=torch.long, device=self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
         return self.data[idx], self.targets[idx]