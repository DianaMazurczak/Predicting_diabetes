import pandas as pd
from sklearn.model_selection import train_test_split
import xlsxwriter

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam

from ucimlrepo import fetch_ucirepo


workbook = xlsxwriter.Workbook('Wyniki_sieci_neuronowe.xlsx')
worksheet = workbook.add_worksheet()


class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset
        self.features = dataset.iloc[:, :-1].values.astype(float)
        self.labels = dataset.iloc[:, -1].astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, :-1].values.astype(float)
        target = self.data.iloc[idx, -1].astype(float)

        sample = {'features': torch.tensor(features, dtype=torch.float32),
                  'target': torch.tensor(target, dtype=torch.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample


def data_preparation(id):
    # pobranie danych z uci repository
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features
    y = dataset.data.targets
    # podział na zbiór testowy i uczący się
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)
    # transforamcja na dane o formacie torch
    training_dataset = CustomDataset(pd.concat([X_train, y_train], axis=1))
    test_dataset = CustomDataset(pd.concat([X_test, y_test], axis=1))
    return training_dataset, test_dataset


class NeuralNetwork1layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class NeuralNetwork2layers(nn.Module):
    def __init__(self, features=16, activation=nn.ReLU()):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, features),
            activation,
            nn.Linear(features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class NeuralNetwork3layers(nn.Module):
    def __init__(self, features1=16, features2=16, activation=nn.ReLU()):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, features1),
            activation,
            nn.Linear(features1, features2),
            activation,
            nn.Linear(features2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class NeuralNetwork4layers(nn.Module):
    def __init__(self, features1=16, features2=16, features3=16, activation=nn.ReLU()):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, features1),
            activation,
            nn.Linear(features1, features2),
            activation,
            nn.Linear(features2, features3),
            activation,
            nn.Linear(features3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def train_loop(dataloader, model, loss_fn, optimizer, batch_size,  weight_for_zeros, weight_for_ones):
    size = len(dataloader.dataset)
    model.train()
    correct, TP, FP, FN = 0, 0, 0, 0

    for batch, sample in enumerate(dataloader):
        X, y = sample['features'], sample['target']

        sample_weights = torch.where(y == 0, weight_for_zeros, weight_for_ones)

        pred = model(X)
        y = y.unsqueeze(1)
        loss = loss_fn(pred, y) * sample_weights
        loss = loss.mean()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted = (pred > 0.3).int()
        correct += (predicted == y).sum().item()
        TP += ((predicted == 1) & (y == 1)).sum().item()
        FP += ((predicted == 1) & (y == 0)).sum().item()
        FN += ((predicted == 0) & (y == 1)).sum().item()

    acc = 100 * correct / size
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return acc, precision, recall


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    TP, FP, FN = 0, 0, 0

    with torch.no_grad():
        for sample in dataloader:
            X, y = sample['features'], sample['target']
            pred = model(X)
            y = y.unsqueeze(1)
            predicted = (pred > 0.3).int()

            # Compute metrics
            correct += (predicted == y).sum().item()
            TP += ((predicted == 1) & (y == 1)).sum().item()
            FP += ((predicted == 1) & (y == 0)).sum().item()
            FN += ((predicted == 0) & (y == 1)).sum().item()

    correct /= size
    acc = 100 * correct
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return acc, precision, recall


def main(id=891):
    nrow = 1
    epochs = 8
    # epoch - liczba powtórzeń, które algorytm uczący się wykona na całym zbiorze
    # batch size - mówi po ilu próbkach model powinien aktualizować parametry
    for batch_size in [32, 64, 128, 256]:
        for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
            for activ in [nn.Sigmoid(), nn.ReLU(), nn.Tanh(), nn.Softplus()]:

                training_dataset, test_dataset = data_preparation(id)
                train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

                model = NeuralNetwork3layers(features1=32, features2=32, activation=activ)

                loss_fn = nn.BCELoss()
                optimizer = Adam(model.parameters(), lr=learning_rate)

                print(f"batch_size = {batch_size} \n"
                      f"learning_rate = {learning_rate} \n"
                      f"activation function = {activ.__class__.__name__} \n")

                for t in range(epochs):
                    print(f"Epoch {t + 1}\n-------------------------------")
                    acc_train, pr_train, rc_train = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, 0.84, 0.16)
                    acc_test, pr_test, rc_test = test_loop(test_dataloader, model, loss_fn)
                    output_list =\
                        [t+1, batch_size, learning_rate, acc_train, pr_train, rc_train, acc_test, pr_test, rc_test]
                    print(output_list)
                    worksheet.write_row(row=nrow, col=0, data=output_list)
                    nrow += 1
    workbook.close()
    print("Done!")


if __name__ == '__main__':
    main(891)
