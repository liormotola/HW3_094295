from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch.nn import Linear
from dataset import HW3Dataset
import matplotlib.pyplot as plt


class GCN(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()

        self.conv1 = GCNConv(num_features, 112)
        self.conv2 = GCNConv(112, 96)
        self.conv3 = GCNConv(96, 64)
        self.classifier = Linear(64, num_classes)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        out = self.classifier(x)

        return out



def train(model, data, optimizer, criterion, num_epochs: int):
    """
    Trains the given model with the given optimizer and loss function for num_epochs epochs.
    Saves the best model to pickle file called "model2.pkl".
    :param model: Model to train
    :param data_sets: Dict where keys are dataset type and values are the corresponding dataset object.
    :param optimizer: optimizer to use during training.
    :param criterion: loss function to use during training.
    :param num_epochs: number of training epochs.
    :param batch_size: batch size
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data.to(device)

    best_acc = 0.0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            optimizer.zero_grad()
            if phase == 'train':
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                pred = out.argmax(dim=1)
                correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
                acc = int(correct.sum()) / len(data.train_mask)
                train_loss.append(loss.detach().cpu())
                train_acc.append(acc)
            else:
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    loss = criterion(out[data.val_mask], data.y[data.val_mask])
                    pred = out.argmax(dim=1)
                    correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
                    acc = int(correct.sum()) / len(data.val_mask)
                    val_loss.append(loss.detach().cpu())
                    val_acc.append(acc)

            epoch_acc = round(acc, 3)
            print(f'{phase.title()} Loss: {loss:.4e} Accuracy: {epoch_acc}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                with open('model.pkl', 'wb') as f:
                    torch.save(model, f)
        print()

    print(f'Best Validation Accuracy: {best_acc:4f}')
    return train_loss,train_acc,val_loss,val_acc


def plot_graphs(train,validation,title):
    plt.plot(train,label="train")
    plt.plot(validation,label= "val")
    plt.xlabel("num epochs")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    data.y = data.y.squeeze(1)
    model = GCN(num_features=dataset.num_features, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss, train_acc, val_loss, val_acc=train(model=model,
                                                  data=data,
                                                  optimizer=optimizer,
                                                  criterion=criterion,
                                                  num_epochs=350)
    plot_graphs(train_loss,val_loss,title="Loss")
    plot_graphs(train_acc,val_acc,title="Accuracy")
