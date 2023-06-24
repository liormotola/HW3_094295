from torch_geometric.nn import GCNConv, GATConv
import torch
import torch.nn.functional as F
from torch.nn import Linear
from dataset import HW3Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

class GAT(torch.nn.Module):
    def __init__(self, num_features,num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features,96,heads=4)
        self.conv2 = GATConv(96*4,128,heads=1)
        self.conv3 = GATConv(128, 64, heads=1)
        self.classifier = Linear(64, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        out = self.classifier(x)
        return out

def train(model, data, optimizer, criterion, num_epochs: int, model_name:str):
    """
    Trains the given model with the given optimizer and loss function for num_epochs epochs.
    Saves the best model to pickle file called model_name.
    :param model: Model to train
    :param data: train and validation data. Data object.
    :param optimizer: optimizer to use during training.
    :param criterion: loss function to use during training.
    :param num_epochs: number of training epochs.
    :param model_name: name to save the model.
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
                correct = pred[data.train_mask] == data.y[data.train_mask]
                acc = int(correct.sum()) / len(data.train_mask)
                train_loss.append(loss.detach().cpu())
                train_acc.append(acc)
            else:
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    loss = criterion(out[data.val_mask], data.y[data.val_mask])
                    pred = out.argmax(dim=1)
                    correct = pred[data.val_mask] == data.y[data.val_mask]
                    acc = int(correct.sum()) / len(data.val_mask)
                    val_loss.append(loss.detach().cpu())
                    val_acc.append(acc)

            epoch_acc = round(acc, 3)
            print(f'{phase.title()} Loss: {loss:.4e} Accuracy: {epoch_acc}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                with open(model_name, 'wb') as f:
                    torch.save(model, f)
        print()

    print(f'Best Validation Accuracy: {best_acc:4f}')
    return train_loss,train_acc,val_loss,val_acc


def plot_graphs(train,validation,title,model_name):
    """
    generates graphs of train and validation data vs num epochs
    :param train: list of train data - could be loss/accuracy vals
    :param validation: list of validation data - could be loss/accuracy vals
    :param title: plot's title
    :param model_name: name of model whose values are presented. will be added to the the title
    """
    plt.plot(train,label="train")
    plt.plot(validation,label= "val")
    plt.xlabel("num epochs")
    plt.ylabel(title)
    plt.title(f"{model_name}: {title}")
    plt.legend()
    plt.show()


def basic(model,model_name):
    """
    trains the model without publish year as feature + plots loss and accuracy graphs
    :param model: model to train
    :param model_name: name to save the model
    """
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    data.y = data.y.squeeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss, train_acc, val_loss, val_acc=train(model=model,
                                                  data=data,
                                                  optimizer=optimizer,
                                                  criterion=criterion,
                                                  num_epochs=500,model_name=model_name)
    plot_graphs(train_loss,val_loss, title=f"Loss", model_name=model_name.strip('.pkl'))
    plot_graphs(train_acc,val_acc, title=f"Accuracy", model_name=model_name.strip('.pkl'))


def years(model,model_name):
    """
    trains the model with publish year as feature + plots loss and accuracy graphs
    :param model: model to train
    :param model_name: name to save the model
    """
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    data.y = data.y.squeeze(1)
    scaler = StandardScaler()
    arr_norm = scaler.fit_transform(data.node_year.numpy())
    years = torch.from_numpy(arr_norm).type(dtype=torch.float32)
    data.x = torch.concat((data.x, years), dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss, train_acc, val_loss, val_acc = train(model=model,
                                                     data=data,
                                                     optimizer=optimizer,
                                                     criterion=criterion,
                                                     num_epochs=500, model_name=model_name)
    plot_graphs(train_loss, val_loss, title=f"Loss", model_name=model_name.strip('.pkl'))
    plot_graphs(train_acc, val_acc, title=f"Accuracy", model_name=model_name.strip('.pkl'))


if __name__ == '__main__':
    torch.manual_seed(42)
    dataset = HW3Dataset(root='data/hw3/')
    #exp 1
    model_1 = GCN(num_features=dataset.num_features, num_classes=dataset.num_classes)
    basic(model=model_1,model_name= "GCN.pkl")
    #exp 2 - GCN + years
    model_2 = GCN(num_features=dataset.num_features + 1, num_classes=dataset.num_classes)
    years(model=model_2,model_name= "GCN_years.pkl")
    #exp 3 - GAT
    model_3 = GAT(num_features=dataset.num_features, num_classes=dataset.num_classes)
    basic(model=model_3,model_name="GAT.pkl")
    #exp 4 - GAT + years
    model_4 = GAT(num_features=dataset.num_features+1, num_classes=dataset.num_classes)
    years(model=model_4, model_name="GAT_years.pkl")

