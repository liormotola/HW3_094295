import torch
from dataset import HW3Dataset
from train import GAT
import pandas as pd

def predict():

    #prepare data
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    data.y = data.y.squeeze(1)
    scaler = torch.load("scaler.pkl")
    arr_norm = scaler.fit_transform(data.node_year.numpy())
    years = torch.from_numpy(arr_norm).type(dtype=torch.float32)
    data.x = torch.concat((data.x, years), dim=1)

    #run model
    model = torch.load("Final_model.pkl")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data.to(device)

    #predict
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    pred_dict = {"idx": list(range(len(pred))), "prediction": pred.cpu().numpy()}
    df = pd.DataFrame(data=pred_dict)
    df.to_csv("prediction.csv", index=False)

if __name__ == '__main__':
    predict()