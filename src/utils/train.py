import os
import pickle
from sklearn.preprocessing import StandardScaler, QuantileTransformer

def transform_score(scores, save=False, output_path=""):
    tsf = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', subsample=1000000000, random_state=0)
    tsf.fit(scores.reshape(-1, 1))

    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(tsf, f, -1)
    return tsf

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y, w) in enumerate(dataloader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        # Compute prediction and loss
        pred = model(X)
        ls = loss_fn(pred, y.unsqueeze(1).double(), w.unsqueeze(1).double())

        # Backpropagation
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        if batch % 1000 == 0:
            ls, current = ls.item(), batch * len(X)
            print(f"loss: {ls:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, return_output=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    scores = []
    ys = []
    ws = []
    with torch.no_grad():
        for X, y, w in dataloader:
            X, y, w = X.to(device), y.to(device), w.to(device)
            pred = model(X)
            scores.append(pred)
            ys.append(y)
            ws.append(w)
            test_loss += (loss_fn(pred, y.unsqueeze(1).double(), w.unsqueeze(1).double())).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    scores = torch.cat(scores).cpu().numpy().reshape(-1)
    ys = torch.cat(ys).cpu().numpy()
    ws = torch.cat(ws).cpu().numpy()
    roc_auc = metric.roc_auc(scores, ys, ws)

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, AUC: {roc_auc} \n")

    if return_output:
        return scores, ys, ws