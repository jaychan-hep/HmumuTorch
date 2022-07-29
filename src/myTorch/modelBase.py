#!/usr/bin/env python3
import sys
import torch
from torch import nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from utils import metric
from . import loss

class hmumuModel(LightningModule):
    def __init__(self, loss_fn=None, lr=0.0001, eps=1e-07, device="cpu"):
        super(hmumuModel, self).__init__()
        self.dvc = device
        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = loss.EventWeightedBCE()
        self.lr = lr
        self.eps = eps
        self.epoch = 0

    def training_step(self, batch, batch_idx):
        X, y, w = batch
        if type(X) == list:
            X = [x.to(self.dvc) for x in X]
        y, w = y.to(self.dvc), w.to(self.dvc)
        logits = self(X)
        ls = self.loss_fn(logits, y.unsqueeze(1).double(), w.unsqueeze(1).double())
        return {"loss": ls, "score": logits.detach(), "target": y, "weight": w}

    def validation_step(self, batch, batch_idx):
        X, y, w = batch
        if type(X) == list:
            X = [x.to(self.dvc) for x in X]
        y, w = y.to(self.dvc), w.to(self.dvc)
        logits = self(X)
        ls = self.loss_fn(logits, y.unsqueeze(1).double(), w.unsqueeze(1).double())
        return {"loss": ls, "score": logits.detach(), "target": y, "weight": w}

    def test_step(self, batch, batch_idx):
        X, y, w = batch
        if type(X) == list:
            X = [x.to(self.dvc) for x in X]
        y, w = y.to(self.dvc), w.to(self.dvc)
        logits = self(X)
        ls = self.loss_fn(logits, y.unsqueeze(1).double(), w.unsqueeze(1).double())
        return {"loss": ls, "score": logits.detach(), "target": y, "weight": w}

    def training_epoch_end(self, training_step_outputs):
        self.training_loss = 0
        scores, ys, ws = None, None, None
        for training_step_output in training_step_outputs:
            self.training_loss += torch.mul(training_step_output["loss"], training_step_output["weight"].sum()).item()
            scores = torch.cat([scores, training_step_output["score"]]) if scores is not None else training_step_output["score"]
            ys = torch.cat([ys, training_step_output["target"]]) if ys is not None else training_step_output["target"]
            ws = torch.cat([ws, training_step_output["weight"]]) if ws is not None else training_step_output["weight"]

        self.training_loss /= ws.sum().item()

        scores = scores.cpu().numpy().reshape(-1)
        ys = ys.cpu().numpy()
        ws = ws.cpu().numpy()
        self.training_roc_auc = metric.roc_auc(scores, ys, ws)

        self.log("train_loss", self.training_loss)
        self.log("train_auc", self.training_roc_auc)

        sys.stdout.write('\033[2K\033[1G')
        sys.stdout.write(f'Epoch {self.epoch}: Train loss - {self.training_loss}, Train AUC - {self.training_roc_auc}, Val loss - {self.validation_loss}, Val AUC - {self.validation_roc_auc} \n')
        self.epoch += 1

    def validation_epoch_end(self, validation_step_outputs):
        self.validation_loss = 0
        scores, ys, ws = None, None, None
        for validation_step_output in validation_step_outputs:
            # self.validation_loss += torch.mul(validation_step_output["loss"], validation_step_output["weight"].sum()).item()
            scores = torch.cat([scores, validation_step_output["score"]]) if scores is not None else validation_step_output["score"]
            ys = torch.cat([ys, validation_step_output["target"]]) if ys is not None else validation_step_output["target"]
            ws = torch.cat([ws, validation_step_output["weight"]]) if ws is not None else validation_step_output["weight"]

        # self.validation_loss /= ws.sum().item()
        self.validation_loss = self.loss_fn(scores, ys.unsqueeze(1).double(), ws.unsqueeze(1).double())

        scores = scores.cpu().numpy().reshape(-1)
        ys = ys.cpu().numpy()
        ws = ws.cpu().numpy()
        self.validation_roc_auc = metric.roc_auc(scores, ys, ws)

        self.log("val_loss", self.validation_loss)
        self.log("val_auc", self.validation_roc_auc)

    def test_epoch_end(self, test_step_outputs):
        self.test_loss = 0
        scores, ys, ws = None, None, None
        for test_step_output in test_step_outputs:
            # self.test_loss += torch.mul(test_step_output["loss"], test_step_output["weight"].sum()).item()
            scores = torch.cat([scores, test_step_output["score"]]) if scores is not None else test_step_output["score"]
            ys = torch.cat([ys, test_step_output["target"]]) if ys is not None else test_step_output["target"]
            ws = torch.cat([ws, test_step_output["weight"]]) if ws is not None else test_step_output["weight"]

        # self.test_loss /= ws.sum().item()
        self.test_loss = self.loss_fn(scores, ys.unsqueeze(1).double(), ws.unsqueeze(1).double())

        self.test_scores = scores.cpu().numpy().reshape(-1)
        self.test_ys = ys.cpu().numpy()
        self.test_ws = ws.cpu().numpy()
        self.test_roc_auc = metric.roc_auc(self.test_scores, self.test_ys, self.test_ws)

        self.log("test_loss", self.test_loss)
        self.log("test_auc", self.test_roc_auc)

        # sys.stdout.write('Finished. \n')
        # sys.stdout.write(f'Test loss - {self.test_loss}, Test AUC - {self.test_roc_auc} \n')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, eps=self.eps)