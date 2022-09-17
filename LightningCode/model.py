import torch
import torch.nn as nn

from LightningCode.config import cfg

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy


class LitLogisticRegression(pl.LightningModule):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        out = self.linear(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).to(device=cfg.device)
        y_pred = self.linear(x)
        cross_entropy_loss = self.loss(y_pred, y)
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for p in self.parameters():
            l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
        loss = cross_entropy_loss + l2_reg
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_cross_entropy_loss', cross_entropy_loss, on_step=True, on_epoch=True)
        self.log('train_reg_loss', l2_reg, on_step=True, on_epoch=True)
        self.log('train_acc_step', self.train_accuracy(y_pred, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).to(device=cfg.device)
        y_pred = self.linear(x)
        cross_entropy_loss = self.loss(y_pred, y)
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for p in self.parameters():
            l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
        loss = cross_entropy_loss + l2_reg
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_cross_entropy_loss', cross_entropy_loss, on_step=True, on_epoch=True)
        self.log('val_reg_loss', l2_reg, on_step=True, on_epoch=True)
        self.log('val_acc_step', self.val_accuracy(y_pred, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train/accuracy_epoch', self.train_accuracy.compute(), prog_bar=True, logger=True)

    def validation_epoch_end(self, outs):
        self.log('val/accuracy_epoch', self.val_accuracy.compute(), prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=cfg.lr)
        return optimizer


# MLP с 1 скрытым слоем
class LitMLP1HL(pl.LightningModule):

    def __init__(self, input_dim, output_dim, hidden_layer_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_layer_dim),
                                    torch.nn.Linear(hidden_layer_dim, output_dim),
                                    nn.ReLU())
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        out = self.mlp(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).to(device=cfg.device)
        y_pred = self.mlp(x)
        cross_entropy_loss = self.loss(y_pred, y)
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for p in self.parameters():
            l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
        loss = cross_entropy_loss + l2_reg
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_cross_entropy_loss', cross_entropy_loss, on_step=True, on_epoch=True)
        self.log('train_reg_loss', l2_reg, on_step=True, on_epoch=True)
        self.log('train_acc_step', self.train_accuracy(y_pred, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).to(device=cfg.device)
        y_pred = self.mlp(x)
        cross_entropy_loss = self.loss(y_pred, y)
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for p in self.parameters():
            l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
        loss = cross_entropy_loss + l2_reg
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_cross_entropy_loss', cross_entropy_loss, on_step=True, on_epoch=True)
        self.log('val_reg_loss', l2_reg, on_step=True, on_epoch=True)
        self.log('val_acc_step', self.val_accuracy(y_pred, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train/accuracy_epoch', self.train_accuracy.compute(), prog_bar=True, logger=True)

    def validation_epoch_end(self, outs):
        self.log('val/accuracy_epoch', self.val_accuracy.compute(), prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=cfg.lr)
        return optimizer


# MLP с 2 скрытыми слоями
class LitMLP2HL(pl.LightningModule):

    def __init__(self, input_dim, output_dim, hidden_layer1_dim=64, hidden_layer2_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_layer1_dim),
                                 nn.ReLU(),
                                 torch.nn.Linear(hidden_layer1_dim, hidden_layer2_dim),
                                 nn.ReLU(),
                                 torch.nn.Linear(hidden_layer2_dim, output_dim))
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        out = self.mlp(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).to(device=cfg.device)
        y_pred = self.mlp(x)
        cross_entropy_loss = self.loss(y_pred, y)
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for p in self.parameters():
            l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
        loss = cross_entropy_loss + l2_reg
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_cross_entropy_loss', cross_entropy_loss, on_step=True, on_epoch=True)
        self.log('train_reg_loss', l2_reg, on_step=True, on_epoch=True)
        self.log('train_acc_step', self.train_accuracy(y_pred, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).to(device=cfg.device)
        y_pred = self.mlp(x)
        cross_entropy_loss = self.loss(y_pred, y)
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for p in self.parameters():
            l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
        loss = cross_entropy_loss + l2_reg
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_cross_entropy_loss', cross_entropy_loss, on_step=True, on_epoch=True)
        self.log('val_reg_loss', l2_reg, on_step=True, on_epoch=True)
        self.log('val_acc_step', self.val_accuracy(y_pred, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train/accuracy_epoch', self.train_accuracy.compute(), prog_bar=True, logger=True)

    def validation_epoch_end(self, outs):
        self.log('val/accuracy_epoch', self.val_accuracy.compute(), prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=cfg.lr)
        return optimizer
