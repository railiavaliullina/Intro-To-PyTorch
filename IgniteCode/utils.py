from torch.autograd import Function
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch


class ConstMultiplication(Function):
    @staticmethod
    def forward(ctx, input, const):
        ctx.constant = const
        return input*const

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*ctx.constant, None


class CustomLossValidation(Metric):

    def __init__(self, model, l2_norm_lambda, output_transform=lambda x: x):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.l2_norm_lambda = l2_norm_lambda
        self.loss = None
        self.cross_entropy_loss = None
        self.reg_loss = None
        self.num_batches = None
        super(CustomLossValidation, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.loss = 0
        self.cross_entropy_loss = 0
        self.reg_loss = 0
        self.num_batches = 0
        super(CustomLossValidation, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        cross_entropy_loss = self.criterion(y_pred, y)
        l2_reg = 0
        for p in self.model.parameters():
            l2_reg = l2_reg + self.l2_norm_lambda * p.norm(2)
        loss = cross_entropy_loss + l2_reg

        self.loss += loss
        self.cross_entropy_loss += cross_entropy_loss
        self.reg_loss += l2_reg
        self.num_batches += 1

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self.num_batches == 0:
            raise NotComputableError('LossValidation must have at least one batch before it can be computed.')
        return self.loss.item()/self.num_batches, self.cross_entropy_loss.item()/self.num_batches, \
               self.reg_loss.item()/self.num_batches
