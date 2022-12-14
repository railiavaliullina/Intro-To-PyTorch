import torch
from config import cfg
from utils import ConstMultiplication


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)
        # использование custom class-a
        if cfg.use_const_multiplication:
            self.const_m = ConstMultiplication.apply
            self.const = cfg.const

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        out = self.linear(x)
        if cfg.use_const_multiplication:
            out = self.const_m(out, self.const)
        return out


# MLP с 1 скрытым слоем
class MLP1HL(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dim=128):
        super(MLP1HL, self).__init__()
        self.input_dim = input_dim
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_layer_dim)
        self.out = torch.nn.Linear(hidden_layer_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.out(self.relu(self.hidden_layer(x)))


# MLP с 2 скрытыми слоями
class MLP2HL(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer1_dim=64, hidden_layer2_dim=64):
        super(MLP2HL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer1_dim = hidden_layer1_dim
        self.hidden_layer2_dim = hidden_layer2_dim
        self.hidden_layer1 = torch.nn.Linear(self.input_dim, self.hidden_layer1_dim)
        self.hidden_layer2 = torch.nn.Linear(self.hidden_layer1_dim, self.hidden_layer2_dim)
        self.out = torch.nn.Linear(self.hidden_layer2_dim, self.output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.out(self.relu(self.hidden_layer2(self.relu(self.hidden_layer1(x)))))

