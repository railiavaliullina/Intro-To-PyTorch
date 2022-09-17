import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# MLP с 1 скрытым слоем
class MLP1HL(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dim=128):
        super(MLP1HL, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_layer_dim)
        self.out = torch.nn.Linear(hidden_layer_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.out(self.relu(self.hidden_layer(x)))


# MLP с 2 скрытыми слоями
class MLP2HL(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer1_dim=64, hidden_layer2_dim=64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer1_dim = hidden_layer1_dim
        self.hidden_layer2_dim = hidden_layer2_dim
        super(MLP2HL, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(self.input_dim, self.hidden_layer1_dim)
        self.hidden_layer2 = torch.nn.Linear(self.hidden_layer1_dim, self.hidden_layer2_dim)
        self.out = torch.nn.Linear(self.hidden_layer2_dim, self.output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.out(self.relu(self.hidden_layer2(self.relu(self.hidden_layer1(x)))))
