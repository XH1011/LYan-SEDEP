import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

class Encoder(nn.Module):
    def __init__(self, n_inp, n_hidden, n_out):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inp, n_hidden),
            nn.ReLU(),
        ) #Sequential(Linear(in_features=2048, out_features=1560, bias=True) ReLU())
        #Sequential(Linear(in_features=102, out_features=1450, bias=True) ReLU())

        self.mu_head = nn.Linear(n_hidden, n_out)
        self.logvar_head = nn.Linear(n_hidden, n_out)
        #Linear(in_features=1560, out_features=64, bias=True)
        #Linear(in_features=1450, out_features=64, bias=True)

        self.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        mu, log_var = self.mu_head(x), self.logvar_head(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, n_inp, n_hidden, n_out):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(n_inp, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )
        #Sequential(Linear(in_features=64, out_features=1660, bias=True) ReLU()
        # Linear(in_features=1660, out_features=2048, bias=True))
        #Sequential(Linear(in_features=64, out_features=660, bias=True) ReLU()
        #Linear(in_features=660, out_features=102, bias=True))
        self.apply(init_weights)

    def forward(self, x):
        return self.decoder(x)


class Classifier(nn.Module):
    def __init__(self, n_inp, n_out):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)
        # Classifier((fc1): Linear(in_features=64, out_features=645, bias=True))
        # Classifier((fc1): Linear(in_features=64, out_features=72, bias=True))

    def forward(self, x):
        return self.fc1(x)
