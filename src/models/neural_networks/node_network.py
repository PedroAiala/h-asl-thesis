import torch
import torch.nn as nn


class NodeNetwork(nn.Module):
    def __init__(self, in_features=512, hidden_features=128, out_features=2, dropout=0.2):
        super().__init__()

        self.layer1 = nn.Linear(in_features, hidden_features)
        self.layer2 = nn.Linear(hidden_features, out_features)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.layer2(x)

        return logits