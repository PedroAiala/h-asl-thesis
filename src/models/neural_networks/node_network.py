import torch
import torch.nn as nn

class NodeNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_features = 512
        self.hidden_features_1 = 128  # Primeira camada escondida (como no paper)
        self.hidden_features_2 = 128  # Segunda camada escondida (como no paper)
        self.out_features = 2
        self.dropout_p = 0.2  # Renomeei para evitar conflito com self.dropout

        # Arquitetura 512 -> 128 -> 128 -> 2
        self.layer1 = nn.Linear(self.in_features, self.hidden_features_1)
        self.layer2 = nn.Linear(self.hidden_features_1, self.hidden_features_2)
        self.layer3 = nn.Linear(self.hidden_features_2, self.out_features)

        # O paper confirma que Tanh é a melhor ativação [cite: 270]
        self.activation = nn.Tanh() 
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        (Arquitetura 512 -> 128 -> 128 -> 2)
        """
        # Bloco 1
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Bloco 2 (Nova camada)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Saída (Logits)
        logits = self.layer3(x)

        return logits