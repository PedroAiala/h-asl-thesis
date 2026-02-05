import torch
import torch.nn as nn

class NodeNetwork(nn.Module):
    """
    The Adapter Network (NodeNetwork) for H-ASL.
    
    It maps face embeddings (typically 512-dim) to a 2D logit space 
    representing the decision boundary (Left vs Right) of a specific node.
    
    Architecture:
      Input (N) -> Linear(256) -> Tanh -> Dropout -> Linear(256) -> Tanh -> Dropout -> Linear(2)
    """
    def __init__(self, input_dim=512):
        super().__init__()
        
        # Arquitetura baseada no paper ASL, com ajuste de largura para 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),          # Tanh é crucial para estabilidade da ASL
            nn.Dropout(0.2),    # Regularização
            
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 2)   # Saída 2D para cálculo da perda angular
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)