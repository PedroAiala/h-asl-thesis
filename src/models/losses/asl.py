import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialSpheresLoss(nn.Module):
    """
    Axial Sphere Loss (ASL) Implementation for H-ASL.
    
    Combines the stability of the Paper's Eq. 4 (Log-Sum-Exp) with the 
    effective training strategy from the experiments (Magnitude penalty only on Unknowns).
    """
    def __init__(self, num_classes:int=2, alpha:float=10.0, beta:float=0.1, gamma:float=0.1, reduction:str='mean'):
        super(AxialSpheresLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma 
        self.reduction = reduction

        # Buffers: Garante que os centros (âncoras) sejam salvos com o modelo e movidos p/ GPU
        # Cria matriz identidade multiplicada por alpha: [[10, 0], [0, 10]]
        self.register_buffer('pos_anchors', torch.eye(self.num_classes) * self.alpha)

    @classmethod
    def logit2distance(cls, logits, centres, p=2):
        """Calcula distância Euclidiana (L2) entre logits e centros."""
        # logits: (B, Dim) -> (B, 1, Dim)
        # centres: (C, Dim)
        # Result: (B, C) - Distância de cada amostra para cada centro
        centres = centres.to(logits.device)
        distances = torch.norm(logits.unsqueeze(1) - centres, p=p, dim=2)
        return distances

    def forward(self, logits, targets):
        """
        logits: (Batch, 2) - Saída da rede (Feature Space)
        targets: (Batch,) - Labels (0 ou 1 para Known, -1 para Unknown/Background)
        """
        # Máscaras booleanas para separar fluxo de dados
        pos_indexes = (targets >= 0) # Known (Galeria)
        neg_indexes = (targets < 0)  # Unknown (Background)
        
        loss = torch.tensor(0.0, device=logits.device)
        
        # --- PARTE 1: CLASSIFICAÇÃO (Known / Galeria) ---
        if pos_indexes.sum() > 0:
            k_logits = logits[pos_indexes]
            k_targets = targets[pos_indexes]

            # 1. Distância de todas as amostras para todos os centros (0 e 1)
            # Shape: (N_known, 2)
            all_dists = self.logit2distance(k_logits, self.pos_anchors)
            
            # 2. Intra-Class Distance (Eq. 3 do Paper)
            # Pega apenas a distância para a classe CORRETA (Target)
            intra_distances = all_dists.gather(dim=1, index=k_targets.view(-1, 1).long())
            
            # 3. Inter-Class Loss (Eq. 4 do Paper - LogSumExp)
            # Maximiza a diferença entre (Distância Certa) e (Distância Errada)
            # Math: log(sum(exp(intra - all)))
            # Isso é equivalente a Softplus(intra - inter), muito mais estável que exp(intra - inter)
            diff = intra_distances - all_dists
            inter_loss = torch.log(torch.exp(diff).sum(dim=1))
            
            if self.reduction == 'mean':
                loss += self.beta * intra_distances.mean() + inter_loss.mean()
            else:
                loss += self.beta * intra_distances.sum() + inter_loss.sum()

        # --- PARTE 2: REJEIÇÃO (Unknown / Background) ---
        # Notebook Strategy: Aplicar magnitude apenas aqui.
        if neg_indexes.sum() > 0:
            uk_logits = logits[neg_indexes]
            
            # Força magnitude a ir para zero (Origem)
            outer_magnitude = torch.norm(uk_logits, p=2, dim=1)
            
            if self.reduction == 'mean':
                loss += self.gamma * outer_magnitude.mean()
            else:
                loss += self.gamma * outer_magnitude.sum()

        return loss