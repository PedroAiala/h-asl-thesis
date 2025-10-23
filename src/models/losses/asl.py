import torch
import torch.nn as nn

class AxialSphereLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.out_features = 2
        self.alpha = 10.0
        self.lambda_val = 0.10

        centers = torch.eye(self.out_features) * self.alpha
        self.register_buffer('centers', centers)


    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the Axial Sphere Loss.

        Args:
            logits (torch.Tensor): The output logits from the model.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss value.
        """
        
        self.centers = self.centers.to(logits.device)

        known_mask = (labels >= 0)
        unknown_mask = (labels < 0)

        logits_known = logits[known_mask]
        labels_known = labels[known_mask]

        logits_unknown = logits[unknown_mask]

        loss_known_total = 0.0
        loss_unknown_total = 0.0
        
        num_known = logits_known.size(0)
        num_unknown = logits_unknown.size(0)

        if num_unknown > 0:
            norm_l2_squared = torch.sum(logits_unknown.pow(2), dim=1)
            
            loss_unknown_total = norm_l2_squared.mean()

  
        if num_known > 0:
            logits_expanded = logits_known.unsqueeze(1)
            centers_expanded = self.centers.unsqueeze(0)

            dists_sq_matrix = (logits_expanded - centers_expanded).pow(2).sum(dim=2)
            d_correct_sq = dists_sq_matrix.gather(dim=1, index=labels_known.unsqueeze(1))

            loss_intra_per_sample = d_correct_sq.squeeze(1) 

            logits_norm_sq = torch.sum(logits_known.pow(2), dim=1) # Shape [N]
            alpha_sq = self.alpha.pow(2)
            
            diff_mag = alpha_sq - logits_norm_sq
            loss_mag_per_sample = torch.clamp(diff_mag, min=0.0)

            dists_diff = d_correct_sq - dists_sq_matrix
            
            exp_dists = torch.exp(dists_diff)
            sum_exp_dists = exp_dists.sum(dim=1)

            loss_inter_per_sample = torch.log(sum_exp_dists)

            lambda_term = self.lambda_val * (loss_intra_per_sample + loss_mag_per_sample)
            loss_known_per_sample = loss_inter_per_sample + lambda_term

            loss_known_total = loss_known_per_sample.mean()

        total_samples = num_known + num_unknown
        if total_samples == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)    

        total_loss = (loss_known_total * num_known + loss_unknown_total * num_unknown) / total_samples

        return total_loss
