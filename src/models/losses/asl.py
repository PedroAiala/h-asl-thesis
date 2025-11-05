import random
import torch
import time


class AxialSpheresLoss(torch.nn.Module):
    def __init__(self, num_classes:int=2, alpha:float=10., beta:float=.1, gamma:float=.01, reduction:str='mean'):
        super(AxialSpheresLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.reduction = reduction

        self.pos_anchors = torch.eye(self.num_classes) * self.alpha

    @classmethod
    def logit2distance(cls, logits, centres, p=2):
        centres = centres.to(logits.device)
        distances = torch.norm(logits.unsqueeze(1) - centres, p=p, dim=2)
        return distances

    @classmethod
    def rejectScores(cls, distances):
        return distances * (1 - torch.nn.functional.softmin(distances, dim=1))

    @classmethod
    def acceptScores(cls, logits, centres, p=2):
        centres = centres.to(logits.device)
        distances = torch.norm(logits.unsqueeze(dim=1) - centres, p=p, dim=2)
        rejections = distances * (1 - torch.nn.functional.softmin(distances, dim=1))
        acceptance = torch.max(rejections) - rejections
        return acceptance * torch.norm(logits, p=p, dim=1).view(-1, 1)

    def forward(self, logits, targets):
        # get boolean target tensor (true/false)
        pos_indexes = (targets >= 0)
        neg_indexes = (targets  < 0)
        # initialize intra and inter-distance tensors
        intra_distances = torch.zeros_like(targets[pos_indexes]).float()
        inter_distances = torch.zeros_like(targets[pos_indexes]).float()
        # compute logit distance to positive anchor
        cross_distances = self.logit2distance(logits[pos_indexes], centres=self.pos_anchors)
        # store intra-class distance (to be minimized)
        intra_distances = cross_distances.gather(dim=1, index=targets[pos_indexes].view(-1, 1).long())
        # store inter-class distance (to be maximized)
        inter_distances = torch.exp(intra_distances - cross_distances).sum(dim=1).log()
        # compute feature magnitude root
        outer_magnitude = logits[neg_indexes].norm(p=2, dim=1) if sum(neg_indexes) else torch.tensor([0.]).to(logits.device)
        # return batch loss
        if   self.reduction == 'mean':
            return self.beta * intra_distances.mean() + inter_distances.mean() + self.gamma * outer_magnitude.mean()
        elif self.reduction ==  'sum':
            return self.beta * intra_distances.sum() + inter_distances.sum() + self.gamma * outer_magnitude.sum()
        else:
            return (intra_distances, inter_distances, outer_magnitude)


torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

num_classes, num_samples, batch_size = 20, 50, 10

values = torch.randn(num_samples, num_classes, requires_grad=True).to(device)
labels = torch.randint(num_classes * 2, (num_samples,), dtype=torch.int64).to(device) - num_classes
print(device, values.shape, labels.shape)

b_values = values[:batch_size]
b_labels = labels[:batch_size]
print(b_labels)


criterion = AxialSpheresLoss(num_classes=num_classes, alpha=10., beta=.1, reduction='mean')
loss_score = criterion(b_values, b_labels)
loss_score.backward()
print('ASL:', loss_score)