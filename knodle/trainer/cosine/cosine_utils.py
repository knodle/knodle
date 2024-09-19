import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        #elf.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod /  torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1))
            dist_sq = dist ** 2
            #print(x0, x1, torch.sum(torch.pow(x0-x1, 2), 1) / x0.shape[-1], dist, dist_sq)
        else:
            print("Error Loss Metric!!")
            return 0
        #dist = torch.sum( - x0 * x1 / np.sqrt(x0.shape[-1]), 1).exp()
        #dist_sq = dist ** 2

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist