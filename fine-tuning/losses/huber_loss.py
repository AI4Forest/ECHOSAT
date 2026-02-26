import torch.nn as nn
import torch

class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss)
    """

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
        delta=1.0,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function
        self.huber_loss = nn.SmoothL1Loss(reduction='mean', beta=delta)

    def forward(self, out, target):
        """
        Applies the Huber loss
        :param out: output of the network
        :param target: target
        :return Huber loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        if target.dim() == 5:
            out = out.squeeze(1)
            B, Y, G, H, W = target.shape
            target = target.reshape(B*Y, G, H, W)
            out = out.reshape(B*Y, 1, H, W)

        out = out.flatten()
        target = target.flatten()

        if self.ignore_value is not None:
            mask = target != self.ignore_value
            out = out[mask]
            target = target[mask]

        loss = self.huber_loss(out, target)

        return loss
