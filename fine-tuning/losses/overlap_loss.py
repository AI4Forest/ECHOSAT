from .l1_loss import L1Loss
import torch.nn as nn
import torch


class OverlapLoss(nn.Module):
    """Mean Absolute error"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
        overlap_lambda=1.0,
        overlap_size=40
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function
        
        self.overlap_lambda = overlap_lambda
        self.l1_loss = L1Loss(ignore_value=ignore_value)
        self.overlap_size = overlap_size
        
    def forward(self, out, target):
        """
        Applies the Overlap loss
        :param out: output of the network
        :param target: target
        :return: overlap loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        left_images = out[::2]
        right_images = out[1::2]
        
        left_cut = left_images[..., -self.overlap_size:,:]
        right_cut = right_images[..., :self.overlap_size,:]

        overlap_loss = nn.functional.mse_loss(left_cut, right_cut, reduction='mean')
        
        loss = overlap_loss * self.overlap_lambda + self.l1_loss(out, target)

        return loss.mean()
