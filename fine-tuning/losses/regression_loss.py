import torch.nn as nn
import torch


class RegressionLoss(nn.Module):
    """Mean Squared error"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
        ignore_non_label_year=True,
        full_disturbance_window=True,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function
        self.ignore_non_label_year = ignore_non_label_year
        self.full_disturbance_window = full_disturbance_window
        
    def forward(self, out, target):
        """
        Applies the L2 loss
        :param out: output of the network
        :param target: target
        :return L2 loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        B, Y, H, W = out.shape

        if self.full_disturbance_window:
            change = (out[:, 1:] - out[:,[0]].repeat(1,Y-1,1,1))
        else:
            change = (out[:, 2:-1] - out[:,[1]].repeat(1,Y-3,1,1))
        mask_no_disturbance = ((change) > -7).all(dim=-3).reshape(B, 1, H, W).expand(B, Y, H, W)

        with torch.no_grad():
            x = torch.arange(Y, dtype=out.dtype, device=out.device)
            mean_x = x.mean(dim=-1, keepdim=True)
            mean_y = out.mean(dim=-3, keepdim=True)

            cov_xy = ((x - mean_x).reshape(1,Y,1,1).expand(B, -1, H, W) * (out - mean_y)).sum(dim=-3, keepdim=True)
            var_x = ((x - mean_x) ** 2).sum(-1,keepdim=True)

            slope = nn.functional.relu(cov_xy / var_x)
            intercept = mean_y - slope * mean_x
            
            year_index = torch.arange(Y, dtype=slope.dtype, device=slope.device).reshape(1, Y, 1, 1).expand(B, -1, H, W)
            
            regression = slope * year_index + intercept
        
        residuals = (out - regression)**2
        residuals = residuals[mask_no_disturbance]

        return residuals.mean()
