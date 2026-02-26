import torch.nn as nn
import torch


class DisturbanceRegressionLoss(nn.Module):
    """Mean Squared error"""

    def __init__(
        self,
        disturbance_indicator= 7,
        slope_min=0,
        slope_max=2,
        full_disturbance_window = False,
        use_l2 = True,
        precalculation_function=None
    ):
        super().__init__()
        self.disturbance_indicator = disturbance_indicator
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.full_disturbance_window = full_disturbance_window
        self.use_l2 = use_l2
        self.precalculation_function = precalculation_function

    def forward(self, out, target):
        """
        Applies the L2 loss
        :param out: output of the network
        :param target: target
        :return L2 loss
        """

        if self.precalculation_function is not None:
            out, target = self.precalculation_function(out, target)
        device = out.device    
        B, Y, H, W = out.shape
        time_dim = -3
        loss_shape = (B, H, W)
        loss_before = torch.zeros(loss_shape, device=device)
        loss_after = torch.zeros_like(loss_before) 
        with torch.no_grad():
            diff = out.diff(axis=time_dim)
            diff = torch.concat([-self.disturbance_indicator * torch.ones((B, 1, H, W), device=diff.device),
                                diff], axis=time_dim)
            if not self.full_disturbance_window:
                diff[:,1,:,:] = 0
                diff[:,-1,:,:] = 0
            disturbance_idx = torch.argmin(diff, dim=time_dim)

        for f in range(out.shape[time_dim]):
            current_dist_index = torch.where(disturbance_idx == f)

            output_before = out[:, :f, :, :]
            output_after = out[:,f:, :, :]

           
            loss_before_current = self.get_loss_per_pixel(output_before)
            loss_after_current = self.get_loss_per_pixel(output_after)
            
            loss_before[current_dist_index] = loss_before_current[current_dist_index]
            loss_after[current_dist_index] = loss_after_current[current_dist_index]
        loss = (loss_after + loss_before) / Y
        return loss.mean()

    def get_loss_per_pixel(self, out):
        device = out.device    
        B, Y, H, W = out.shape
        time_dim = -3
        if Y <= 2:
            return torch.zeros((B, H, W), device=device)
        with torch.no_grad():
            x= torch.arange(Y, device=device, dtype = out.dtype)
            mean_x = x.mean(dim=0, keepdim=True)
            mean_y = out.mean(dim = time_dim, keepdim=True)

            cov_xy = ((x - mean_x).reshape(1,Y,1,1).expand(B, -1, H, W) * (out - mean_y)).sum(dim=time_dim, keepdim=True)
            var_x = ((x - mean_x) ** 2).sum(-1,keepdim=True)

            slope = torch.clamp(cov_xy / var_x, min=self.slope_min, max=self.slope_max)
            intercept = mean_y - slope * mean_x
            year_index = torch.arange(Y, dtype=slope.dtype, device=slope.device).reshape(1, Y, 1, 1).expand(B, -1, H, W)
            predictions = slope * year_index + intercept
        if self.use_l2:    
            residuals = (torch.abs(out - predictions)**2).sum(dim=time_dim)
        else:
            residuals = torch.abs(out - predictions).sum(dim=time_dim)    
        return residuals
