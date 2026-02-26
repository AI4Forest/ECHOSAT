import torch.nn as nn
import torch


class DisturbanceRegressionLoss2Heads(nn.Module):
    """Mean Squared error"""

    def __init__(
        self,
        disturbance_indicator= 7,
        slope_min=0,
        slope_max=2,
        full_disturbance_window = False,
        use_l2 = True,
        max_intercept_after_disturbance = 100,
        disturbance_factor = 1,
        no_disturbance_factor = 1,
        slope_no_disturbance = None
    ):
        super().__init__()
        self.disturbance_indicator = disturbance_indicator
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.full_disturbance_window = full_disturbance_window
        self.use_l2 = use_l2
        self.max_intercept_after_disturbance = max_intercept_after_disturbance
        self.disturbance_factor = disturbance_factor
        self.no_disturbance_factor = no_disturbance_factor
        self.slope_no_disturbance = slope_no_disturbance
        
    def forward(self, out, target):
        """
        Applies the L2 loss
        :param out: output of the network, shape (B, 2, Y, H, W)
        :param target: target
        :return L2 loss
        """
        
        out1 = out[:,1,...]  # Shape: (B, Y, H, W)  
        fitted_values, disturbance_mask, slope_no_disturbance = self.get_regression(out, target)

        loss = (fitted_values - out1)**2 if self.use_l2 else torch.abs(fitted_values - out1)
        loss = loss.mean(dim=-3)
        if self.no_disturbance_factor != 1:
            loss[slope_no_disturbance < self.slope_no_disturbance] = loss[slope_no_disturbance < self.slope_no_disturbance] * self.no_disturbance_factor
        
        if self.disturbance_factor != 1:
            loss[disturbance_mask] = loss[disturbance_mask] * self.disturbance_factor
        # loss = (loss_after + loss_before) / Y
        return loss.mean()

    def get_disturbance_idx(self, out0):
        B, Y, H, W = out0.shape
        time_dim = -3
        with torch.no_grad():
            diff = out0.diff(axis=time_dim)
            if self.disturbance_indicator > 0:
                diff = torch.concat([-self.disturbance_indicator * torch.ones((B, 1, H, W), device=diff.device),
                                diff], axis=time_dim)
            else:  
                out0_short = out0[:,:-1]
                out0_shift = out0[:,1:]
                out0_shift2 = torch.concat([out0[:,2:], out0[:,-1:]], axis=time_dim) 
                mask_disturbance = (0.5*out0_short <= -diff) & (torch.min(torch.stack([out0_shift, out0_shift2], axis =0),axis=0).values <= 10) & (diff < -4) ## Shape [B,Y,H,W]
                mask_disturbance_surround = nn.functional.max_pool3d(input= mask_disturbance.to(torch.float32), kernel_size=(1,3,3), stride=1, padding=(0,1,1)) ## Shape [B,Y,H,W]
                mask_only_disturbance_surround = mask_disturbance_surround > mask_disturbance
                mask_disturbance = mask_disturbance_surround.clone()
                diff = -1*mask_disturbance.to(torch.int) ## -1 if disturbance, 0 else
                diff = torch.concat([torch.zeros((B, 1, H, W), device=diff.device), diff], axis=time_dim)
                diff_only_surround = -1*mask_only_disturbance_surround.to(torch.int)
                diff_only_surround = torch.concat([torch.zeros((B, 1, H, W), device=diff.device), diff_only_surround], axis=time_dim)
            if not self.full_disturbance_window:
                diff[:,1,:,:] = 0
                diff[:,-1,:,:] = 0
            disturbance_idx = torch.argmin(diff, dim=time_dim)
            if self.disturbance_indicator <= 0:
                disturbance_idx_surround = torch.argmin(diff_only_surround, dim=time_dim)
                disturbance_idx_surround[disturbance_idx > 0] = 0  # If disturbance detected, set surround to 0
                # disturbance_idx[disturbance_idx == 0] = disturbance_idx_surround[disturbance_idx == 0]
            else:
                disturbance_idx_surround = torch.zeros_like(disturbance_idx)
        return disturbance_idx, disturbance_idx_surround

    def get_regression(self, out, target):
        out0 = out[:,0,...]  # Shape: (B, Y, H, W)
        out1 = out[:,1,...]  # Shape: (B, Y, H, W)

        device = out.device    
        B, Y, H, W = out0.shape
        time_dim = -3
        fitted_values = torch.zeros_like(out0)
        slope_no_disturbance = torch.zeros((B, H, W), device=device)

        disturbance_idx, disturbance_idx_surround = self.get_disturbance_idx(out0)

        for f in range(out.shape[time_dim]):
            current_dist_index = torch.where(disturbance_idx == f)

            output_before0 = out0[:, :f, :, :]
            output_after0 = out0[:,f:, :, :]
            output_before1 = out1[:, :f, :, :]
            output_after1 = out1[:,f:, :, :]
            if f == 0:
                ## No disturbance detected
                minimal_intercept = torch.zeros((B, H, W), device=device)
                if self.disturbance_indicator <= 0:
                    minimal_intercept = torch.zeros((B, H, W), device=device)

                fitted_values_after_current, slope_after_current = self.get_fitted_values_local(output_after0, output_after1, min_intercept = None)
                fitted_values_before_current = torch.zeros((B,1,H,W), device=device)
                # loss_before_current = self.get_loss_per_pixel(output_before0, output_before1)
                # loss_after_current = self.get_loss_per_pixel(output_after0, output_after1, min_intercept = minimal_intercept)
            else:
                ## Disturbance detected
                fitted_values_before_current, _ = self.get_fitted_values_local(output_before0, output_before1)
                fitted_values_after_current, _ = self.get_fitted_values_local(output_after0, output_after1, max_intercept=self.max_intercept_after_disturbance)
                # loss_before_current = self.get_loss_per_pixel(output_before0, output_before1)
                # loss_after_current = self.get_loss_per_pixel(output_after0, output_after1, max_intercept=self.max_intercept_after_disturbance)    
            
            if f > 0:
                fitted_values[current_dist_index[0], :f, current_dist_index[1], current_dist_index[2]] = fitted_values_before_current[current_dist_index[0],:f,current_dist_index[1],current_dist_index[2]]
            fitted_values[current_dist_index[0], f:, current_dist_index[1], current_dist_index[2]] = fitted_values_after_current[current_dist_index[0], :, current_dist_index[1], current_dist_index[2]]
            slope_no_disturbance[current_dist_index[0], current_dist_index[1], current_dist_index[2]] = slope_after_current[current_dist_index[0], 0, current_dist_index[1], current_dist_index[2]]
            # loss_before[current_dist_index] = loss_before_current[current_dist_index]
            # loss_after[current_dist_index] = loss_after_current[current_dist_index]

        disturbance_mask = (disturbance_idx > 0)
        return fitted_values, disturbance_mask, slope_no_disturbance

    def get_fitted_values_local(self, out0, out1, max_intercept = 100, min_intercept = None):
        device = out0.device
        B, Y, H, W = out0.shape
        time_dim = -3
        if Y <= 1:
            return out0, torch.zeros((B,1,H,W), device=device)
        with torch.no_grad():
            x= torch.arange(Y, device=device, dtype = out0.dtype)
            mean_x = x.mean(dim=0, keepdim=True)
            mean_y = out0.mean(dim = time_dim, keepdim=True)

            cov_xy = ((x - mean_x).reshape(1,Y,1,1).expand(B, -1, H, W) * (out0 - mean_y)).sum(dim=time_dim, keepdim=True)
            var_x = ((x - mean_x) ** 2).sum(-1,keepdim=True)

            slope = cov_xy / var_x
            intercept = mean_y - slope * mean_x
            intercept = torch.clamp(intercept, min=0, max=max_intercept)
            if min_intercept is not None:
                intercept = torch.max(torch.concat([intercept, min_intercept.unsqueeze(1)], axis = 1), axis=1).values.unsqueeze(1)
                slope = (mean_y - intercept) / mean_x
                slope = slope
            year_index = torch.arange(Y, dtype=slope.dtype, device=slope.device).reshape(1, Y, 1, 1).expand(B, -1, H, W)
            fitted_values = torch.clamp(slope, min=self.slope_min, max=self.slope_max) * year_index + intercept
        return fitted_values, slope