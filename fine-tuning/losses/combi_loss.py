import torch.nn as nn
import torch
from .l1_loss import L1Loss
from .regression_loss import RegressionLoss
from .regression_loss_disturbance import DisturbanceRegressionLoss


class CombiLoss(nn.Module):
    """
    Combination of L1 and Regression Loss
    """

    def __init__(
        self,
        ignore_value=0,
        pre_calculation_function=None,
        lambda_regression=1.0,
        disturbance_indicator=7,
        slope_min=0,
        slope_max=2,
        full_disturbance_window=True
    ):
        super().__init__()
        self.pre_calculation_function = pre_calculation_function
        
        # Initialize the individual loss components
        self.l1_loss = L1Loss(ignore_value=ignore_value, pre_calculation_function=pre_calculation_function)
        

        self.regression_loss = DisturbanceRegressionLoss(disturbance_indicator=disturbance_indicator, 
                                                                     slope_min=slope_min, 
                                                                     slope_max=slope_max,
                                                                     full_disturbance_window=full_disturbance_window,
                                                                     precalculation_function=pre_calculation_function)

        # Store lambda parameters
        self.lambda_regression = lambda_regression


    def forward(self, out, target):
        """
        Applies the combined loss
        :param out: output of the network
        :param target: target
        :return: Combined loss
        """
        
        # out.shape = torch.Size([2, 7, 1, 256, 256])
        # target.shape = torch.Size([2, 7, 6, 256, 256])
        
        # Calculate individual loss components using the new loss classes
        l1_loss = self.l1_loss(out, target)

        regression_loss = self.regression_loss(out, target)
        
        # Combine losses with lambda weights
        loss = l1_loss + self.lambda_regression * regression_loss
            
        return loss
        