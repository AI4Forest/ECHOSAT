import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch.optim.lr_scheduler import OneCycleLR
import wandb

from models.swin_video_unet import SwinVideoUnet
from losses.huber_loss import HuberLoss


class ForestHeightLightningModule(pl.LightningModule):
    """PyTorch Lightning module for forest height prediction with SwinVideoUnet."""
    
    def __init__(self):
        super().__init__()
        
        self.model = SwinVideoUnet(
            input_shape=(7, 12, 18, 96, 96),  # 7 years, 6 months, 12 channels, 64x64 spatial
            embed_dim=24,
            encoder_depths=(2, 2, 2, 2),
            decoder_depths=(2,2,2,2),
            num_heads=(4, 8, 12, 24),
            window_size_temporal=2,
            window_size_spatial=6,
            reduce_time=(28, 14, 7),
            patch_size_time=1,
            patch_size_image=1,
            temporal_skip_reduction="transformer_year",
            use_final_convs=False,
            downsample_per_year=True,
        )
        precalculation_function = lambda out, target: (out, torch.sum(target, dim=-3))
        
        self.huber_loss = HuberLoss(ignore_value=0, pre_calculation_function=precalculation_function)
        
        # Metrics
        self.train_huber = MeanMetric()
        self.val_huber = MeanMetric()
        
        # Training configuration
        self.learning_rate = 0.0001
        self.weight_decay = 1e-5
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        outputs = self(x)
        
        # Calculate loss
        huber_loss = self.huber_loss(outputs, y)
        
        # Update metrics
        self.train_huber(huber_loss)
        
        # Log metrics
        self.log('train/huber', huber_loss, on_step=True, on_epoch=True)
        
        return huber_loss
    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        
        # Forward pass
        outputs = self(x)
        
        # Calculate losses
        huber_loss = self.huber_loss(outputs, y)
        
        # Update metrics
        self.val_huber(huber_loss)
        
        # Log metrics
        self.log('val/huber', huber_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return huber_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # OneCycleLR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos',
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
        
