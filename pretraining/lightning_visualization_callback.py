import torch
import pytorch_lightning as pl
import wandb
from typing import Any, Dict, List, Optional
import numpy as np

import visualization


class MultiYearEvolutionCallback(pl.Callback):
    """
    Custom Lightning callback to generate multi-year evolution visualizations
    for the first 10 validation batches at the end of each validation epoch.
    """
    
    def __init__(self, max_batches: int = 10):
        """
        Initialize the callback.
        
        Args:
            max_batches: Maximum number of validation batches to visualize
        """
        super().__init__()
        self.max_batches = max_batches
        self.validation_data = []
        
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[torch.Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Store validation data for the first max_batches batches."""
        if batch_idx < self.max_batches:
            x, y = batch
            with torch.no_grad():
                # Get model predictions
                predictions = pl_module(x)
                
                # Store data for visualization
                
                self.validation_data.append({
                    'inputs': x.detach().cpu(),
                    'labels': y.detach().cpu(), 
                    'outputs': predictions.detach().cpu(),
                    'batch_idx': batch_idx
                })
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate and log multi-year evolution visualizations."""
                    
        if not self.validation_data:
            return
            
        # Process each stored batch
        for data in self.validation_data:
            try:
                # Create visualization using the same processing as runner.py
                viz_fn = self._get_visualization_function()
                fig = viz_fn(
                    inputs=data['inputs'],
                    labels=data['labels'], 
                    outputs=data['outputs']
                )
                
                # Log to wandb
                if trainer.logger and hasattr(trainer.logger, 'experiment'):
                    trainer.logger.experiment.log({
                        f'val/multi_year_evolution_batch_{data["batch_idx"]}': wandb.Image(fig)
                    })
                
                # Close figure to free memory
                import matplotlib.pyplot as plt
                plt.close(fig)
                
            except Exception as e:
                print(f"Visualization failed for batch {data['batch_idx']}: {e}")
                continue
        
        # Clear stored data to save memory
        self.validation_data = []
    
    def _get_visualization_function(self):
        """Get the multi-year evolution visualization function with proper data processing."""
        
        # Use a lambda to process variables for visualization
        process_variables = lambda inputs, labels, outputs: (
            (inputs + 1) / 2 if outputs.ndim > 1 else inputs,
            labels.sum(axis=-3) if outputs.ndim > 1 else labels,
            outputs
        )
        return visualization.get_multi_year_evolution_visualization(
            process_variables=process_variables,
            ignore_value=0,
            rgb_channels=[3, 2, 1]
        )
