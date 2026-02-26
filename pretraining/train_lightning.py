#!/usr/bin/env python3
"""
Minimal PyTorch Lightning training script for forest height prediction.
Usage: torchrun --nproc_per_node=N train_lightning.py
"""

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from lightning_module import ForestHeightLightningModule
from lightning_dataset import create_dataloaders
from lightning_visualization_callback import MultiYearEvolutionCallback


def main():
    
    # Set NCCL environment variables for better stability
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes timeout
    os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Enable blocking wait
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Better error handling
    os.environ['NCCL_DEBUG'] = 'INFO'  # Enable debug logging
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if causing issues
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P if causing issues
    os.environ['NCCL_SHM_DISABLE'] = '1'  # Disable shared memory if causing issues
    os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '1000000'  # Enable trace buffer
    
    torch.set_float32_matmul_precision("high")
    
    parser = argparse.ArgumentParser(description='Train forest height model with PyTorch Lightning')
    parser.add_argument('--data_path', type=str, default="../dataset", 
                       help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers per GPU')
    parser.add_argument('--max_epochs', type=int, default=3,
                       help='Maximum number of epochs (-1 for infinite)')
    parser.add_argument('--max_steps', type=int, default=400000,
                       help='Maximum number of epochs (-1 for infinite)')
    parser.add_argument('--val_check_interval', type=float, default=0.5, 
                       help='Validation check interval (fraction of epoch)')
    parser.add_argument('--log_every_n_steps', type=int, default=4,
                       help='Log every N steps')
    parser.add_argument('--save_top_k', type=int, default=3,
                       help='Number of best models to save')
    parser.add_argument('--experiment_name', type=str, default='ECHOSAT',
                       help='Experiment name')
    parser.add_argument('--viz_max_batches', type=int, default=20,
                       help='Maximum number of validation batches to visualize')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Create model
    model = ForestHeightLightningModule()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup logging
    wandb_logger = WandbLogger(
        project="ECHOSAT",
        name=args.experiment_name,
        log_model=True
    )
    
    # Get wandb run ID
    wandb_run_id = wandb_logger.experiment.id if wandb_logger.experiment else "unknown"
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/huber',
        mode='min',
        save_top_k=args.save_top_k,
        dirpath='checkpoints',
        filename=f'forest-height-{wandb_run_id}-{{epoch:02d}}',
        save_last=True,
        every_n_epochs=1,
        save_on_exception=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup visualization callback
    viz_callback = MultiYearEvolutionCallback(
        max_batches=args.viz_max_batches
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, viz_callback],
        precision="32", 
        accelerator='gpu', #change to GPU if available
        devices=[0,1],  # Use all available GPUs
        strategy='ddp_find_unused_parameters_true',
        deterministic=False,  # Disable for better performance
        benchmark=True,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=1,
        sync_batchnorm=True,  # Sync batch norm across GPUs
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Print configuration
    print(f"Training configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Number of workers per GPU: {args.num_workers}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Validation check interval: {args.val_check_interval}")
    print(f"  Log every N steps: {args.log_every_n_steps}")
    print(f"  Visualization batches: {args.viz_max_batches}")
    print(f"  Strategy: DDP")
    print(f"  Devices: {trainer.num_devices}")
    
    # Start training
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    print("Training completed!")
    
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
