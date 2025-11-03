"""
Multi-GPU Training Script using PyTorch DistributedDataParallel (DDP)

Optimized for 8x NVIDIA B200 GPUs.
Uses all available GPUs for 8x faster training.

Usage:
    torchrun --nproc_per_node=8 train_multigpu.py [args]
"""

import argparse
import time
import os
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from dataset import GraphColoringDataset, collate_batch
from model import MultiTaskGATv2, MultiTaskLoss


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    return local_rank, dist.get_world_size(), dist.get_rank()


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_epoch(model, loader, optimizer, loss_fn, device, scaler, epoch, local_rank, world_size):
    """Train for one epoch with DDP"""
    model.train()
    total_losses = {k: 0 for k in ['total', 'color', 'chromatic', 'graph_type', 'difficulty']}
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        targets = {
            'y_colors': batch.y_colors,
            'y_chromatic': batch.y_chromatic,
            'y_graph_type': batch.y_graph_type,
            'y_difficulty': batch.y_difficulty,
        }

        optimizer.zero_grad()

        with autocast():
            predictions = model(batch)
            loss, losses = loss_fn(predictions, targets)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses (only on rank 0 for logging)
        if local_rank == 0:
            for k, v in losses.items():
                total_losses[k] += v

            if (batch_idx + 1) % 50 == 0:
                print(f"  [GPU {local_rank}] Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {losses['total']:.4f} "
                      f"(color: {losses['color']:.4f}, "
                      f"chromatic: {losses['chromatic']:.4f}, "
                      f"type: {losses['graph_type']:.4f}, "
                      f"diff: {losses['difficulty']:.4f})")

    # Average losses
    if local_rank == 0:
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    return None


@torch.no_grad()
def validate(model, loader, loss_fn, device, local_rank):
    """Validate model"""
    model.eval()
    total_losses = {k: 0 for k in ['total', 'color', 'chromatic', 'graph_type', 'difficulty']}
    num_batches = len(loader)

    chromatic_mae = 0
    color_accuracy = 0
    graph_type_accuracy = 0

    for batch in loader:
        batch = batch.to(device)

        targets = {
            'y_colors': batch.y_colors,
            'y_chromatic': batch.y_chromatic,
            'y_graph_type': batch.y_graph_type,
            'y_difficulty': batch.y_difficulty,
        }

        with autocast():
            predictions = model(batch)
            loss, losses = loss_fn(predictions, targets)

        for k, v in losses.items():
            total_losses[k] += v

        chromatic_mae += torch.abs(predictions['chromatic'] - targets['y_chromatic']).mean().item()
        color_preds = predictions['color_logits'].argmax(dim=1)
        color_accuracy += (color_preds == targets['y_colors']).float().mean().item()
        type_preds = predictions['graph_type_logits'].argmax(dim=1)
        graph_type_accuracy += (type_preds == targets['y_graph_type'].squeeze()).float().mean().item()

    # Gather metrics across all GPUs
    for k in total_losses:
        total_losses[k] = torch.tensor(total_losses[k]).cuda()
        dist.all_reduce(total_losses[k], op=dist.ReduceOp.SUM)
        total_losses[k] = total_losses[k].item() / dist.get_world_size()

    chromatic_mae_tensor = torch.tensor(chromatic_mae).cuda()
    color_acc_tensor = torch.tensor(color_accuracy).cuda()
    type_acc_tensor = torch.tensor(graph_type_accuracy).cuda()

    dist.all_reduce(chromatic_mae_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(color_acc_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(type_acc_tensor, op=dist.ReduceOp.SUM)

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    metrics = {
        'chromatic_mae': chromatic_mae_tensor.item() / (num_batches * dist.get_world_size()),
        'color_accuracy': color_acc_tensor.item() / (num_batches * dist.get_world_size()),
        'graph_type_accuracy': type_acc_tensor.item() / (num_batches * dist.get_world_size()),
    }

    return avg_losses, metrics


def main(args):
    # Setup distributed training
    local_rank, world_size, global_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    if global_rank == 0:
        print("="*80)
        print("Multi-GPU GNN Training for Graph Coloring")
        print("="*80)
        print(f"\nDistributed Training Setup:")
        print(f"  World size: {world_size} GPUs")
        print(f"  Device: {torch.cuda.get_device_name(local_rank)}")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory * world_size / 1e9:.0f} GB")

    # Datasets
    train_dataset = GraphColoringDataset(args.data_dir, split='train', max_colors=args.max_colors)
    val_dataset = GraphColoringDataset(args.data_dir, split='val', max_colors=args.max_colors)

    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    if global_rank == 0:
        print(f"\nDatasets:")
        print(f"  Train: {len(train_dataset)} graphs ({len(train_dataset)//world_size} per GPU)")
        print(f"  Val:   {len(val_dataset)} graphs")

    # Model
    model = MultiTaskGATv2(
        node_feature_dim=16,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_colors=args.max_colors,
        num_graph_types=8,
        dropout=args.dropout,
    ).to(device)

    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if global_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Effective batch size: {args.batch_size * world_size}")

    # Loss, optimizer, scheduler
    loss_fn = MultiTaskLoss(
        color_weight=0.5,
        chromatic_weight=0.25,
        graph_type_weight=0.15,
        difficulty_weight=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * world_size,  # Scale learning rate
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=(global_rank == 0),
    )

    scaler = GradScaler()

    # TensorBoard (only on rank 0)
    writer = None
    if global_rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"\nStarting training for {args.epochs} epochs...")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
        print(f"  Learning rate: {args.lr * world_size}")
        print("="*80)

    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Set epoch for sampler (ensures different shuffle each epoch)
        train_sampler.set_epoch(epoch)

        if global_rank == 0:
            print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler, epoch, local_rank, world_size)

        if global_rank == 0 and train_losses:
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(color: {train_losses['color']:.4f}, "
                  f"chromatic: {train_losses['chromatic']:.4f}, "
                  f"type: {train_losses['graph_type']:.4f}, "
                  f"diff: {train_losses['difficulty']:.4f})")

        # Validate
        val_losses, val_metrics = validate(model, val_loader, loss_fn, device, local_rank)

        if global_rank == 0:
            print(f"  Val Loss:   {val_losses['total']:.4f} "
                  f"(color: {val_losses['color']:.4f}, "
                  f"chromatic: {val_losses['chromatic']:.4f}, "
                  f"type: {val_losses['graph_type']:.4f}, "
                  f"diff: {val_losses['difficulty']:.4f})")
            print(f"  Val Metrics: "
                  f"Chromatic MAE: {val_metrics['chromatic_mae']:.2f}, "
                  f"Color Acc: {val_metrics['color_accuracy']:.3f}, "
                  f"Type Acc: {val_metrics['graph_type_accuracy']:.3f}")

            # Learning rate scheduling
            scheduler.step(val_losses['total'])

            # TensorBoard logging
            if train_losses:
                for k, v in train_losses.items():
                    writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_losses.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val_metrics/{k}', v, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            # Checkpointing (only on rank 0)
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # .module for DDP
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }
                torch.save(checkpoint, args.checkpoint_dir / 'best_model.pt')
                print(f"  ✅ New best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")

            # Early stopping
            if patience_counter >= args.early_stop_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break

            epoch_time = time.time() - epoch_start
            print(f"  Epoch time: {epoch_time:.1f}s")

    # Cleanup
    if global_rank == 0:
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print(f"✅ Training complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  GPUs used: {world_size}")
        print(f"  Best model saved to: {args.checkpoint_dir / 'best_model.pt'}")
        print("="*80)

        writer.close()

        # Save metadata
        total_params = sum(p.numel() for p in model.parameters())
        metadata = {
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_time_minutes': total_time / 60,
            'gpus_used': world_size,
            'final_lr': optimizer.param_groups[0]['lr'],
            'model_params': total_params,
        }
        with open(args.checkpoint_dir / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-dir', type=str, default='../../training_data')
    parser.add_argument('--max-colors', type=int, default=200)

    # Model
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size PER GPU (effective = batch_size * num_gpus)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--early-stop-patience', type=int, default=15)

    # System
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')

    args = parser.parse_args()

    # Create directories
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    Path(args.log_dir).mkdir(exist_ok=True, parents=True)

    main(args)
