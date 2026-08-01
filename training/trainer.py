"""
Trainer Module for EEGMotorImageryModel.

Orchestrates training epochs, validation epochs, AMP mixed precision, gradient clipping,
metric calculation, checkpoint saving/resuming, and experiment logging.
"""

import os
import time
import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.state import TrainState
from training.device import get_device
from training.checkpoint import CheckpointManager
from metrics import compute_classification_metrics
from loggers.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Model-agnostic Trainer for executing model training, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        exp_logger: Optional[ExperimentLogger] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or {}
        train_cfg = self.config.get("training", {})

        self.device = device or get_device(train_cfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = int(train_cfg.get("epochs", 100))
        self.gradient_clip = float(train_cfg.get("gradient_clip", 1.0))
        self.use_amp = bool(train_cfg.get("mixed_precision", True)) and (self.device.type == "cuda")

        # AMP GradScaler if CUDA available
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Checkpoint manager
        ckpt_cfg = self.config.get("checkpoint", {})
        save_dir = ckpt_cfg.get("save_dir", "outputs/checkpoints")
        save_best = bool(ckpt_cfg.get("save_best", True))
        monitor = str(ckpt_cfg.get("monitor", "val_accuracy"))
        mode = str(ckpt_cfg.get("mode", "max"))

        self.checkpoint_manager = CheckpointManager(
            save_dir=save_dir, save_best=save_best, monitor=monitor, mode=mode
        )

        # Logger
        self.exp_logger = exp_logger
        self.state = TrainState(config=self.config)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Execute a single training epoch across batches with per-stage timing breakdown."""
        self.model.train()
        total_loss = 0.0
        all_logits = []
        all_targets = []

        is_cuda = self.device.type == "cuda"

        # Timing accumulators (in milliseconds)
        data_times = []
        h2d_times = []
        fwd_times = []
        loss_times = []
        bwd_times = []
        opt_times = []

        total_batches = len(dataloader)
        batch_size = dataloader.batch_size or 32

        t_data_start = time.perf_counter()

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            if is_cuda:
                torch.cuda.synchronize()
            t_data_end = time.perf_counter()
            data_ms = (t_data_end - t_data_start) * 1000.0
            data_times.append(data_ms)

            # 1. H2D Transfer
            t_h2d_start = time.perf_counter()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            if is_cuda:
                torch.cuda.synchronize()
            t_h2d_end = time.perf_counter()
            h2d_ms = (t_h2d_end - t_h2d_start) * 1000.0
            h2d_times.append(h2d_ms)

            self.optimizer.zero_grad()

            # 2. Forward Pass & 3. Loss Computation & 4. Backward Pass & 5. Optimizer Step
            if self.scaler:
                t_fwd_start = time.perf_counter()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(x_batch)
                if is_cuda:
                    torch.cuda.synchronize()
                t_fwd_end = time.perf_counter()
                fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0

                t_loss_start = time.perf_counter()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss = self.criterion(logits, y_batch)
                if is_cuda:
                    torch.cuda.synchronize()
                t_loss_end = time.perf_counter()
                loss_ms = (t_loss_end - t_loss_start) * 1000.0

                t_bwd_start = time.perf_counter()
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                if is_cuda:
                    torch.cuda.synchronize()
                t_bwd_end = time.perf_counter()
                bwd_ms = (t_bwd_end - t_bwd_start) * 1000.0

                t_opt_start = time.perf_counter()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if is_cuda:
                    torch.cuda.synchronize()
                t_opt_end = time.perf_counter()
                opt_ms = (t_opt_end - t_opt_start) * 1000.0
            else:
                t_fwd_start = time.perf_counter()
                logits = self.model(x_batch)
                if is_cuda:
                    torch.cuda.synchronize()
                t_fwd_end = time.perf_counter()
                fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0

                t_loss_start = time.perf_counter()
                loss = self.criterion(logits, y_batch)
                if is_cuda:
                    torch.cuda.synchronize()
                t_loss_end = time.perf_counter()
                loss_ms = (t_loss_end - t_loss_start) * 1000.0

                t_bwd_start = time.perf_counter()
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                if is_cuda:
                    torch.cuda.synchronize()
                t_bwd_end = time.perf_counter()
                bwd_ms = (t_bwd_end - t_bwd_start) * 1000.0

                t_opt_start = time.perf_counter()
                self.optimizer.step()
                if is_cuda:
                    torch.cuda.synchronize()
                t_opt_end = time.perf_counter()
                opt_ms = (t_opt_end - t_opt_start) * 1000.0

            fwd_times.append(fwd_ms)
            loss_times.append(loss_ms)
            bwd_times.append(bwd_ms)
            opt_times.append(opt_ms)

            total_loss += loss.item() * x_batch.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y_batch.detach().cpu())
            self.state.global_step += 1

            # Log timing every 100 batches
            curr_batch = batch_idx + 1
            if curr_batch % 100 == 0 or curr_batch == total_batches:
                recent_slice = slice(-min(100, curr_batch), None)
                avg_data = sum(data_times[recent_slice]) / len(data_times[recent_slice])
                avg_h2d = sum(h2d_times[recent_slice]) / len(h2d_times[recent_slice])
                avg_fwd = sum(fwd_times[recent_slice]) / len(fwd_times[recent_slice])
                avg_loss_t = sum(loss_times[recent_slice]) / len(loss_times[recent_slice])
                avg_bwd = sum(bwd_times[recent_slice]) / len(bwd_times[recent_slice])
                avg_opt = sum(opt_times[recent_slice]) / len(opt_times[recent_slice])
                avg_total = avg_data + avg_h2d + avg_fwd + avg_loss_t + avg_bwd + avg_opt
                throughput = (batch_size / (avg_total / 1000.0)) if avg_total > 0 else 0.0

                gpu_alloc = (torch.cuda.memory_allocated() / (1024**2)) if is_cuda else 0.0
                gpu_res = (torch.cuda.memory_reserved() / (1024**2)) if is_cuda else 0.0

                log_msg = (
                    f"Batch [{curr_batch:4d}/{total_batches:4d}] Per-Batch Timing (Last 100 Avg):\n"
                    f"  Data Loading : {avg_data:6.2f} ms ({avg_data/avg_total*100:4.1f}%)\n"
                    f"  H2D Transfer : {avg_h2d:6.2f} ms ({avg_h2d/avg_total*100:4.1f}%)\n"
                    f"  Forward Pass : {avg_fwd:6.2f} ms ({avg_fwd/avg_total*100:4.1f}%)\n"
                    f"  Loss Compute : {avg_loss_t:6.2f} ms ({avg_loss_t/avg_total*100:4.1f}%)\n"
                    f"  Backward Pass: {avg_bwd:6.2f} ms ({avg_bwd/avg_total*100:4.1f}%)\n"
                    f"  Optimizer    : {avg_opt:6.2f} ms ({avg_opt/avg_total*100:4.1f}%)\n"
                    f"  Total Batch  : {avg_total:6.2f} ms | Throughput: {throughput:6.1f} samples/sec\n"
                    f"  GPU Allocated: {gpu_alloc:6.1f} MB | GPU Reserved: {gpu_res:6.1f} MB"
                )
                logger.info(log_msg)

            t_data_start = time.perf_counter()

        avg_loss = total_loss / len(dataloader.dataset)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        metrics = compute_classification_metrics(cat_logits, cat_targets)
        metrics["loss"] = avg_loss

        timing_stats = {
            "data_ms": sum(data_times) / len(data_times),
            "h2d_ms": sum(h2d_times) / len(h2d_times),
            "fwd_ms": sum(fwd_times) / len(fwd_times),
            "loss_ms": sum(loss_times) / len(loss_times),
            "bwd_ms": sum(bwd_times) / len(bwd_times),
            "opt_ms": sum(opt_times) / len(opt_times),
            "total_train_sec": (sum(data_times) + sum(h2d_times) + sum(fwd_times) + sum(loss_times) + sum(bwd_times) + sum(opt_times)) / 1000.0,
        }

        return metrics, timing_stats

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[Dict[str, float], float]:
        """Execute validation epoch across batches with timing measurement."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_targets = []

        is_cuda = self.device.type == "cuda"
        t_val_start = time.perf_counter()

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * x_batch.size(0)
                all_logits.append(logits.cpu())
                all_targets.append(y_batch.cpu())

        if is_cuda:
            torch.cuda.synchronize()
        val_sec = time.perf_counter() - t_val_start

        avg_loss = total_loss / len(dataloader.dataset)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        metrics = compute_classification_metrics(cat_logits, cat_targets)
        metrics["loss"] = avg_loss
        return metrics, val_sec

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Optional[str] = None,
    ) -> TrainState:
        """
        Execute full training loop over epochs.

        Args:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            resume_path: Optional checkpoint path to resume training from

        Returns:
            Updated TrainState object
        """
        if resume_path and os.path.exists(resume_path):
            self.state = self.checkpoint_manager.load_checkpoint(
                checkpoint_path=resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.device,
            )
            logger.info(f"Resumed training from epoch {self.state.epoch}")

        start_epoch = self.state.epoch + 1
        for epoch in range(start_epoch, self.epochs + 1):
            self.state.epoch = epoch

            train_metrics, timing_stats = self.train_epoch(train_loader)
            val_metrics, val_sec = self.validate_epoch(val_loader)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("accuracy", 0.0))
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Merge epoch metrics for logging
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "learning_rate": current_lr,
            }

            if self.exp_logger:
                self.exp_logger.log_metrics(epoch_log, epoch=epoch)

            # Save checkpoints
            val_acc = val_metrics["accuracy"]
            self.checkpoint_manager.save_if_best(
                model=self.model,
                train_state=self.state,
                current_metric=val_acc,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )

            # Save latest checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                train_state=self.state,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                filename="latest.pt",
            )

            tot_b = len(train_loader)
            tot_train_s = timing_stats["total_train_sec"]
            avg_batch_ms = (tot_train_s * 1000.0 / tot_b) if tot_b > 0 else 0.0
            tot_samples = len(train_loader.dataset)
            throughput = (tot_samples / tot_train_s) if tot_train_s > 0 else 0.0
            val_throughput = (len(val_loader.dataset) / val_sec) if val_sec > 0 else 0.0

            gpu_alloc = (torch.cuda.memory_allocated() / (1024**2)) if (self.device.type == "cuda") else 0.0
            gpu_res = (torch.cuda.memory_reserved() / (1024**2)) if (self.device.type == "cuda") else 0.0

            summary_table = (
                f"\n================================================================================\n"
                f"Epoch {epoch:03d}/{self.epochs:03d} Timing Summary ({tot_b} Batches, {tot_samples:,} Samples)\n"
                f"--------------------------------------------------------------------------------\n"
                f"  Data Loading     : {timing_stats['data_ms']:6.2f} ms / batch (Total: {timing_stats['data_ms']*tot_b/1000.0:6.2f}s | {timing_stats['data_ms']/avg_batch_ms*100:4.1f}%)\n"
                f"  Host->Device H2D : {timing_stats['h2d_ms']:6.2f} ms / batch (Total: {timing_stats['h2d_ms']*tot_b/1000.0:6.2f}s | {timing_stats['h2d_ms']/avg_batch_ms*100:4.1f}%)\n"
                f"  Forward Pass     : {timing_stats['fwd_ms']:6.2f} ms / batch (Total: {timing_stats['fwd_ms']*tot_b/1000.0:6.2f}s | {timing_stats['fwd_ms']/avg_batch_ms*100:4.1f}%)\n"
                f"  Loss Compute     : {timing_stats['loss_ms']:6.2f} ms / batch (Total: {timing_stats['loss_ms']*tot_b/1000.0:6.2f}s | {timing_stats['loss_ms']/avg_batch_ms*100:4.1f}%)\n"
                f"  Backward Pass    : {timing_stats['bwd_ms']:6.2f} ms / batch (Total: {timing_stats['bwd_ms']*tot_b/1000.0:6.2f}s | {timing_stats['bwd_ms']/avg_batch_ms*100:4.1f}%)\n"
                f"  Optimizer Step   : {timing_stats['opt_ms']:6.2f} ms / batch (Total: {timing_stats['opt_ms']*tot_b/1000.0:6.2f}s | {timing_stats['opt_ms']/avg_batch_ms*100:4.1f}%)\n"
                f"--------------------------------------------------------------------------------\n"
                f"  Avg Batch Latency: {avg_batch_ms:6.2f} ms / batch\n"
                f"  Training Time    : {tot_train_s:6.2f}s (Throughput: {throughput:6.1f} samples/sec)\n"
                f"  Validation Time  : {val_sec:6.2f}s ({len(val_loader)} Batches, {val_throughput:6.1f} samples/sec)\n"
                f"  Total Epoch Time : {tot_train_s + val_sec:6.2f}s\n"
                f"  GPU Memory       : {gpu_alloc:6.1f} MB Allocated | {gpu_res:6.1f} MB Reserved\n"
                f"--------------------------------------------------------------------------------\n"
                f"Epoch {epoch:03d}/{self.epochs:03d} Metrics | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.6f}\n"
                f"================================================================================"
            )
            logger.info(summary_table)
            print(summary_table)

        return self.state
