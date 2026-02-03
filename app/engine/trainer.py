"""
Training Engine for Neuro-Econometric Market Alpha Engine.

This module implements a custom training loop with:
- Mixed optimization (ARDL + Neural Network)
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Checkpointing
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import json

from app.config import Config
from app.models.fusion import NeuroEconometricNet, HybridTrainingWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridLoss(nn.Module):
    """
    Custom loss function for Neuro-Econometric Network.
    
    Mathematical Foundation:
        L_total = λ₁·L_mse + λ₂·L_directional + λ₃·L_ardl_agreement + λ₄·L_regularization
    
    Where:
        - L_mse: Mean Squared Error (price prediction accuracy)
        - L_directional: Directional accuracy loss (correct trend prediction)
        - L_ardl_agreement: Penalty for excessive deviation from ARDL
        - L_regularization: L2 penalty on gating weights
    
    Rationale:
        Pure MSE can lead to models that predict magnitude but wrong direction.
        Directional loss ensures profitable trading signals.
    """
    
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_directional: float = 0.5,
        lambda_agreement: float = 0.3,
        lambda_reg: float = 0.01
    ):
        """
        Initialize hybrid loss.
        
        Args:
            lambda_mse: Weight for MSE loss.
            lambda_directional: Weight for directional accuracy loss.
            lambda_agreement: Weight for ARDL agreement loss.
            lambda_reg: Weight for regularization.
        """
        super(HybridLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_directional = lambda_directional
        self.lambda_agreement = lambda_agreement
        self.lambda_reg = lambda_reg
        
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ardl_predictions: torch.Tensor,
        alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        
        Args:
            predictions: Fused predictions. Shape: (batch, 1)
            targets: Ground truth. Shape: (batch, 1)
            ardl_predictions: ARDL branch predictions. Shape: (batch, 1)
            alpha: Gating weights. Shape: (batch, 1)
        
        Returns:
            Tuple of (total_loss, loss_components).
        """
        # MSE Loss
        loss_mse = self.mse_loss(predictions, targets)
        
        # Directional Loss
        pred_direction = torch.sign(predictions - targets.mean())
        true_direction = torch.sign(targets - targets.mean())
        loss_directional = 1 - (pred_direction == true_direction).float().mean()
        
        # ARDL Agreement Loss (penalize large deviations)
        loss_agreement = torch.mean((predictions - ardl_predictions) ** 2)
        
        # Regularization (encourage moderate gating)
        # Penalize α too close to 0 or 1 (overconfidence)
        loss_reg = torch.mean(alpha * (1 - alpha))  # Maximum at α=0.5
        loss_reg = -loss_reg  # Negative because we want to maximize diversity
        
        # Total loss
        total_loss = (
            self.lambda_mse * loss_mse +
            self.lambda_directional * loss_directional +
            self.lambda_agreement * loss_agreement +
            self.lambda_reg * loss_reg
        )
        
        # Log components
        loss_components = {
            "total": total_loss.item(),
            "mse": loss_mse.item(),
            "directional": loss_directional.item(),
            "agreement": loss_agreement.item(),
            "regularization": loss_reg.item()
        }
        
        return total_loss, loss_components


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = Config.EARLY_STOP_PATIENCE,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' or 'max' (whether lower or higher is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric.
            epoch: Current epoch number.
        
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            is_improvement = score < self.best_score - self.min_delta
        else:
            is_improvement = score > self.best_score + self.min_delta
        
        if is_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


class Trainer:
    """
    Main training engine for Neuro-Econometric Network.
    
    Implements:
    - Custom training loop with gradient descent
    - Learning rate scheduling
    - Gradient clipping for stability
    - Checkpointing best model
    - Training/validation split
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: NeuroEconometricNet,
        learning_rate: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY,
        device: torch.device = Config.DEVICE,
        checkpoint_dir: Path = Config.MODEL_DIR
    ):
        """
        Initialize trainer.
        
        Args:
            model: NeuroEconometricNet instance.
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization coefficient.
            device: Computation device.
            checkpoint_dir: Directory for saving checkpoints.
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler (ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Loss function
        self.criterion = HybridLoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=Config.EARLY_STOP_PATIENCE)
        
        # Training wrapper
        self.wrapper = HybridTrainingWrapper(self.model, device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'alpha_mean': [],
            'alpha_std': []
        }
        
        logger.info("Trainer initialized")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        epoch_losses = []
        epoch_loss_components = {
            'mse': [],
            'directional': [],
            'agreement': [],
            'regularization': []
        }
        alpha_values = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            features, ardl_preds, volatility_features, sentiment, targets = batch
            
            features = features.to(self.device)
            ardl_preds = ardl_preds.to(self.device)
            volatility_features = volatility_features.to(self.device)
            sentiment = sentiment.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions, alpha, neural_preds = self.model(
                features,
                ardl_preds,
                volatility_features,
                sentiment
            )
            
            # Compute loss
            loss, loss_components = self.criterion(
                predictions,
                targets,
                ardl_preds,
                alpha
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Log metrics
            epoch_losses.append(loss.item())
            for key, value in loss_components.items():
                if key != 'total':
                    epoch_loss_components[key].append(value)
            alpha_values.append(alpha.detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Epoch summary
        metrics = {
            'train_loss': np.mean(epoch_losses),
            'train_mse': np.mean(epoch_loss_components['mse']),
            'train_directional': np.mean(epoch_loss_components['directional']),
            'alpha_mean': np.mean(np.concatenate(alpha_values)),
            'alpha_std': np.std(np.concatenate(alpha_values))
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        val_losses = []
        val_predictions = []
        val_targets = []
        alpha_values = []
        
        with torch.no_grad():
            for batch in val_loader:
                features, ardl_preds, volatility_features, sentiment, targets = batch
                
                features = features.to(self.device)
                ardl_preds = ardl_preds.to(self.device)
                volatility_features = volatility_features.to(self.device)
                sentiment = sentiment.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions, alpha, neural_preds = self.model(
                    features,
                    ardl_preds,
                    volatility_features,
                    sentiment
                )
                
                # Compute loss
                loss, _ = self.criterion(predictions, targets, ardl_preds, alpha)
                
                val_losses.append(loss.item())
                val_predictions.append(predictions.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                alpha_values.append(alpha.cpu().numpy())
        
        # Compute metrics
        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)
        
        mse = np.mean((val_predictions - val_targets) ** 2)
        mae = np.mean(np.abs(val_predictions - val_targets))
        
        # Directional accuracy
        pred_direction = np.sign(val_predictions - val_targets.mean())
        true_direction = np.sign(val_targets - val_targets.mean())
        directional_accuracy = np.mean(pred_direction == true_direction)
        
        metrics = {
            'val_loss': np.mean(val_losses),
            'val_mse': mse,
            'val_mae': mae,
            'val_directional_accuracy': directional_accuracy,
            'alpha_mean': np.mean(np.concatenate(alpha_values)),
            'alpha_std': np.std(np.concatenate(alpha_values))
        }
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = Config.NUM_EPOCHS
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Number of training epochs.
        
        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Directional Acc: {val_metrics['val_directional_accuracy']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['learning_rates'].append(current_lr)
            self.history['alpha_mean'].append(train_metrics['alpha_mean'])
            self.history['alpha_std'].append(train_metrics['alpha_std'])
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics, best=True)
                logger.info(f"Best model saved with val_loss: {best_val_loss:.4f}")
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, best=False)
            
            # Early stopping
            if self.early_stopping(val_metrics['val_loss'], epoch):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number.
            metrics: Current metrics.
            best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        if best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
