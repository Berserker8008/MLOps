"""
CIFAR-10 CNN Model

This module contains the CNN architecture for CIFAR-10 image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger()


class CIFAR10CNN(nn.Module):
    """Convolutional Neural Network for CIFAR-10 image classification."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CNN model.
        
        Args:
            config: Model configuration dictionary
        """
        super(CIFAR10CNN, self).__init__()
        
        self.input_size = config.get('input_size', [32, 32, 3])
        self.num_classes = config.get('num_classes', 10)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Calculate the size after convolutions and pooling
        # Input: 32x32 -> After 3 pooling layers: 
        conv_output_size = 4 * 256
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info("Initialized CIFAR-10 CNN model",
                   input_size=self.input_size,
                   num_classes=self.num_classes,
                   dropout_rate=self.dropout_rate)
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third convolutional block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "CIFAR10CNN",
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "CNN"
        }


class CIFAR10Trainer:
    """Trainer class for CIFAR-10 CNN model."""
    
    def __init__(self, model: CIFAR10CNN, config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        optimizer_name = config.get('optimizer', 'adam').lower()
        learning_rate = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = config.get('momentum', 0.9)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup loss function
        loss_function = config.get('loss_function', 'cross_entropy')
        if loss_function == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # Setup scheduler
        scheduler_name = config.get('scheduler', 'cosine_annealing')
        if scheduler_name == 'step':
            step_size = config.get('scheduler_step_size', 30)
            gamma = config.get('scheduler_gamma', 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == 'cosine_annealing':
            T_max = config.get('scheduler_T_max', 100)
            eta_min = config.get('scheduler_eta_min', 0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_name == 'reduce_on_plateau':
            patience = config.get('scheduler_patience', 10)
            factor = config.get('scheduler_factor', 0.5)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=patience, factor=factor
            )
        else:
            self.scheduler = None
        
        logger.info("Initialized CIFAR-10 trainer",
                   optimizer=optimizer_name,
                   learning_rate=learning_rate,
                   weight_decay=weight_decay,
                   loss_function=loss_function,
                   scheduler=scheduler_name,
                   device=device)
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f"Training Epoch: {epoch}",
                           batch=batch_idx,
                           loss=loss.item(),
                           accuracy=100. * correct / total)
        
        # Update learning rate
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Update scheduler if using ReduceLROnPlateau
        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_info': self.model.get_model_info()
        }, path)
        logger.info("Saved model", path=path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded model", path=path)
