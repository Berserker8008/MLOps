"""
Simple CIFAR-10 CNN Model

A lightweight CNN architecture for CIFAR-10 image classification.
Much simpler than ResNet but still effective for learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import structlog

logger = structlog.get_logger()


class SimpleCNN(nn.Module):
    """Simple CNN architecture for CIFAR-10 classification."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simple CNN model.
        
        Args:
            config: Model configuration dictionary
        """
        super(SimpleCNN, self).__init__()
        
        self.num_classes = config.get('num_classes', 10)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_classes)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        logger.info("Initialized Simple CNN model",
                   num_classes=self.num_classes,
                   dropout_rate=self.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers with dropout
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
            "model_name": "SimpleCNN",
            "num_classes": int(self.num_classes),
            "dropout_rate": float(self.dropout_rate),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "architecture": "Simple CNN"
        }


class SimpleTrainer:
    """Trainer class for the simple CNN model."""
    
    def __init__(self, model: SimpleCNN, config: Dict[str, Any], device: str = "cpu"):
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
        
        # Setup optimizer (default to Adam for simplicity)
        learning_rate = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 1e-4)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Simple step scheduler (optional)
        if config.get('use_scheduler', True):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config.get('scheduler_step_size', 20), 
                gamma=config.get('scheduler_gamma', 0.5)
            )
        else:
            self.scheduler = None
        
        logger.info("Initialized simple trainer",
                   learning_rate=learning_rate,
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
        if self.scheduler:
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


# Example usage
if __name__ == "__main__":
    # Simple model configuration
    config = {
        'num_classes': 10,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'use_scheduler': True,
        'scheduler_step_size': 20,
        'scheduler_gamma': 0.5
    }
    
    # Create model
    model = SimpleCNN(config)
    
    # Print model info
    info = model.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Parameters: {info['total_parameters']:,}")
    print(f"Architecture: {info['architecture']}")
    
    # Example with dummy data
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
