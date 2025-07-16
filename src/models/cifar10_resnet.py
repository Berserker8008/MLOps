"""
CIFAR-10 ResNet Model

This module contains the ResNet architecture for CIFAR-10 image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger()


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFAR10ResNet(nn.Module):
    """ResNet architecture for CIFAR-10 classification."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ResNet model.
        
        Args:
            config: Model configuration dictionary
        """
        super(CIFAR10ResNet, self).__init__()
        
        self.input_size = config.get('input_size', [32, 32])
        self.num_classes = config.get('num_classes', 10)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
        # ResNet configuration
        num_blocks = config.get('num_blocks', [2, 2, 2, 2])  # ResNet-18 style
        
        self.in_planes = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(512 * BasicBlock.expansion, self.num_classes)
        
        logger.info("Initialized CIFAR-10 ResNet model",
                   input_size=self.input_size,
                   num_classes=self.num_classes,
                   dropout_rate=self.dropout_rate,
                   num_blocks=num_blocks)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a residual layer."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout and classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "CIFAR10ResNet",
            "input_size": self.input_size,
            "num_classes": int(self.num_classes),
            "dropout_rate": float(self.dropout_rate),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "architecture": "ResNet"
        }


class CIFAR10Trainer:
    """Trainer class for CIFAR-10 ResNet model."""
    
    def __init__(self, model: CIFAR10ResNet, config: Dict[str, Any], device: str = "cpu"):
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
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup loss function
        loss_function = config.get('loss_function', 'cross_entropy')
        if loss_function == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # Setup scheduler
        scheduler_name = config.get('scheduler', 'cosine')
        if scheduler_name == 'cosine':
            T_max = config.get('scheduler_t_max', 200)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
        elif scheduler_name == 'step':
            step_size = config.get('scheduler_step_size', 50)
            gamma = config.get('scheduler_gamma', 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        else:
            self.scheduler = None
        
        logger.info("Initialized trainer",
                   optimizer=optimizer_name,
                   learning_rate=learning_rate,
                   loss_function=loss_function,
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
