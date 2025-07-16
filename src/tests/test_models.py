"""
Unit tests for CIFAR-10 ML model components.

This module contains tests for the CIFAR-10 CNN model and related components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cifar_cnn import CIFAR10CNN, CIFAR10Trainer
from data.cifar_dataset import CIFAR10Dataset, CIFAR10DataManager


class TestCIFAR10CNN:
    """Test cases for CIFAR-10 CNN model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'input_size': [32, 32, 3],
            'num_classes': 10,
            'dropout_rate': 0.3
        }
        self.model = CIFAR10CNN(self.config)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.input_size == [32, 32, 3]
        assert self.model.num_classes == 10
        assert self.model.dropout_rate == 0.3
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        
        output = self.model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        
        assert info['model_name'] == 'CIFAR10CNN'
        assert info['input_size'] == [32, 32, 3]
        assert info['num_classes'] == 10
        assert info['architecture'] == 'CNN'
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
        
        # CIFAR-10 model should have more parameters than MNIST due to RGB input
        assert total_params > 100000  # Should be much larger than MNIST model
    
    def test_model_device_transfer(self):
        """Test model transfer to different devices."""
        if torch.cuda.is_available():
            self.model.cuda()
            assert next(self.model.parameters()).is_cuda
            
            self.model.cpu()
            assert not next(self.model.parameters()).is_cuda
    
    def test_model_weight_initialization(self):
        """Test that model weights are properly initialized."""
        # Check that weights are not all zeros
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                assert not torch.all(param == 0), f"Weights in {name} are all zero"
                assert torch.isfinite(param).all(), f"Non-finite weights in {name}"


class TestCIFAR10Trainer:
    """Test cases for CIFAR-10 trainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'loss_function': 'cross_entropy',
            'scheduler': 'cosine_annealing',
            'scheduler_T_max': 100,
            'scheduler_eta_min': 0.00001
        }
        self.model = CIFAR10CNN({'input_size': [32, 32, 3], 'num_classes': 10, 'dropout_rate': 0.3})
        self.trainer = CIFAR10Trainer(self.model, self.config, device='cpu')
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.model == self.model
        assert self.trainer.config == self.config
        assert self.trainer.device == 'cpu'
        assert self.trainer.optimizer is not None
        assert self.trainer.criterion is not None
    
    def test_trainer_with_adam_optimizer(self):
        """Test trainer with Adam optimizer."""
        config = self.config.copy()
        config['optimizer'] = 'adam'
        
        trainer = CIFAR10Trainer(self.model, config, device='cpu')
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_trainer_with_sgd_optimizer(self):
        """Test trainer with SGD optimizer."""
        config = self.config.copy()
        config['optimizer'] = 'sgd'
        config['momentum'] = 0.9
        
        trainer = CIFAR10Trainer(self.model, config, device='cpu')
        assert isinstance(trainer.optimizer, torch.optim.SGD)
    
    def test_trainer_with_adamw_optimizer(self):
        """Test trainer with AdamW optimizer."""
        config = self.config.copy()
        config['optimizer'] = 'adamw'
        
        trainer = CIFAR10Trainer(self.model, config, device='cpu')
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
    
    def test_trainer_invalid_optimizer(self):
        """Test trainer with invalid optimizer."""
        config = self.config.copy()
        config['optimizer'] = 'invalid'
        
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            CIFAR10Trainer(self.model, config, device='cpu')
    
    def test_trainer_invalid_loss_function(self):
        """Test trainer with invalid loss function."""
        config = self.config.copy()
        config['loss_function'] = 'invalid'
        
        with pytest.raises(ValueError, match="Unsupported loss function"):
            CIFAR10Trainer(self.model, config, device='cpu')
    
    def test_cosine_annealing_scheduler(self):
        """Test cosine annealing scheduler."""
        config = self.config.copy()
        config['scheduler'] = 'cosine_annealing'
        
        trainer = CIFAR10Trainer(self.model, config, device='cpu')
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_step_scheduler(self):
        """Test step scheduler."""
        config = self.config.copy()
        config['scheduler'] = 'step'
        config['scheduler_step_size'] = 30
        config['scheduler_gamma'] = 0.1
        
        trainer = CIFAR10Trainer(self.model, config, device='cpu')
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)
    
    def test_reduce_on_plateau_scheduler(self):
        """Test reduce on plateau scheduler."""
        config = self.config.copy()
        config['scheduler'] = 'reduce_on_plateau'
        config['scheduler_patience'] = 10
        config['scheduler_factor'] = 0.5
        
        trainer = CIFAR10Trainer(self.model, config, device='cpu')
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading."""
        save_path = tmp_path / "test_cifar_model.pth"
        
        # Save model
        self.trainer.save_model(str(save_path))
        assert save_path.exists()
        
        # Load model
        new_trainer = CIFAR10Trainer(self.model, self.config, device='cpu')
        new_trainer.load_model(str(save_path))
        
        # Check that model parameters are the same
        for param1, param2 in zip(self.trainer.model.parameters(), 
                                 new_trainer.model.parameters()):
            assert torch.equal(param1, param2)
    
    def test_weight_decay_parameter(self):
        """Test that weight decay is properly set."""
        config = self.config.copy()
        config['weight_decay'] = 1e-3
        
        trainer = CIFAR10Trainer(self.model, config, device='cpu')
        assert trainer.optimizer.param_groups[0]['weight_decay'] == 1e-3


class TestCIFAR10Dataset:
    """Test cases for CIFAR-10 dataset."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_dir = "test_data"
        self.cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    @patch('torchvision.datasets.CIFAR10')
    def test_dataset_initialization(self, mock_cifar10):
        """Test dataset initialization."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 50000
        mock_cifar10.return_value = mock_dataset
        
        dataset = CIFAR10Dataset(self.data_dir, train=True)
        
        mock_cifar10.assert_called_once_with(
            root=self.data_dir,
            train=True,
            download=True,
            transform=dataset.transform
        )
        
        assert dataset.classes == self.cifar10_classes
    
    @patch('torchvision.datasets.CIFAR10')
    def test_dataset_length(self, mock_cifar10):
        """Test dataset length."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 50000
        mock_cifar10.return_value = mock_dataset
        
        dataset = CIFAR10Dataset(self.data_dir, train=True)
        assert len(dataset) == 50000
    
    @patch('torchvision.datasets.CIFAR10')
    def test_dataset_getitem(self, mock_cifar10):
        """Test dataset item retrieval."""
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = Mock(return_value=(torch.randn(3, 32, 32), 5))
        mock_cifar10.return_value = mock_dataset
        
        dataset = CIFAR10Dataset(self.data_dir, train=True)
        item = dataset[0]
        
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], int)
    
    def test_class_names_mapping(self):
        """Test class names mapping."""
        dataset = CIFAR10Dataset(self.data_dir, train=True)
        class_names = dataset.get_class_names()
        
        assert len(class_names) == 10
        assert class_names[0] == 'airplane'
        assert class_names[9] == 'truck'
        assert all(isinstance(name, str) for name in class_names.values())
    
    def test_training_transforms(self):
        """Test that training dataset has data augmentation."""
        dataset = CIFAR10Dataset(self.data_dir, train=True)
        
        # Check that transforms include data augmentation
        transform_names = [type(t).__name__ for t in dataset.transform.transforms]
        
        # Should include augmentation transforms for training
        assert 'RandomHorizontalFlip' in transform_names
        assert 'RandomRotation' in transform_names
        assert 'RandomCrop' in transform_names
        assert 'ToTensor' in transform_names
        assert 'Normalize' in transform_names
    
    def test_test_transforms(self):
        """Test that test dataset has minimal transforms."""
        dataset = CIFAR10Dataset(self.data_dir, train=False)
        
        # Check that transforms are minimal for test
        transform_names = [type(t).__name__ for t in dataset.transform.transforms]
        
        # Should only include basic transforms for test
        assert 'ToTensor' in transform_names
        assert 'Normalize' in transform_names
        # Should not include augmentation
        assert 'RandomHorizontalFlip' not in transform_names
        assert 'RandomRotation' not in transform_names


class TestCIFAR10DataManager:
    """Test cases for CIFAR-10 data manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
        self.data_dir = "test_data"
        self.data_manager = CIFAR10DataManager(self.data_dir, self.config)
    
    def test_data_manager_initialization(self):
        """Test data manager initialization."""
        assert self.data_manager.data_dir == self.data_dir
        assert self.data_manager.config == self.config
        assert self.data_manager.train_split == 0.8
        assert self.data_manager.val_split == 0.1
        assert self.data_manager.test_split == 0.1
    
    def test_invalid_splits(self):
        """Test data manager with invalid splits."""
        invalid_config = {
            'train_split': 0.5,
            'val_split': 0.3,
            'test_split': 0.3  # Sum > 1
        }
        
        with pytest.raises(ValueError, match="Data splits must sum to 1.0"):
            CIFAR10DataManager(self.data_dir, invalid_config)
    
    @patch('data.dataset.CIFAR10Dataset')
    def test_create_data_loaders(self, mock_dataset_class):
        """Test data loader creation."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 50000
        mock_dataset_class.return_value = mock_dataset
        
        train_loader, val_loader, test_loader = self.data_manager.create_data_loaders(128)
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check that persistent workers are enabled
        assert train_loader.persistent_workers == True
        assert val_loader.persistent_workers == True
        assert test_loader.persistent_workers == True
    
    @patch('data.dataset.CIFAR10Dataset')
    def test_get_test_loader(self, mock_dataset_class):
        """Test test loader creation."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10000
        mock_dataset_class.return_value = mock_dataset
        
        test_loader = self.data_manager.get_test_loader(128)
        
        assert test_loader is not None
        assert test_loader.batch_size == 128
        assert test_loader.persistent_workers == True
    
    def test_save_data_info(self, tmp_path):
        """Test data info saving."""
        output_path = tmp_path / "cifar_data_info.json"
        
        with patch('data.dataset.CIFAR10Dataset') as mock_dataset_class:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 50000
            mock_dataset.get_class_distribution = Mock(return_value={i: 5000 for i in range(10)})
            mock_dataset.get_class_names = Mock(return_value={i: f"class_{i}" for i in range(10)})
            mock_dataset_class.return_value = mock_dataset
            
            info = self.data_manager.save_data_info(str(output_path))
            
            assert output_path.exists()
            assert 'train_size' in info
            assert 'test_size' in info
            assert 'num_classes' in info
            assert 'class_names' in info
            assert 'image_size' in info
            assert info['image_size'] == [32, 32, 3]
            assert info['num_classes'] == 10


class TestCIFAR10ModelIntegration:
    """Integration tests for CIFAR-10 model components."""
    
    def test_model_training_cycle(self):
        """Test complete model training cycle."""
        # Create model
        config = {
            'input_size': [32, 32, 3],
            'num_classes': 10,
            'dropout_rate': 0.3
        }
        model = CIFAR10CNN(config)
        
        # Create trainer
        trainer_config = {
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'loss_function': 'cross_entropy'
        }
        trainer = CIFAR10Trainer(model, trainer_config, device='cpu')
        
        # Create dummy data (RGB images)
        batch_size = 4
        dummy_data = torch.randn(batch_size, 3, 32, 32)
        dummy_targets = torch.randint(0, 10, (batch_size,))
        
        # Create mock data loader
        mock_loader = [(dummy_data, dummy_targets)]
        
        # Test training epoch
        metrics = trainer.train_epoch(mock_loader, epoch=1)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'learning_rate' in metrics
        assert metrics['loss'] >= 0
        assert 0 <= metrics['accuracy'] <= 100
    
    def test_model_validation(self):
        """Test model validation."""
        # Create model
        config = {
            'input_size': [32, 32, 3],
            'num_classes': 10,
            'dropout_rate': 0.3
        }
        model = CIFAR10CNN(config)
        
        # Create trainer
        trainer_config = {
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'loss_function': 'cross_entropy'
        }
        trainer = CIFAR10Trainer(model, trainer_config, device='cpu')
        
        # Create dummy data (RGB images)
        batch_size = 4
        dummy_data = torch.randn(batch_size, 3, 32, 32)
        dummy_targets = torch.randint(0, 10, (batch_size,))
        
        # Create mock data loader
        mock_loader = [(dummy_data, dummy_targets)]
        
        # Test validation
        metrics = trainer.validate(mock_loader)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['loss'] >= 0
        assert 0 <= metrics['accuracy'] <= 100
    
    def test_model_with_different_batch_sizes(self):
        """Test model with different batch sizes."""
        config = {
            'input_size': [32, 32, 3],
            'num_classes': 10,
            'dropout_rate': 0.3
        }
        model = CIFAR10CNN(config)
        
        # Test with different batch sizes
        batch_sizes = [1, 16, 32, 64]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 32, 32)
            output = model(input_tensor)
            
            assert output.shape == (batch_size, 10)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = {
            'input_size': [32, 32, 3],
            'num_classes': 10,
            'dropout_rate': 0.3
        }
        model = CIFAR10CNN(config)
        
        # Create dummy data
        input_tensor = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 10, (2,))
        
        # Forward pass
        output = model(input_tensor)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert input_tensor.grad is not None
        
        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
    
    def test_model_deterministic_output(self):
        """Test that model produces deterministic output in eval mode."""
        config = {
            'input_size': [32, 32, 3],
            'num_classes': 10,
            'dropout_rate': 0.3
        }
        model = CIFAR10CNN(config)
        model.eval()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        input_tensor = torch.randn(2, 3, 32, 32)
        
        # Two forward passes should give same result in eval mode
        torch.manual_seed(42)
        output1 = model(input_tensor)
        
        torch.manual_seed(42)
        output2 = model(input_tensor)
        
        assert torch.equal(output1, output2)
    
    def test_model_stochastic_output_training(self):
        """Test that model produces stochastic output in training mode due to dropout."""
        config = {
            'input_size': [32, 32, 3],
            'num_classes': 10,
            'dropout_rate': 0.5  # High dropout for noticeable difference
        }
        model = CIFAR10CNN(config)
        model.train()
        
        input_tensor = torch.randn(2, 3, 32, 32)
        
        # Two forward passes should give different results in training mode
        output1 = model(input_tensor)
        output2 = model(input_tensor)
        
        # With dropout, outputs should be different
        assert not torch.equal(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__])
