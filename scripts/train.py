#!/usr/bin/env python3
"""
Train CIFAR-10 CNN model with MLflow experiment tracking.

This script trains a CNN model on the CIFAR-10 dataset and tracks experiments with MLflow.
"""

import os
import sys
import json
import argparse
import yaml
import mlflow
import mlflow.pytorch
from pathlib import Path
import structlog
import torch
import numpy as np
from json_tricks import dump


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.cifar_dataset import CIFAR10DataManager
from models.cifar_cnn import CIFAR10CNN, CIFAR10Trainer
from utils.metrics import ModelEvaluator

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config: dict, data_dir: str = "data/cifar10", 
                models_dir: str = "models", device: str = "cpu"):
    """
    Train the CIFAR-10 CNN model.
    
    Args:
        config: Training configuration
        data_dir: Directory containing the data
        models_dir: Directory to save models
        device: Device to train on ('cpu' or 'cuda')
    """
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Set up MLflow
    mlflow.set_tracking_uri(config['experiment']['tracking_uri'])
    mlflow.set_experiment(config['experiment']['name'])
    
    with mlflow.start_run():
        # Log parameters
        training_params = {
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['training']['batch_size'],
            "epochs": config['training']['epochs'],
            "optimizer": config['training']['optimizer'],
            "dropout_rate": config['model']['dropout_rate'],
            "weight_decay": config['training']['weight_decay'],
            "scheduler": config['training']['scheduler'],
            "early_stopping_patience": config['validation']['early_stopping_patience']
        }
        
        # Log hyperparameters
        hyperparams = config.get('hyperparameters', {})
        all_params = {**training_params, **hyperparams}
        
        mlflow.log_params(all_params)
        
        # Create data manager
        data_manager = CIFAR10DataManager(data_dir, config['data'])
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_manager.create_data_loaders(
            config['training']['batch_size']
        )
        
        # Create model
        model = CIFAR10CNN(config['model'])
        
        # Create trainer
        trainer = CIFAR10Trainer(model, config['training'], device=device)
        
        # Log model architecture info
        model_info = model.get_model_info()
        mlflow.log_params({
            "total_parameters": model_info['total_parameters'],
            "trainable_parameters": model_info['trainable_parameters'],
            "model_name": model_info['model_name']
        })
        
        # Training loop
        best_val_accuracy = 0.0
        early_stopping_counter = 0
        early_stopping_patience = config['validation']['early_stopping_patience']
        
        logger.info("Starting training", 
                   epochs=config['training']['epochs'],
                   device=device,
                   model_params=model_info['total_parameters'])
        
        for epoch in range(config['training']['epochs']):
            # Train epoch
            train_metrics = trainer.train_epoch(train_loader, epoch + 1)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                f"train_loss": train_metrics['loss'],
                f"train_accuracy": train_metrics['accuracy'],
                f"val_loss": val_metrics['loss'],
                f"val_accuracy": val_metrics['accuracy'],
                f"learning_rate": train_metrics['learning_rate']
            }, step=epoch)
            
            # Log additional metrics
            mlflow.log_metrics({
                f"epoch": epoch + 1,
                f"early_stopping_counter": early_stopping_counter
            }, step=epoch)
            
            logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}",
                       train_loss=train_metrics['loss'],
                       train_accuracy=train_metrics['accuracy'],
                       val_loss=val_metrics['loss'],
                       val_accuracy=val_metrics['accuracy'],
                       lr=train_metrics['learning_rate'])
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_path = os.path.join(models_dir, "best_model.pth")
                trainer.save_model(best_model_path)
                
                # Log model artifact
                mlflow.log_artifact(best_model_path, "models")
                
                early_stopping_counter = 0
                logger.info("New best model saved", 
                           accuracy=best_val_accuracy,
                           path=best_model_path)
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                logger.info("Early stopping triggered", 
                           patience=early_stopping_patience,
                           best_accuracy=best_val_accuracy)
                break
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        evaluator = ModelEvaluator(model, device=device)
        test_results = evaluator.evaluate_model(test_loader)
        
        # Log test metrics (accuracy now consistently in percentage format)
        test_metrics = test_results['metrics']
        
        mlflow.log_metrics({
            "test_accuracy": test_metrics['accuracy'],
            "test_loss": test_metrics['loss'],
            "test_precision_macro": test_metrics['precision_macro'],
            "test_recall_macro": test_metrics['recall_macro'],
            "test_f1_macro": test_metrics['f1_macro']
        })
        
        # Log per-class metrics
        for i in range(10):
            mlflow.log_metrics({
                f"test_precision_class_{i}": test_metrics.get(f'precision_class_{i}', 0),
                f"test_recall_class_{i}": test_metrics.get(f'recall_class_{i}', 0),
                f"test_f1_class_{i}": test_metrics.get(f'f1_class_{i}', 0)
            })
        
        # Save final model
        final_model_path = os.path.join(models_dir, "final_model.pth")
        trainer.save_model(final_model_path)
        mlflow.log_artifact(final_model_path, "models")
        
        # Save metrics
        metrics_path = os.path.join(models_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(convert_numpy(test_metrics), f, indent=2)
        mlflow.log_artifact(metrics_path, "metrics")
        
        # Log model info
        model_info_path = os.path.join(models_dir, "model_info.json")
        with open(model_info_path, 'w') as f:
            dump(model_info, f, indent=2)
        mlflow.log_artifact(model_info_path, "model_info")
        
        # Log class distribution
        class_dist_path = os.path.join(models_dir, "class_distribution.json")
        with open(class_dist_path, 'w') as f:
            json.dump(convert_numpy(test_results['class_distribution']), f, indent=2)
        mlflow.log_artifact(class_dist_path, "analysis")
        
        logger.info("Training completed",
                   best_val_accuracy=best_val_accuracy,
                   test_accuracy=test_metrics['accuracy'],
                   test_f1_macro=test_metrics['f1_macro'])
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'test_metrics': test_metrics,
            'model_info': model_info,
            'class_distribution': test_results['class_distribution']
        }


def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        # Convert both keys and values to handle numpy types in dictionary keys
        return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(i) for i in obj)
    elif isinstance(obj, set):
        return list(convert_numpy(i) for i in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train CIFAR-10 CNN model")
    parser.add_argument("--config", default="configs/training.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--data-dir", default="data/cifar10",
                       help="Directory containing the data")
    parser.add_argument("--models-dir", default="models",
                       help="Directory to save models")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to train on ('cpu' or 'cuda')")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error("Configuration file not found", path=args.config)
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Override parameters if provided
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    # Train model
    try:
        results = train_model(config, args.data_dir, args.models_dir, args.device)
        
        print(f"✅ CIFAR-10 training completed successfully!")
        print(f"🏆 Best validation accuracy: {results['best_val_accuracy']:.2f}%")
        print(f"📊 Test accuracy: {results['test_metrics']['accuracy']:.2f}%")
        print(f"📊 Test F1-macro: {results['test_metrics']['f1_macro']:.4f}")
        print(f"🔢 Model parameters: {results['model_info']['total_parameters']:,}")
        print(f"📁 Models saved to: {args.models_dir}")
        print(f"🔗 MLflow experiment: {config['experiment']['name']}")
        
        # Print class distribution
        print(f"\n📊 Test set class distribution:")
        for class_idx, count in results['class_distribution'].items():
            print(f"   Class {class_idx}: {count} samples")
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
