#!/usr/bin/env python3
"""
Download and prepare CIFAR-10 dataset.

This script downloads the CIFAR-10 dataset and prepares it for training.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import structlog
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.cifar_dataset import CIFAR10Dataset, CIFAR10DataManager

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

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def download_cifar10_data(data_dir: str = "data/cifar10", config: dict = None):
    """
    Download and prepare CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store the data
        config: Configuration dictionary
    """
    if config is None:
        config = {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info("Starting CIFAR-10 data download", data_dir=data_dir)
    
    try:
        # Download training data
        logger.info("Downloading training data")
        train_dataset = CIFAR10Dataset(data_dir, train=True)
        
        # Download test data
        logger.info("Downloading test data")
        test_dataset = CIFAR10Dataset(data_dir, train=False)
        
        # Create data manager
        data_manager = CIFAR10DataManager(data_dir, config)
        
        # Save data information
        data_info_path = os.path.join(data_dir, "data_info.json")
        data_info = data_manager.save_data_info(data_info_path)
        
        # Create additional statistics
        logger.info("Computing dataset statistics")
        train_stats = compute_dataset_statistics(train_dataset)
        test_stats = compute_dataset_statistics(test_dataset)
        
        # Create comprehensive metrics
        metrics = {
            "dataset_name": "CIFAR-10",
            "train_size": data_info["train_size"],
            "test_size": data_info["test_size"],
            "num_classes": data_info["num_classes"],
            "image_size": data_info["image_size"],
            "class_names": data_info["class_names"],
            "train_class_distribution": data_info["train_class_distribution"],
            "test_class_distribution": data_info["test_class_distribution"],
            "train_statistics": train_stats,
            "test_statistics": test_stats,
            "data_splits": data_info["splits"]
        }
        
        # Save metrics
        metrics_path = os.path.join(data_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save class distribution visualization data
        save_class_distribution_data(data_info, data_dir)
        
        logger.info("CIFAR-10 data download completed",
                   train_size=data_info["train_size"],
                   test_size=data_info["test_size"],
                   data_dir=data_dir)
        
        return data_info
        
    except Exception as e:
        logger.error("Failed to download CIFAR-10 data", error=str(e))
        raise


def compute_dataset_statistics(dataset):
    """Compute statistics for the dataset."""
    logger.info("Computing dataset statistics", dataset_size=len(dataset))
    
    # Sample a subset for statistics computation (to avoid memory issues)
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    pixel_values = []
    for idx in indices:
        image, _ = dataset[idx]
        # Convert tensor to numpy if needed
        if hasattr(image, 'numpy'):
            image = image.numpy()
        pixel_values.append(image.flatten())
    
    pixel_values = np.concatenate(pixel_values)
    
    stats = {
        "mean": float(np.mean(pixel_values)),
        "std": float(np.std(pixel_values)),
        "min": float(np.min(pixel_values)),
        "max": float(np.max(pixel_values)),
        "median": float(np.median(pixel_values)),
        "sample_size": sample_size
    }
    
    return stats


def save_class_distribution_data(data_info, data_dir):
    """Save class distribution data for visualization."""
    train_dist = data_info["train_class_distribution"]
    test_dist = data_info["test_class_distribution"]
    
    # Create distribution data with class names
    distribution_data = {
        "classes": CIFAR10_CLASSES,
        "train_distribution": [train_dist.get(i, 0) for i in range(10)],
        "test_distribution": [test_dist.get(i, 0) for i in range(10)]
    }
    
    dist_path = os.path.join(data_dir, "class_distribution.json")
    with open(dist_path, 'w') as f:
        json.dump(distribution_data, f, indent=2)
    
    logger.info("Saved class distribution data", path=dist_path)


def verify_dataset_integrity(data_dir: str):
    """Verify the integrity of the downloaded dataset."""
    logger.info("Verifying dataset integrity", data_dir=data_dir)
    
    try:
        # Check if data files exist
        required_files = ["data_info.json", "metrics.json", "class_distribution.json"]
        for file_name in required_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning("Missing file", file=file_path)
                return False
        
        # Load and verify data info
        data_info_path = os.path.join(data_dir, "data_info.json")
        with open(data_info_path, 'r') as f:
            data_info = json.load(f)
        
        # Verify expected values
        expected_train_size = 50000
        expected_test_size = 10000
        expected_num_classes = 10
        
        if data_info["train_size"] != expected_train_size:
            logger.error("Unexpected train size", 
                        expected=expected_train_size, 
                        actual=data_info["train_size"])
            return False
        
        if data_info["test_size"] != expected_test_size:
            logger.error("Unexpected test size", 
                        expected=expected_test_size, 
                        actual=data_info["test_size"])
            return False
        
        if data_info["num_classes"] != expected_num_classes:
            logger.error("Unexpected number of classes", 
                        expected=expected_num_classes, 
                        actual=data_info["num_classes"])
            return False
        
        # Verify class distribution
        train_dist = data_info["train_class_distribution"]
        test_dist = data_info["test_class_distribution"]
        
        if len(train_dist) != 10 or len(test_dist) != 10:
            logger.error("Unexpected class distribution length")
            return False
        
        # Check if all classes are present
        for i in range(10):
            if str(i) not in train_dist or str(i) not in test_dist:
                logger.error("Missing class in distribution", class_id=i)
                return False
        
        logger.info("Dataset integrity verification passed")
        return True
        
    except Exception as e:
        logger.error("Dataset integrity verification failed", error=str(e))
        return False


def create_data_summary(data_dir: str):
    """Create a summary of the downloaded data."""
    try:
        # Load data info
        data_info_path = os.path.join(data_dir, "data_info.json")
        with open(data_info_path, 'r') as f:
            data_info = json.load(f)
        
        # Load metrics
        metrics_path = os.path.join(data_dir, "metrics.json")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Create summary
        summary = {
            "dataset": "CIFAR-10",
            "description": "The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.",
            "classes": CIFAR10_CLASSES,
            "statistics": {
                "total_images": data_info["train_size"] + data_info["test_size"],
                "train_images": data_info["train_size"],
                "test_images": data_info["test_size"],
                "image_dimensions": "32x32x3",
                "num_classes": 10,
                "images_per_class": 6000
            },
            "splits": data_info["splits"],
            "files": {
                "data_info": "data_info.json",
                "metrics": "metrics.json",
                "class_distribution": "class_distribution.json"
            }
        }
        
        # Save summary
        summary_path = os.path.join(data_dir, "README.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Created data summary", path=summary_path)
        return summary
        
    except Exception as e:
        logger.error("Failed to create data summary", error=str(e))
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download CIFAR-10 dataset")
    parser.add_argument("--data-dir", default="data/cifar10", 
                       help="Directory to store the data")
    parser.add_argument("--config", default=None,
                       help="Path to configuration file")
    parser.add_argument("--verify", action="store_true",
                       help="Verify dataset integrity after download")
    parser.add_argument("--force", action="store_true",
                       help="Force redownload even if data exists")
    
    args = parser.parse_args()
    
    # Check if data already exists
    if os.path.exists(args.data_dir) and not args.force:
        data_info_path = os.path.join(args.data_dir, "data_info.json")
        if os.path.exists(data_info_path):
            print(f"⚠️  CIFAR-10 data already exists in {args.data_dir}")
            print(f"    Use --force to redownload")
            
            if args.verify:
                print("🔍 Verifying existing data...")
                if verify_dataset_integrity(args.data_dir):
                    print("✅ Dataset integrity verification passed")
                else:
                    print("❌ Dataset integrity verification failed")
                    return
            
            # Load existing data info
            with open(data_info_path, 'r') as f:
                data_info = json.load(f)
        else:
            # Directory exists but no data_info.json, proceed with download
            data_info = download_cifar10_data(args.data_dir)
    else:
        # Load configuration if provided
        config = None
        if args.config and os.path.exists(args.config):
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Download data
        data_info = download_cifar10_data(args.data_dir, config)
        
        # Verify dataset integrity if requested
        if args.verify:
            print("🔍 Verifying dataset integrity...")
            if not verify_dataset_integrity(args.data_dir):
                print("❌ Dataset integrity verification failed")
                return
    
    # Create data summary
    summary = create_data_summary(args.data_dir)
    
    print(f"✅ CIFAR-10 data ready!")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Training samples: {data_info['train_size']:,}")
    print(f"📊 Test samples: {data_info['test_size']:,}")
    print(f"🎯 Number of classes: {data_info['num_classes']}")
    print(f"🏷️ Classes: {', '.join(data_info['class_names'].values())}")
    print(f"📐 Image size: {data_info['image_size']}")
    
    # Print class distribution
    train_dist = data_info["train_class_distribution"]
    print(f"\n📊 Training set class distribution:")
    for i, class_name in enumerate(CIFAR10_CLASSES):
        count = train_dist.get(str(i), 0)
        print(f"   {class_name}: {count:,} images")
    
    print(f"\n📄 Files created:")
    print(f"   - data_info.json: Dataset information")
    print(f"   - metrics.json: Dataset metrics and statistics")
    print(f"   - class_distribution.json: Class distribution data")
    print(f"   - README.json: Dataset summary")


if __name__ == "__main__":
    main()
