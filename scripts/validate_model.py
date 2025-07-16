#!/usr/bin/env python3
"""
Test script to verify model metrics are calculated correctly.
"""

import os
import sys
import torch
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import CIFAR10DataManager
from models.cifar_cnn import CIFAR10CNN
from utils.metrics import ModelEvaluator

def test_model_metrics():
    """Test that model metrics are calculated correctly."""
    print("🧪 Testing model metrics...")
    
    # Check if model exists
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train a model first with: python scripts/train.py")
        return False
    
    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('config', {}).get('model', {})
        
        model = CIFAR10CNN(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"🔧 Device: {device}")
        
        # Create data manager
        data_config = {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
        data_manager = CIFAR10DataManager("data/cifar10", data_config)
        
        # Get test loader
        test_loader = data_manager.get_test_loader(batch_size=128)
        print(f"📊 Test loader created with {len(test_loader)} batches")
        
        # Evaluate model
        evaluator = ModelEvaluator(model, device=device)
        results = evaluator.evaluate_model(test_loader)
        
        metrics = results['metrics']
        
        print(f"\n📈 Model Performance Metrics:")
        print(f"   Accuracy: {metrics['accuracy']:.2f}%")
        print(f"   Loss: {metrics['loss']:.4f}")
        print(f"   Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"   Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"   F1-Score (macro): {metrics['f1_macro']:.4f}")
        
        # Check if accuracy is in reasonable range
        if 0 <= metrics['accuracy'] <= 100:
            print(f"✅ Accuracy is in correct percentage format")
        else:
            print(f"❌ Accuracy format issue: {metrics['accuracy']}")
        
        # Show class distribution
        class_dist = results['class_distribution']
        print(f"\n📊 Test set class distribution:")
        for class_idx, count in class_dist.items():
            print(f"   Class {class_idx}: {count} samples")
        
        # Save metrics for inspection
        output_path = "test_metrics_output.json"
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'class_distribution': class_dist
            }, f, indent=2)
        
        print(f"\n💾 Metrics saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing metrics: {e}")
        return False

if __name__ == "__main__":
    success = test_model_metrics()
    sys.exit(0 if success else 1)
