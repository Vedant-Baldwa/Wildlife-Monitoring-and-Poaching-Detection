#!/usr/bin/env python3
"""
Script to restart training with improved configuration
"""

import os
import sys
import yaml
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO

def load_config(config_path):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def restart_training():
    """Restart training with improved configuration"""
    
    # Load configuration
    config_path = project_root / "config" / "wildlife_training_config.yaml"
    config = load_config(config_path)
    
    print("=== Wildlife Detection Training Restart ===")
    print(f"Configuration loaded from: {config_path}")
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['data']['yaml_path']}")
    print(f"Epochs: {config['data']['epochs']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Image size: {config['data']['imgsz']}")
    print(f"Learning rate: {config['training']['lr0']}")
    print(f"Project name: {config['training']['name']}")
    
    # Check if best checkpoint exists
    best_checkpoint = project_root / "results" / "wildlife_detection_v2" / "weights" / "best.pt"
    if best_checkpoint.exists():
        print(f"\nFound best checkpoint: {best_checkpoint}")
        print("Will resume training from best checkpoint...")
        model_path = str(best_checkpoint)
    else:
        print(f"\nNo best checkpoint found at: {best_checkpoint}")
        print("Will start training from scratch...")
        model_path = config['model']['name']
    
    # Initialize model
    model = YOLO(model_path)
    
    # Prepare training arguments
    train_args = {
        'data': str(project_root / config['data']['yaml_path']),
        'epochs': config['data']['epochs'],
        'imgsz': config['data']['imgsz'],
        'batch': config['data']['batch_size'],
        'device': config['data']['device'],
        'project': str(project_root / config['training']['project']),
        'name': config['training']['name'],
        'exist_ok': config['training']['exist_ok'],
        'pretrained': config['training']['pretrained'],
        'optimizer': config['training']['optimizer'],
        'lr0': config['training']['lr0'],
        'lrf': config['training']['lrf'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'warmup_momentum': config['training']['warmup_momentum'],
        'warmup_bias_lr': config['training']['warmup_bias_lr'],
        'box': config['training']['box'],
        'cls': config['training']['cls'],
        'dfl': config['training']['dfl'],
        'label_smoothing': config['training']['label_smoothing'],
        'dropout': config['training']['dropout'],
        'patience': config['training']['patience'],
        'save_period': config['training']['save_period'],
        'seed': config['training']['seed'],
        'deterministic': config['training']['deterministic'],
        'single_cls': config['training']['single_cls'],
        'rect': config['training']['rect'],
        'cos_lr': config['training']['cos_lr'],
        'close_mosaic': config['training']['close_mosaic'],
        'amp': config['training']['amp'],
        'cache': config['training']['cache'],
        'workers': config['training']['workers'],
        'plots': config['training']['plots'],
        'save': config['training']['save'],
        'val': config['training']['val'],
        'verbose': config['training']['verbose'],
    }
    
    print("\n=== Training Arguments ===")
    for key, value in train_args.items():
        print(f"{key}: {value}")
    
    # Start training
    print(f"\nStarting training with improved configuration...")
    print("Press Ctrl+C to stop training early if needed.")
    
    try:
        results = model.train(**train_args)
        print(f"\nTraining completed successfully!")
        print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    restart_training()
