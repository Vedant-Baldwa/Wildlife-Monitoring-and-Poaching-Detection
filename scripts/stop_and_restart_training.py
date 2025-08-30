#!/usr/bin/env python3
"""
Script to stop current training and restart with improved configuration
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def stop_training():
    """Stop any running training processes"""
    print("Stopping any running training processes...")
    
    try:
        # On Windows, look for Python processes that might be training
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'python.exe' in line and 'train' in line.lower():
                    # Extract PID and kill the process
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            print(f"Killing training process with PID: {pid}")
                            subprocess.run(['taskkill', '/PID', str(pid), '/F'], 
                                         shell=True, check=True)
                        except (ValueError, subprocess.CalledProcessError):
                            pass
        
        print("Training processes stopped.")
        
    except Exception as e:
        print(f"Error stopping processes: {e}")
        print("Please manually stop any running training processes.")

def restart_training():
    """Restart training with improved configuration"""
    print("\n=== Restarting Training with Improved Configuration ===")
    
    # Import after adding to path
    from ultralytics import YOLO
    import yaml
    
    # Load configuration
    config_path = project_root / "config" / "wildlife_training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print(f"Configuration: {config_path}")
    print(f"Model: {config['model']['name']}")
    print(f"Learning rate: {config['training']['lr0']}")
    print(f"Project name: {config['training']['name']}")
    
    # Check for best checkpoint
    best_checkpoint = project_root / "results" / "wildlife_detection_v2" / "weights" / "best.pt"
    if best_checkpoint.exists():
        print(f"Found best checkpoint: {best_checkpoint}")
        print("Resuming from best checkpoint...")
        model_path = str(best_checkpoint)
    else:
        print("No checkpoint found, starting from scratch...")
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
    
    # Add augmentation parameters if they exist
    augmentation_params = [
        'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 
        'shear', 'perspective', 'flipud', 'fliplr', 'mosaic'
    ]
    
    for param in augmentation_params:
        if param in config['training']:
            train_args[param] = config['training'][param]
    
    print("\n=== Key Improvements ===")
    print(f"• Increased learning rate: {config['training']['lr0']}")
    print(f"• Added label smoothing: {config['training']['label_smoothing']}")
    print(f"• Added dropout: {config['training']['dropout']}")
    print(f"• Adjusted loss weights (box: {config['training']['box']}, cls: {config['training']['cls']})")
    print(f"• Reduced patience: {config['training']['patience']}")
    print(f"• Added data augmentation for class imbalance")
    
    print(f"\nStarting training...")
    print("Press Ctrl+C to stop training early if needed.")
    
    try:
        results = model.train(**train_args)
        print(f"\n✅ Training completed successfully!")
        print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training stopped by user.")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        return False
    
    return True

def main():
    """Main function to stop and restart training"""
    print("=== Wildlife Detection Training Restart ===")
    
    # Stop current training
    stop_training()
    
    # Wait a moment
    time.sleep(2)
    
    # Restart with improved configuration
    restart_training()

if __name__ == "__main__":
    main()
