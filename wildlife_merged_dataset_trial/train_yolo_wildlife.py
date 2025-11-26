#!/usr/bin/env python3
"""
Optimized YOLO Training Script for Wildlife Detection
Optimized for RTX 4050 with 6GB VRAM
Features:
- GPU memory optimization
- Automatic batch size calculation
- Training stability improvements
- Comprehensive logging
- Model checkpointing
"""

import os
import sys
import yaml
import argparse
import torch
import psutil
import GPUtil
from pathlib import Path
from ultralytics import YOLO
import logging
from time import time
import numpy as np
import gc

# ---------- Logging Setup ----------
LOGFILE = "wildlife_training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE), 
        logging.StreamHandler(sys.stdout)
    ]
)

# ---------- Configuration ----------
DEFAULT_CONFIG_PATH = Path("config/wildlife_training_config.yaml")

def check_system_resources():
    """Check system resources and GPU availability."""
    logging.info("=== System Resources Check ===")
    
    # CPU Info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    logging.info(f"CPU: {cpu_count} cores, {cpu_percent:.1f}% usage")
    logging.info(f"RAM: {memory.total / 1024**3:.1f}GB total, {memory.available / 1024**3:.1f}GB available")
    
    # GPU Info
    gpu_available = False
    if torch.cuda.is_available():
        gpu_available = True
        logging.info(f"CUDA Version: {torch.version.cuda}")
        logging.info(f"PyTorch Version: {torch.__version__}")
        logging.info(f"GPU Count: {torch.cuda.device_count()}")
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                logging.info(f"GPU {i}: {gpu.name}")
                logging.info(f"  Memory: {gpu.memoryTotal}MB total, {gpu.memoryFree}MB free")
                logging.info(f"  Load: {gpu.load*100:.1f}%")
                logging.info(f"  Temperature: {gpu.temperature}°C")
        except Exception as e:
            logging.warning(f"Could not get detailed GPU info: {e}")
        
        # PyTorch GPU info
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"PyTorch GPU {i}: {props.name}")
            logging.info(f"  Memory: {props.total_memory / 1024**3:.1f}GB")
            logging.info(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        logging.warning("CUDA is not available - training will use CPU")
    
    return gpu_available

def calculate_optimal_batch_size(model_size='s', gpu_memory_gb=None):
    """Calculate optimal batch size for RTX 4050 with 6GB VRAM."""
    if gpu_memory_gb is None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory_gb = gpus[0].memoryTotal / 1024
            else:
                gpu_memory_gb = 6  # RTX 4050 default
        except:
            gpu_memory_gb = 6  # RTX 4050 default
    
    # Conservative batch sizes for RTX 4050 (6GB VRAM)
    # These are optimized for stability and memory efficiency
    batch_sizes = {
        'n': {4: 8, 6: 12, 8: 16, 12: 24, 16: 32},
        's': {4: 6, 6: 10, 8: 14, 12: 20, 16: 28},
        'm': {4: 4, 6: 8, 8: 12, 12: 16, 16: 24},
        'l': {4: 2, 6: 4, 8: 8, 12: 12, 16: 16},
        'x': {4: 1, 6: 2, 8: 4, 12: 8, 16: 12}
    }
    
    # Find the closest memory size
    memory_keys = sorted(batch_sizes[model_size].keys())
    closest_memory = min(memory_keys, key=lambda x: abs(x - gpu_memory_gb))
    
    optimal_batch = batch_sizes[model_size][closest_memory]
    
    # For RTX 4050, be extra conservative
    if gpu_memory_gb <= 6:
        optimal_batch = max(4, optimal_batch - 2)
    
    logging.info(f"Optimal batch size for {model_size} model with {gpu_memory_gb:.1f}GB GPU: {optimal_batch}")
    
    return optimal_batch

def create_optimized_config(path: Path = DEFAULT_CONFIG_PATH):
    """Create optimized training configuration for wildlife detection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check system resources
    gpu_available = check_system_resources()
    
    # Calculate optimal batch size
    optimal_batch = calculate_optimal_batch_size('s')
    
    # Optimized configuration for wildlife detection
    config = {
        'model': {
            'name': 'yolov8s.pt',  # Start with small model for RTX 4050
            'pretrained': True
        },
        'data': {
            'yaml_path': 'dataset/data.yaml',
            'epochs': 100,  # Reduced for faster iteration
            'batch_size': optimal_batch,
            'imgsz': 640,
            'device': '0' if gpu_available else 'cpu'
        },
        'training': {
            'patience': 20,  # Reduced patience for faster convergence
            'save_period': 10,
            'cache': False,  # Disable cache to save memory
            'workers': min(4, psutil.cpu_count()),  # Reduced workers
            'project': 'results',
            'name': 'wildlife_detection_v2',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',  # Better for small datasets
            'verbose': True,
            'seed': 42,
            'deterministic': False,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # Mixed precision for memory efficiency
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'lr0': 0.0005,  # Conservative learning rate
            'lrf': 0.0001,  # Conservative final learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,  # Moderate weight decay
            'warmup_epochs': 3.0,  # Shorter warmup
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.01,
            'box': 0.7,  # Reduced box loss weight
            'cls': 0.3,  # Reduced classification loss weight
            'dfl': 1.5,  # Increased DFL loss weight
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'local_rank': -1,
            'quad': False,
            'linear_lr': False,
            'dataloader': 'auto',
            'upload_dataset': False,
            'bbox_interval': -1,
            'artifact_alias': 'latest',
            'export_after_train': True,
            'export_formats': ['onnx', 'torchscript']
        }
    }
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logging.info(f"Optimized training configuration saved to {path}")
    return path

def load_config(path: Path):
    """Load training configuration from YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def setup_gpu_optimization():
    """Setup GPU optimization for RTX 4050."""
    if torch.cuda.is_available():
        # Memory optimization
        torch.cuda.empty_cache()
        gc.collect()
        
        # Performance optimization
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory fraction (use 90% of available GPU memory)
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        logging.info("GPU optimization settings applied")
        return True
    return False

def monitor_gpu_memory():
    """Monitor GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logging.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        return allocated, reserved, total
    return None, None, None

def train_wildlife_model(cfg, model_size_override=None, batch_override=None, epochs_override=None, 
                        resume_from=None, no_export=False):
    """Train wildlife detection model with optimized settings."""
    
    # Setup GPU optimization
    gpu_available = setup_gpu_optimization()
    
    # Model selection
    model_name = cfg.get("model", {}).get("name", "yolov8s.pt")
    if model_size_override:
        model_name = f"yolov8{model_size_override}.pt"
        logging.info(f"Model override: {model_name}")
    
    # Auto-detect optimal batch size
    if batch_override is None and gpu_available:
        model_size = model_size_override or 's'
        optimal_batch = calculate_optimal_batch_size(model_size)
        cfg['data']['batch_size'] = optimal_batch
        logging.info(f"Auto-detected batch size: {optimal_batch}")
    
    # Training parameters
    batch = int(batch_override) if batch_override else int(cfg.get("data", {}).get("batch_size", 8))
    epochs = int(epochs_override) if epochs_override else int(cfg.get("data", {}).get("epochs", 100))
    imgsz = int(cfg.get("data", {}).get("imgsz", 640))
    data_yaml = cfg.get("data", {}).get("yaml_path", "dataset/data.yaml")
    
    # Training settings
    tcfg = cfg.get("training", {})
    project = tcfg.get("project", "results")
    name = tcfg.get("name", "wildlife_detection_v2")
    workers = int(tcfg.get("workers", 4))
    exist_ok = bool(tcfg.get("exist_ok", True))
    cache = bool(tcfg.get("cache", False))
    optimizer = tcfg.get("optimizer", "AdamW")
    lr0 = float(tcfg.get("lr0", 0.0005))
    lrf = float(tcfg.get("lrf", 0.0001))
    weight_decay = float(tcfg.get("weight_decay", 0.0005))
    patience = int(tcfg.get("patience", 20))
    save_period = int(tcfg.get("save_period", 10))
    amp = bool(tcfg.get("amp", True))
    resume = bool(tcfg.get("resume", False)) or resume_from is not None
    rect = bool(tcfg.get("rect", False))
    single_cls = bool(tcfg.get("single_cls", False))
    cos_lr = bool(tcfg.get("cos_lr", True))
    val = bool(tcfg.get("val", True))
    plots = bool(tcfg.get("plots", True))
    export_after_train = bool(tcfg.get("export_after_train", True)) and not no_export
    export_formats = tcfg.get("export_formats", ["onnx", "torchscript"])
    
    # Advanced parameters
    close_mosaic = int(tcfg.get("close_mosaic", 10))
    warmup_epochs = float(tcfg.get("warmup_epochs", 3.0))
    warmup_momentum = float(tcfg.get("warmup_momentum", 0.8))
    warmup_bias_lr = float(tcfg.get("warmup_bias_lr", 0.01))
    label_smoothing = float(tcfg.get("label_smoothing", 0.0))
    overlap_mask = bool(tcfg.get("overlap_mask", True))
    mask_ratio = int(tcfg.get("mask_ratio", 4))
    dropout = float(tcfg.get("dropout", 0.0))
    
    logging.info("=== Wildlife Detection Training Configuration ===")
    logging.info(f"Model: {model_name}")
    logging.info(f"Data: {data_yaml}")
    logging.info(f"Epochs: {epochs}, Batch: {batch}, Image Size: {imgsz}")
    logging.info(f"Device: {'GPU' if gpu_available else 'CPU'}")
    logging.info(f"Optimizer: {optimizer}, Learning Rate: {lr0}")
    logging.info(f"AMP (Mixed Precision): {amp}")
    logging.info(f"Cache: {cache}, Workers: {workers}")
    logging.info(f"Patience: {patience}, Save Period: {save_period}")
    
    # Monitor initial GPU memory
    monitor_gpu_memory()
    
    # Load model
    logging.info("Loading model...")
    if resume_from and Path(resume_from).exists():
        model = YOLO(resume_from)
        logging.info(f"Resuming from: {resume_from}")
    else:
        model = YOLO(model_name)
        logging.info(f"Starting with pretrained model: {model_name}")
    
    # Prepare training arguments
    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        project=project,
        name=name,
        exist_ok=exist_ok,
        pretrained=True,
        cache=cache,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        patience=patience,
        save_period=save_period,
        amp=amp,
        resume=resume,
        rect=rect,
        single_cls=single_cls,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        label_smoothing=label_smoothing,
        overlap_mask=overlap_mask,
        mask_ratio=mask_ratio,
        dropout=dropout,
        val=val,
        plots=plots,
        verbose=True,
        seed=42,
        deterministic=False
    )
    
    # Start training
    start = time()
    logging.info("Starting wildlife detection training...")
    
    try:
        results = model.train(**train_kwargs)
        
        # Monitor final GPU memory
        monitor_gpu_memory()
        
    except RuntimeError as e:
        logging.exception("RuntimeError during training.")
        if "out of memory" in str(e).lower():
            logging.error("Out of memory error. Recommendations:")
            logging.error("1. Reduce batch size (try --batch-size 4)")
            logging.error("2. Use smaller model (try --model-size n)")
            logging.error("3. Reduce image size (try --imgsz 512)")
        raise
    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise
    
    elapsed = time() - start
    logging.info(f"Training completed in {elapsed/60:.2f} minutes ({elapsed/3600:.2f} hours).")
    
    try:
        save_dir = Path(results.save_dir)
    except Exception:
        save_dir = Path(project) / name
    
    logging.info(f"Results saved to: {save_dir}")
    
    # Export model if requested
    if export_after_train:
        logging.info("Exporting trained model...")
        for fmt in export_formats:
            try:
                model.export(format=fmt)
                logging.info(f"Export {fmt} completed.")
            except Exception as ex:
                logging.exception(f"Export to {fmt} failed: {ex}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Optimized YOLOv8 training for wildlife detection")
    parser.add_argument("--create-config", action="store_true", help="Create optimized config and exit")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--model-size", choices=['n','s','m','l','x'], default=None, help="Override model size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-export", action="store_true", help="Disable export after training")
    args = parser.parse_args()
    
    logging.info("=== Wildlife Detection Training ===")
    
    if args.create_config:
        create_optimized_config()
        logging.info("Optimized config created. Edit config/wildlife_training_config.yaml if needed and re-run.")
        return
    
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logging.warning(f"Config {cfg_path} not found — creating optimized config.")
        create_optimized_config(cfg_path)
    
    cfg = load_config(cfg_path)
    
    try:
        results = train_wildlife_model(
            cfg, 
            model_size_override=args.model_size, 
            batch_override=args.batch_size,
            epochs_override=args.epochs, 
            resume_from=args.resume,
            no_export=args.no_export
        )
        logging.info("Wildlife detection training finished successfully!")
        
        # Show best model path
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            logging.info(f"Best model: {best_model_path}")
            logging.info(f"Model size: {best_model_path.stat().st_size / 1024**2:.1f} MB")
            logging.info("\nNext steps:")
            logging.info(f"1. Test detection: python scripts/detect_wildlife.py --model {best_model_path} --input test_image.jpg")
            logging.info(f"2. Process videos: python scripts/detect_wildlife.py --model {best_model_path} --input video.mp4")
        
    except Exception as e:
        logging.exception(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
