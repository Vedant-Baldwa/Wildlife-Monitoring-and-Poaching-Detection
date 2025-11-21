#!/usr/bin/env python3
"""
detect.py
Unified CLI for image / video / batch inference & annotation using Ultralytics YOLOv8.

Usage examples:
  python3 detect.py --mode image --input /path/img.jpg --output /path/out.jpg
  python3 detect.py --mode video --input /path/in.mp4 --output /path/out_dir --fast
  python3 detect.py --mode batch --input /path/images_dir --output /path/out_dir --save-csv

Defaults:
  classes file default: /mnt/data/classes.txt  (uploaded in your environment)
"""

import argparse
from pathlib import Path
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import time
from typing import List, Tuple

# ---------------------------
# Utilities: drawing & IO
# ---------------------------
def load_class_names(classes_path: str) -> List[str]:
    p = Path(classes_path)
    if not p.exists():
        raise FileNotFoundError(f"classes.txt not found at {p} (pass --classes to override)")
    names = []
    with open(p, "r", encoding="utf8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(None, 1)
            if len(parts) == 1:
                names.append(parts[0])
            else:
                if parts[0].isdigit():
                    names.append(parts[1].strip())
                else:
                    names.append(s)
    return names

def draw_boxes_cv2(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes_idx: np.ndarray,
                   class_names: List[str], color=(255, 180, 0), thickness=2, font_scale=0.6) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes_idx):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        label = f"{class_names[int(cls)]} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # draw filled rectangle as label background
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, y0), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out

def make_output_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------------------
# Core processing functions
# ---------------------------
def run_image(model: YOLO, img_path: Path, out_path: Path, class_names: List[str],
              conf: float=0.25, imgsz: int=1024, device='0', save_csv: bool=False):
    res = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, device=device, save=False, verbose=False)
    r = res[0]
    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.array([])
    scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.array([])
    cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.array([])

    img = cv2.imread(str(img_path))
    annotated = draw_boxes_cv2(img, boxes, scores, cls_idx, class_names)
    # write annotated image
    out_path_parent = out_path.parent
    if out_path_parent:
        make_output_dir(out_path_parent)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image => {out_path}")

    if save_csv:
        rows = []
        for (x1, y1, x2, y2), sc, cl in zip(boxes, scores, cls_idx):
            rows.append({"image": img_path.name, "x1": float(x1), "y1": float(y1),
                         "x2": float(x2), "y2": float(y2), "conf": float(sc),
                         "cls_idx": int(cl), "cls_name": class_names[int(cl)]})
        if rows:
            csv_out = out_path.with_suffix(".csv")
            pd.DataFrame(rows).to_csv(csv_out, index=False)
            print("Saved detections CSV =>", csv_out)

def run_batch(model: YOLO, imgs_dir: Path, out_dir: Path, class_names: List[str],
              conf: float=0.25, imgsz: int=1024, device='0', save_csv: bool=False):
    make_output_dir(out_dir)
    img_paths = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"Found {len(img_paths)} images in {imgs_dir}")
    all_rows = []
    for i, p in enumerate(img_paths):
        res = model.predict(source=str(p), conf=conf, imgsz=imgsz, device=device, save=False, verbose=False)
        r = res[0]
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.array([])
        scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.array([])
        cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.array([])

        img = cv2.imread(str(p))
        annotated = draw_boxes_cv2(img, boxes, scores, cls_idx, class_names)
        outp = out_dir / p.name
        cv2.imwrite(str(outp), annotated)
        if (i + 1) % 50 == 0 or (i+1)==len(img_paths):
            print(f"Processed {i+1}/{len(img_paths)}")
        for (x1, y1, x2, y2), sc, cl in zip(boxes, scores, cls_idx):
            all_rows.append({"image": p.name, "x1": float(x1), "y1": float(y1),
                             "x2": float(x2), "y2": float(y2), "conf": float(sc),
                             "cls_idx": int(cl), "cls_name": class_names[int(cl)]})
    if save_csv:
        csv_out = out_dir / "detections.csv"
        pd.DataFrame(all_rows).to_csv(csv_out, index=False)
        print("Saved combined CSV =>", csv_out)

def run_video_fast(model: YOLO, video_in: Path, out_dir: Path, class_names: List[str],
                   conf: float=0.25, imgsz: int=1024, device='0'):
    """
    Use ultralytics built-in fast video annotate (save=True). This is typically faster.
    """
    make_output_dir(out_dir)
    # model.predict with save=True saves annotated video inside save_dir
    results = model.predict(source=str(video_in), conf=conf, imgsz=imgsz, device=device, save=True, save_dir=str(out_dir))
    print("Ultralytics saved annotated outputs to:", out_dir)
    return results

def run_video_framebyframe(model: YOLO, video_in: Path, video_out: Path, class_names: List[str],
                           conf: float=0.25, imgsz: int=1024, device='0'):
    make_output_dir(video_out.parent)
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # model.predict accepts numpy arrays
        results = model.predict(source=frame, conf=conf, imgsz=imgsz, device=device, save=False, verbose=False)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.array([])
        scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.array([])
        cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.array([])

        annotated_frame = draw_boxes_cv2(frame, boxes, scores, cls_idx, class_names)
        writer.write(annotated_frame)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"Processed frames: {frame_idx}")
    writer.release()
    cap.release()
    t1 = time.time()
    print(f"Video written to {video_out} (frames: {frame_idx}, time: {t1-t0:.1f}s)")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 detection CLI - image/video/batch")
    p.add_argument("--mode", type=str, choices=["image", "video", "batch"], required=True, help="Operation mode")
    p.add_argument("--input", type=str, required=True, help="Input path (file for image/video, dir for batch)")
    p.add_argument("--output", type=str, required=True, help="Output path (file or directory). For video, pass output directory for fast mode or file path for frame mode.")
    p.add_argument("--weights", type=str, default=r"C:\Users\vedan\Machine Learning\SWE PROJECT\new_after_midsem\first_run\weights\best.pt", help="Path to weights (best.pt / last.pt). If not provided will fall back to 'yolov8x.pt'")
    p.add_argument("--classes", type=str, default=r"C:\Users\vedan\Machine Learning\SWE PROJECT\dataset\classes.txt", help="Path to classes.txt (default is uploaded /mnt/data/classes.txt)")
    p.add_argument("--imgsz", type=int, default=1024, help="Inference image size (px)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--device", type=str, default="0", help="Device id for Ultralytics (e.g. '0' or 'cpu')")
    p.add_argument("--save-csv", action="store_true", help="Save detections as CSV (image -> .csv, batch -> combined csv)")
    p.add_argument("--fast", action="store_true", help="For video: use ultralytics fast save (save=True) instead of frame-by-frame")
    return p.parse_args()
def main():
    args = parse_args()
    mode = args.mode
    inp = Path(args.input)
    out = Path(args.output)
    imgsz = args.imgsz
    conf = args.conf
    device = args.device

    # load classes
    class_names = load_class_names(args.classes)
    print(f"Loaded {len(class_names)} classes. Sample: {class_names[:5]}")

    # resolve weights default: try to find checkpoint in output folder (useful) else fallback
    if args.weights:
        weights_path = args.weights
    else:
        # try to detect a local runs/train/*/weights/best.pt or last.pt
        local_ckpt = None
        runs_root = Path("runs") / "train"
        if runs_root.exists():
            for r in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                candidate_best = r / "weights" / "best.pt"
                candidate_last = r / "weights" / "last.pt"
                if candidate_best.exists():
                    local_ckpt = candidate_best
                    break
                if candidate_last.exists():
                    local_ckpt = candidate_last
                    break
        if local_ckpt:
            weights_path = str(local_ckpt)
            print("Auto-selected weights:", weights_path)
        else:
            weights_path = "yolov8x.pt"
            print("No local checkpoint found, falling back to hub weights:", weights_path)

    # set CUDA visible devices if user passed numeric device id (so OpenCV/GPU mapping consistent)
    try:
        if device != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    except Exception:
        pass

    # init model
    print("Loading model:", weights_path)
    model = YOLO(weights_path)

    # run according to mode
    if mode == "image":
        if not inp.exists() or not inp.is_file():
            raise SystemExit(f"Input image not found: {inp}")
        # determine output file path
        if out.is_dir():
            out_path = out / inp.name
        else:
            out_path = out
        run_image(model, inp, out_path, class_names, conf=conf, imgsz=imgsz, device=device, save_csv=args.save_csv)

    elif mode == "batch":
        if not inp.exists() or not inp.is_dir():
            raise SystemExit(f"Input must be a directory for batch mode: {inp}")
        make_output_dir(out)
        run_batch(model, inp, out, class_names, conf=conf, imgsz=imgsz, device=device, save_csv=args.save_csv)

    elif mode == "video":
        if not inp.exists() or not inp.is_file():
            raise SystemExit(f"Input video not found: {inp}")
        if args.fast:
            make_output_dir(out)
            run_video_fast(model, inp, out, class_names, conf=conf, imgsz=imgsz, device=device)
        else:
            # output should be a file path for annotated video
            if out.is_dir():
                # create file in this dir with same base name
                out_file = out / (inp.stem + "_annotated.mp4")
            else:
                out_file = out
            run_video_framebyframe(model, inp, out_file, class_names, conf=conf, imgsz=imgsz, device=device)
    else:
        raise SystemExit("Unknown mode")

if __name__ == "__main__":
    main()
