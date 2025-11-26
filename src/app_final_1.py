# app_final_1.py — Minimal / fixed version of your app (based on /mnt/data/app2.py)
# Fixes:
#  - prevents IndexError when class id is outside class-names list (safe label lookup)
#  - robust handling of OBB results (extract_obb) and axis-aligned fallback
#  - tolerant draw_boxes implementation (pads/truncates arrays)
#
# Save this as app_final_1.py and run with `streamlit run app_final_1.py`.

import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile, os, time, subprocess, shutil
import cv2
import numpy as np
import pandas as pd
from typing import List, Optional
import base64
import mimetypes

# -------------------------
# Config / Defaults
# -------------------------
DEFAULT_SAMPLE_FILE = r"C:\Users\vedan\Machine Learning\SWE PROJECT\poaching\dataset\test\images\9c8a8884-5-2f1c_jpg.rf.834654bc5d9407eb11056f0ca121fd50.jpg"

# -------------------------
# Helpers (kept from original app + fixes)
# -------------------------
def display_video_robust(holder, video_path: Path, text_holder=None):
    video_path = Path(video_path)
    if text_holder is None:
        text_holder = holder

    if not video_path.exists():
        text_holder.error(f"Video not found: {video_path}")
        return

    size_mb = video_path.stat().st_size / (1024*1024)
    text_holder.info(f"Found annotated video: {video_path} ({size_mb:.2f} MB)")

    # 1) try st.video with path (fast)
    try:
        holder.video(str(video_path))
        text_holder.success("Displayed video via st.video(path).")
        return
    except Exception as e:
        text_holder.warning(f"st.video(path) failed: {e}")

    # 2) try reading bytes and passing to st.video
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        holder.video(video_bytes)
        text_holder.success("Displayed video via st.video(bytes).")
        return
    except Exception as e:
        text_holder.warning(f"st.video(bytes) failed: {e}")

    # 3) fallback: embed HTML5 video with base64
    try:
        mime, _ = mimetypes.guess_type(str(video_path))
        if mime is None:
            mime = "video/mp4"
        b64 = base64.b64encode(video_bytes).decode()
        html = f"""
        <video controls width="800">
          <source src="data:{mime};base64,{b64}" type="{mime}">
          Your browser does not support the video tag.
        </video>
        """
        holder.markdown(html, unsafe_allow_html=True)
        text_holder.success("Displayed video via HTML5 <video> with base64 payload.")
        return
    except Exception as e:
        text_holder.error(f"All display methods failed: {e}")
        return

def load_class_names(classes_path: str) -> List[str]:
    p = Path(classes_path)
    if not p.exists():
        # keep silent here (sidebar will show warning). return empty list.
        return []
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

def safe_get_label(class_names, cls):
    """
    Safe lookup: return class name if available, otherwise the numeric id as string.
    """
    try:
        idx = int(cls)
    except Exception:
        return str(cls)
    if class_names and isinstance(class_names, (list, tuple)) and 0 <= idx < len(class_names):
        return class_names[idx]
    return str(idx)

def draw_boxes_cv2(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes_idx: np.ndarray, class_names: List[str]):
    """
    Draw axis-aligned boxes on BGR image.
    This function is defensive: pads/truncates scores & classes to match boxes length and uses safe label lookup.
    """
    out = img.copy()
    if boxes is None or len(boxes) == 0:
        return out

    n = boxes.shape[0]

    # normalize scores and classes arrays
    try:
        scores = np.array(scores).flatten()
    except Exception:
        scores = np.zeros((0,))
    try:
        classes_idx = np.array(classes_idx).flatten().astype(int)
    except Exception:
        classes_idx = np.zeros((0,), dtype=int)

    # pad/truncate to length n
    if scores.size < n:
        scores = np.pad(scores, (0, n - scores.size), mode="constant", constant_values=0.0)
    else:
        scores = scores[:n]

    if classes_idx.size < n:
        classes_idx = np.pad(classes_idx, (0, n - classes_idx.size), mode="constant", constant_values=0)
    else:
        classes_idx = classes_idx[:n]

    font_scale = max(0.4, min(1.0, img.shape[1] / 1000.0))

    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes_idx):
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        color = (255, 180, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        label = f"{safe_get_label(class_names, cls)} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, y0), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out

def auto_select_weights(runs_root=Path("runs") / "train") -> Optional[str]:
    if not runs_root.exists():
        return None
    for r in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        best = r / "weights" / "best.pt"
        last = r / "weights" / "last.pt"
        if best.exists():
            return str(best)
        if last.exists():
            return str(last)
    return None

def list_out_dir(out_dir: Path, limit: int = 200):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    files = sorted(out_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    info = []
    for p in files:
        try:
            info.append((p.relative_to(out_dir).as_posix(), p.suffix, p.stat().st_size))
        except Exception:
            info.append((str(p), p.suffix, None))
    return info

def ffmpeg_assemble_frames(frames_dir: Path, out_file: Path, fps: int = 25) -> Optional[Path]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-pattern_type", "glob",
            "-i", str(frames_dir / "*.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_file)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_file
    except Exception:
        return None

def python_assemble_frames(frames_dir: Path, out_file: Path, fps: int = 25) -> Optional[Path]:
    imgs = sorted([p for p in Path(frames_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not imgs:
        return None
    first = cv2.imread(str(imgs[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    return out_file

def ffmpeg_convert_avi_to_mp4(avi_path: Path, mp4_path: Path) -> Optional[Path]:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(avi_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(mp4_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp4_path
    except Exception:
        return None

def python_convert_avi_to_mp4(avi_path: Path, mp4_path: Path) -> Optional[Path]:
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()
    cap.release()
    return mp4_path

# -------------------------
# Minimal OBB helpers (NEW)
# -------------------------
def extract_obb(r):
    """Extract OBB outputs from Ultralytics OBB result 'r'."""
    if r is None or not hasattr(r, "obb") or r.obb is None:
        return np.zeros((0,4)), np.zeros((0,5)), np.zeros((0,)), np.zeros((0,), dtype=int)
    obb = r.obb
    # axis-aligned xyxy
    try:
        xyxy = obb.xyxy.cpu().numpy()
    except Exception:
        xyxy = np.zeros((0,4))
    # rotated xywha
    try:
        xywha = obb.xywha.cpu().numpy()
    except Exception:
        xywha = np.zeros((0,5))
    # scores & classes
    try:
        scores = obb.conf.cpu().numpy()
    except Exception:
        scores = np.zeros((0,))
    try:
        classes = obb.cls.cpu().numpy().astype(int)
    except Exception:
        classes = np.zeros((0,), dtype=int)
    return xyxy, xywha, scores, classes

def draw_rotated_obb(img_bgr, xywha, scores, classes, class_names=None):
    out = img_bgr.copy()
    if xywha is None or len(xywha) == 0:
        return out
    font_scale = max(0.4, min(1.0, img_bgr.shape[1] / 1000.0))
    # ensure arrays shape safe
    n = xywha.shape[0]
    try:
        scores = np.array(scores).flatten()
    except Exception:
        scores = np.zeros((n,))
    try:
        classes = np.array(classes).flatten().astype(int)
    except Exception:
        classes = np.zeros((n,), dtype=int)
    if scores.size < n:
        scores = np.pad(scores, (0, n - scores.size), mode="constant", constant_values=0.0)
    else:
        scores = scores[:n]
    if classes.size < n:
        classes = np.pad(classes, (0, n - classes.size), mode="constant", constant_values=0)
    else:
        classes = classes[:n]
    for (cx, cy, w, h, a), sc, cl in zip(xywha, scores, classes):
        angle = float(a)
        # handle radian vs degree heuristics: if angle small assume radians and convert
        if abs(angle) <= 2 * np.pi and abs(angle) < 10:
            angle_deg = np.degrees(angle)
        else:
            angle_deg = angle
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
        pts = cv2.boxPoints(rect).astype(int)
        color = (0, 255, 0)
        cv2.polylines(out, [pts], True, color, 2, lineType=cv2.LINE_AA)
        # place label at top-left of box
        tlx, tly = pts.min(axis=0)
        label = f"{safe_get_label(class_names, cl)} {float(sc):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y0 = max(0, tly - th - 6)
        cv2.rectangle(out, (tlx, y0), (tlx + tw, tly), color, -1)
        cv2.putText(out, label, (tlx, tly - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out

# -------------------------
# Cached model loader (works for multiple weights paths separately)
# -------------------------
@st.cache_resource
def get_model(weights_path: str):
    return YOLO(weights_path)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Wildlife & Poaching Detection", layout="wide")
st.title("Wildlife & Poaching Detection (YOLOv8) — Image / Video / Batch")

# Sidebar controls for Wildlife model
st.sidebar.header("Wildlife Model Settings")
classes_default = r"C:\Users\vedan\Machine Learning\SWE PROJECT\dataset_2\classes.txt"
classes_path = st.sidebar.text_input("Wildlife classes.txt path", classes_default)
names = load_class_names(classes_path)
st.sidebar.write(f"Loaded {len(names)} wildlife classes." if names else "No wildlife classes loaded.")

weights_default = auto_select_weights() or r"C:\Users\vedan\Machine Learning\SWE PROJECT\new_after_midsem\african_wildlife_runs\detect\wildlife_yolov8x_african\weights\best.pt"
weights_path = st.sidebar.text_input("Wildlife Weights (best.pt / last.pt)", weights_default)
if not Path(weights_path).exists() and weights_path != "yolov8x.pt":
    st.sidebar.warning("Wildlife weights path doesn't exist. App will try to load fallback 'yolov8x.pt' from the hub (internet).")

# Sidebar controls for Poaching model (separate)
st.sidebar.markdown("---")
st.sidebar.header("Poaching Model Settings")
poach_classes_default = r"C:\Users\vedan\Machine Learning\SWE PROJECT\poaching\dataset\README.dataset.txt"
poach_classes_path = st.sidebar.text_input("Poaching classes.txt path", poach_classes_default)
poach_names = load_class_names(poach_classes_path)
st.sidebar.write(f"Loaded {len(poach_names)} poaching classes." if poach_names else "No poaching classes loaded.")

poach_weights_default = r"C:\Users\vedan\Machine Learning\SWE PROJECT\poaching\poacher_obb\exp1\weights\best.pt"  # leave blank by default
poach_weights_path = st.sidebar.text_input("Poaching Weights (best.pt / last.pt)", poach_weights_default)
if poach_weights_path and not Path(poach_weights_path).exists():
    st.sidebar.warning("Poaching weights path doesn't exist. Leave blank to skip loading.")

# Common controls
imgsz = st.sidebar.select_slider("Image size (px)", options=[384, 512, 640, 768, 1024], value=1024)
conf_th = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
device = st.sidebar.text_input("Device ('0' for GPU index 0 or 'cpu')", "0")
if device != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

st.sidebar.markdown("---")
st.sidebar.write("Models will be cached after first load (keeps GPU memory). Use 'Reload model' below to clear cache.")

if st.sidebar.button("Reload wildlife model"):
    get_model.clear()
    st.experimental_rerun()

# Load wildlife model
with st.spinner("Loading wildlife model..."):
    try:
        model = get_model(weights_path)
    except Exception as e:
        st.error(f"Failed to load wildlife model: {e}")
        st.stop()

# Load poaching model (optional)
poach_model = None
if poach_weights_path:
    with st.spinner("Loading poaching model..."):
        try:
            poach_model = get_model(poach_weights_path)
        except Exception as e:
            st.warning(f"Failed to load poaching model: {e}")
            poach_model = None
else:
    st.sidebar.info("No poaching weights provided. Poaching tab will still be shown but will error if you try to run inference.")

# Top-level tabs: Wildlife and Poaching
top_tab1, top_tab2 = st.tabs(["Wildlife Detection", "Poaching Detection"])

# -------------------------
# Wildlife Detection tab (original app flows)
# -------------------------
with top_tab1:
    st.header("Wildlife Detection")
    # reuse original Image/Video/Batch tabs inside Wildlife
    w_tab1, w_tab2, w_tab3 = st.tabs(["Image", "Video", "Batch"])

    # Image (wildlife)
    with w_tab1:
        st.subheader("Image inference — Wildlife")
        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded_img = st.file_uploader("Upload image (wildlife)", type=["jpg", "jpeg", "png"], key="wild_img_upl")
            url_input = st.text_input("Or paste image URL or local file URL (file://...)", DEFAULT_SAMPLE_FILE, key="wild_img_url")
            run_button = st.button("Run wildlife inference on image", key="wild_img_btn")
        with col2:
            st.write("Result")
            placeholder = st.empty()

        if run_button:
            if uploaded_img is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_img.name).suffix)
                tfile.write(uploaded_img.getvalue()); tfile.flush(); tfile.close()
                img_path = tfile.name
            elif url_input:
                if url_input.startswith("file://"):
                    img_path = url_input.replace("file://", "")
                else:
                    import requests
                    try:
                        r = requests.get(url_input, timeout=10)
                        if r.status_code == 200:
                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            tfile.write(r.content); tfile.flush(); tfile.close()
                            img_path = tfile.name
                        else:
                            st.error("Could not download URL: status=" + str(r.status_code)); img_path = None
                    except Exception as e:
                        st.error(f"Failed to download URL: {e}"); img_path = None
            else:
                st.error("Provide an uploaded file or a URL"); img_path = None

            if img_path:
                try:
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        st.error("Failed to open image. Check the path/URL.")
                    else:
                        with st.spinner("Running wildlife model..."):
                            res = model.predict(source=img_bgr, conf=conf_th, imgsz=imgsz, device=device, save=False, verbose=False)
                            r = res[0]
                            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.array([])
                            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.array([])
                            cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.array([])
                            annotated = draw_boxes_cv2(img_bgr, boxes, scores, cls_idx, names)
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            placeholder.image(annotated_rgb, use_container_width=True)
                            df = pd.DataFrame([{"class_idx": int(c), "class": (names[int(c)] if names and int(c) < len(names) else int(c)), "conf": float(sc),
                                                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                                               for (x1, y1, x2, y2), sc, c in zip(boxes, scores, cls_idx)])
                            if not df.empty:
                                st.dataframe(df)
                            else:
                                st.info("No wildlife detections above threshold.")
                finally:
                    if uploaded_img is not None:
                        try:
                            os.unlink(tfile.name)
                        except:
                            pass

    # Video (wildlife)
    with w_tab2:
        st.subheader("Video inference — Wildlife")
        colv1, colv2 = st.columns([1, 2])
        with colv1:
            uploaded_vid = st.file_uploader("Upload wildlife video (mp4, avi, mov)", type=["mp4", "avi", "mov"], key="wild_vid_upl")
            vid_url = st.text_input("Or paste local path / URL to video (file:///...)", "", key="wild_vid_url")
            fast_mode = st.checkbox("Fast (Ultralytics save=True) - quicker", value=True, key="wild_fast")
            run_vid = st.button("Run wildlife inference on video", key="wild_vid_btn")
        with colv2:
            out_vid_holder = st.empty()
            out_text_holder = st.empty()

        if run_vid:
            # choose source
            if uploaded_vid is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_vid.name).suffix)
                tfile.write(uploaded_vid.getvalue()); tfile.flush(); tfile.close()
                video_src = tfile.name
            elif vid_url:
                if vid_url.startswith("file://"):
                    video_src = vid_url.replace("file://", "")
                else:
                    import requests
                    try:
                        r = requests.get(vid_url, stream=True, timeout=10)
                        if r.status_code == 200:
                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    tfile.write(chunk)
                            tfile.flush(); tfile.close()
                            video_src = tfile.name
                        else:
                            st.error("Could not download video: status=" + str(r.status_code)); video_src = None
                    except Exception as e:
                        st.error(f"Failed to download video: {e}"); video_src = None
            else:
                st.error("Provide a video upload or file:// path"); video_src = None

            if video_src:
                out_dir = Path(r"C:\Users\vedan\Machine Learning\SWE PROJECT\new_after_midsem\runs\detect\predict")
                try:
                    if fast_mode:
                        st.info("Running fast wildlife video annotate (Ultralytics). This may take time.")
                        with st.spinner("Annotating wildlife video..."):
                            _ = model.predict(
                                source=video_src,
                                conf=conf_th,
                                imgsz=imgsz,
                                device=device,
                                save=True,
                                save_dir=str(out_dir)
                            )
                        # find annotated video file
                        ann_vid = None
                        for f in sorted(out_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
                            if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                                ann_vid = f
                                break
                        if ann_vid:
                            if ann_vid.suffix.lower() == ".avi":
                                out_text_holder.info("Annotated .avi found — converting to mp4 for browser playback...")
                                mp4_out = ann_vid.with_name(ann_vid.stem + "_converted.mp4")
                                converted = ffmpeg_convert_avi_to_mp4(ann_vid, mp4_out)
                                if converted is None:
                                    converted = python_convert_avi_to_mp4(ann_vid, mp4_out)
                                if converted:
                                    st.success("Annotated wildlife video converted to MP4.")
                                    final_video_path = Path(converted)
                                    display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                    with open(converted, "rb") as fh:
                                        st.download_button("Download annotated wildlife video (mp4)", fh.read(), file_name=converted.name)
                                else:
                                    st.error("Failed to convert .avi to .mp4. Download the .avi to convert locally.")
                                    with open(ann_vid, "rb") as fh:
                                        st.download_button("Download annotated wildlife video (avi)", fh.read(), file_name=ann_vid.name)
                            else:
                                st.success("Wildlife video annotated successfully!")
                                final_video_path = Path(ann_vid)
                                display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                with open(ann_vid, "rb") as fh:
                                    st.download_button("Download annotated wildlife video", fh.read(), file_name=ann_vid.name)
                        else:
                            frames = sorted([p for p in out_dir.rglob("*.jpg")])
                            if frames:
                                st.info(f"Detected {len(frames)} annotated frames — attempting to assemble into mp4...")
                                assembled = None
                                out_mp4 = out_dir / (Path(video_src).stem + "_annotated_combined.mp4")
                                assembled = ffmpeg_assemble_frames(out_dir, out_mp4, fps=25)
                                if assembled is None:
                                    assembled = python_assemble_frames(out_dir, out_mp4, fps=25)
                                if assembled:
                                    st.success("Assembled annotated wildlife video.")
                                    final_video_path = Path(assembled)
                                    display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                    with open(assembled, "rb") as fh:
                                        st.download_button("Download annotated wildlife video", fh.read(), file_name=assembled.name)
                                else:
                                    st.error("Failed to assemble frames into a video. Try turning off Fast mode or install ffmpeg.")
                            else:
                                st.error("Annotated video not found and no frames detected. Check the out directory:")
                                files_info = list_out_dir(out_dir, limit=200)
                                if files_info:
                                    out_text_holder.text("\n".join([f"{p} | {s} | {size} bytes" for (p, s, size) in files_info[:100]]))
                                else:
                                    out_text_holder.text("No files written to output directory (out_dir may be empty).")
                    else:
                        st.info("Running slower frame-by-frame wildlife annotation (writing new video).")
                        t_out = out_dir / (Path(video_src).stem + "_annotated.mp4")
                        cap = cv2.VideoCapture(str(video_src))
                        if not cap.isOpened():
                            st.error("Cannot open video.")
                        else:
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(str(t_out), fourcc, fps, (w, h))
                            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                            pbar = st.progress(0)
                            i = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                res = model.predict(frame, conf=conf_th, imgsz=imgsz, device=device, save=False, verbose=False)
                                r = res[0]
                                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else []
                                scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else []
                                cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else []
                                annotated = draw_boxes_cv2(frame, boxes, scores, cls_idx, names)
                                writer.write(annotated)
                                i += 1
                                if frames > 0:
                                    pbar.progress(min(1.0, i / frames))
                            writer.release()
                            cap.release()
                            st.success("Wildlife video annotated successfully!")
                            final_video_path = Path(t_out)
                            display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                            with open(t_out, "rb") as fh:
                                st.download_button("Download annotated wildlife video (mp4)", fh.read(), file_name=t_out.name)
                finally:
                    if uploaded_vid is not None:
                        try:
                            os.unlink(tfile.name)
                        except Exception:
                            pass

    # Batch (wildlife)
    with w_tab3:
        st.subheader("Batch image folder inference — Wildlife")
        folder_path = st.text_input("Enter folder path on server (e.g. /dataset/test/images) or upload a zip", "", key="wild_batch_folder")
        uploaded_zip = st.file_uploader("Or upload a zip of images (optional)", type=["zip"], key="wild_batch_zip")
        run_batch_btn = st.button("Run wildlife batch inference", key="wild_batch_btn")
        if run_batch_btn:
            if uploaded_zip is not None:
                tmpdir = Path(tempfile.mkdtemp())
                zipf = tmpdir / "upload.zip"
                with open(zipf, "wb") as f:
                    f.write(uploaded_zip.getvalue())
                import zipfile
                with zipfile.ZipFile(zipf, "r") as z:
                    z.extractall(tmpdir / "images")
                imgs_dir = tmpdir / "images"
            else:
                imgs_dir = Path(folder_path)
            if not imgs_dir.exists() or not imgs_dir.is_dir():
                st.error("Input folder not found or invalid.")
            else:
                outdir = Path(tempfile.mkdtemp(prefix="yolo_batch_out_"))
                st.info(f"Running on {len(list(imgs_dir.glob('*.*')))} files. Outputs will be saved to server temp folder: {outdir}")
                pbar = st.progress(0)
                files = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                rows = []
                for i, p in enumerate(files):
                    res = model.predict(source=str(p), conf=conf_th, imgsz=imgsz, device=device, save=False, verbose=False)
                    r = res[0]
                    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else []
                    scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else []
                    cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else []
                    img = cv2.imread(str(p))
                    annotated = draw_boxes_cv2(img, boxes, scores, cls_idx, names)
                    outp = outdir / p.name
                    cv2.imwrite(str(outp), annotated)
                    for (x1, y1, x2, y2), sc, cl in zip(boxes, scores, cls_idx):
                        label = safe_get_label(names, cl)
                        try:
                            scf = float(sc)
                        except Exception:
                            scf = 0.0
                        rows.append({"image": p.name, "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": scf, "cls_idx": int(cl) if (isinstance(cl, (int, np.integer)) or str(cl).isdigit()) else None, "cls_name": label})
                    pbar.progress((i + 1) / len(files))
                df = pd.DataFrame(rows)
                csv_p = outdir / "detections.csv"
                df.to_csv(csv_p, index=False)
                st.success("Wildlife batch complete.")
                st.write("Annotated images saved to server temp folder:", outdir)
                st.download_button("Download detections CSV", csv_p.read_bytes(), file_name="detections.csv")

# -------------------------
# Poaching Detection tab (new)
# -------------------------
with top_tab2:
    st.header("Poaching Detection")
    p_tab1, p_tab2 = st.tabs(["Image", "Video"])

    # Poaching Image
    with p_tab1:
        st.subheader("Image inference — Poaching")
        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded_img = st.file_uploader("Upload image (poaching)", type=["jpg", "jpeg", "png"], key="poach_img_upl")
            url_input = st.text_input("Or paste image URL or local file URL (file://...)", DEFAULT_SAMPLE_FILE, key="poach_img_url")
            run_button = st.button("Run poaching inference on image", key="poach_img_btn")
        with col2:
            st.write("Result")
            placeholder = st.empty()

        if run_button:
            if poach_model is None:
                st.error("Poaching model not loaded. Set Poaching Weights in sidebar.")
            else:
                if uploaded_img is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_img.name).suffix)
                    tfile.write(uploaded_img.getvalue()); tfile.flush(); tfile.close()
                    img_path = tfile.name
                elif url_input:
                    if url_input.startswith("file://"):
                        img_path = url_input.replace("file://", "")
                    else:
                        import requests
                        try:
                            r = requests.get(url_input, timeout=10)
                            if r.status_code == 200:
                                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                                tfile.write(r.content); tfile.flush(); tfile.close()
                                img_path = tfile.name
                            else:
                                st.error("Could not download URL: status=" + str(r.status_code)); img_path = None
                        except Exception as e:
                            st.error(f"Failed to download URL: {e}"); img_path = None
                else:
                    st.error("Provide an uploaded file or a URL"); img_path = None

                if img_path:
                    try:
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is None:
                            st.error("Failed to open image. Check the path/URL.")
                        else:
                            with st.spinner("Running poaching model..."):
                                # convert to RGB for predict (ultralytics usually expects RGB)
                                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                                res = poach_model.predict(source=img_rgb, conf=conf_th, imgsz=imgsz, device=device, save=False, verbose=False)
                                r = res[0]
                                xyxy, xywha, scores, classes = extract_obb(r)
                                annotated = img_bgr.copy()
                                rows = []
                                if xywha is not None and xywha.shape[0] > 0:
                                    annotated = draw_rotated_obb(annotated, xywha, scores, classes, poach_names)
                                    for (cx, cy, w, h, a), sc, cl in zip(xywha, scores, classes):
                                        label = safe_get_label(poach_names, cl)
                                        try:
                                            scf = float(sc)
                                        except Exception:
                                            scf = 0.0
                                        x1 = float(cx) - float(w) / 2.0
                                        y1 = float(cy) - float(h) / 2.0
                                        x2 = float(cx) + float(w) / 2.0
                                        y2 = float(cy) + float(h) / 2.0
                                        rows.append({"class_idx": int(cl) if (isinstance(cl, (int, np.integer)) or str(cl).isdigit()) else None,
                                                     "class": label,
                                                     "conf": scf,
                                                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                                elif xyxy is not None and xyxy.shape[0] > 0:
                                    annotated = draw_boxes_cv2(annotated, xyxy, scores, classes, poach_names)
                                    for (x1, y1, x2, y2), sc, cl in zip(xyxy, scores, classes):
                                        label = safe_get_label(poach_names, cl)
                                        try:
                                            scf = float(sc)
                                        except Exception:
                                            scf = 0.0
                                        rows.append({"class_idx": int(cl) if (isinstance(cl, (int, np.integer)) or str(cl).isdigit()) else None,
                                                     "class": label,
                                                     "conf": scf,
                                                     "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
                                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                placeholder.image(annotated_rgb, use_container_width=True)
                                df = pd.DataFrame(rows)
                                if not df.empty:
                                    st.dataframe(df)
                                else:
                                    st.info("No poaching detections above threshold.")
                    finally:
                        if uploaded_img is not None:
                            try:
                                os.unlink(tfile.name)
                            except:
                                pass

    # Poaching Video
    with p_tab2:
        st.subheader("Video inference — Poaching")
        colv1, colv2 = st.columns([1, 2])
        with colv1:
            uploaded_vid = st.file_uploader("Upload poaching video (mp4, avi, mov)", type=["mp4", "avi", "mov"], key="poach_vid_upl")
            vid_url = st.text_input("Or paste local path / URL to video (file:///...)", "", key="poach_vid_url")
            fast_mode = st.checkbox("Fast (Ultralytics save=True) - quicker", value=True, key="poach_fast")
            run_vid = st.button("Run poaching inference on video", key="poach_vid_btn")
        with colv2:
            out_vid_holder = st.empty()
            out_text_holder = st.empty()

        if run_vid:
            if poach_model is None:
                st.error("Poaching model not loaded. Set Poaching Weights in sidebar.")
            else:
                if uploaded_vid is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_vid.name).suffix)
                    tfile.write(uploaded_vid.getvalue()); tfile.flush(); tfile.close()
                    video_src = tfile.name
                elif vid_url:
                    if vid_url.startswith("file://"):
                        video_src = vid_url.replace("file://", "")
                    else:
                        import requests
                        try:
                            r = requests.get(vid_url, stream=True, timeout=10)
                            if r.status_code == 200:
                                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:
                                        tfile.write(chunk)
                                tfile.flush(); tfile.close()
                                video_src = tfile.name
                            else:
                                st.error("Could not download video: status=" + str(r.status_code)); video_src = None
                        except Exception as e:
                            st.error(f"Failed to download video: {e}"); video_src = None
                else:
                    st.error("Provide a video upload or file:// path"); video_src = None

                if video_src:
                    out_dir = Path(tempfile.mkdtemp(prefix="poach_out_"))
                    try:
                        if fast_mode:
                            st.info("Running fast poaching video annotate (Ultralytics). This may take time.")
                            with st.spinner("Annotating poaching video..."):
                                _ = poach_model.predict(
                                    source=video_src,
                                    conf=conf_th,
                                    imgsz=imgsz,
                                    device=device,
                                    save=True,
                                    save_dir=str(out_dir)
                                )
                            ann_vid = None
                            for f in sorted(out_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
                                if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                                    ann_vid = f
                                    break
                            if ann_vid:
                                if ann_vid.suffix.lower() == ".avi":
                                    out_text_holder.info("Annotated .avi found — converting to mp4 for browser playback...")
                                    mp4_out = ann_vid.with_name(ann_vid.stem + "_converted.mp4")
                                    converted = ffmpeg_convert_avi_to_mp4(ann_vid, mp4_out)
                                    if converted is None:
                                        converted = python_convert_avi_to_mp4(ann_vid, mp4_out)
                                    if converted:
                                        st.success("Annotated poaching video converted to MP4.")
                                        final_video_path = Path(converted)
                                        display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                        with open(converted, "rb") as fh:
                                            st.download_button("Download annotated poaching video (mp4)", fh.read(), file_name=converted.name)
                                    else:
                                        st.error("Failed to convert .avi to .mp4. Download the .avi to convert locally.")
                                        with open(ann_vid, "rb") as fh:
                                            st.download_button("Download annotated poaching video (avi)", fh.read(), file_name=ann_vid.name)
                                else:
                                    st.success("Poaching video annotated successfully!")
                                    final_video_path = Path(ann_vid)
                                    display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                    with open(ann_vid, "rb") as fh:
                                        st.download_button("Download annotated poaching video", fh.read(), file_name=ann_vid.name)
                            else:
                                frames = sorted([p for p in out_dir.rglob("*.jpg")])
                                if frames:
                                    st.info(f"Detected {len(frames)} annotated frames — attempting to assemble into mp4...")
                                    assembled = None
                                    out_mp4 = out_dir / (Path(video_src).stem + "_annotated_combined.mp4")
                                    assembled = ffmpeg_assemble_frames(out_dir, out_mp4, fps=25)
                                    if assembled is None:
                                        assembled = python_assemble_frames(out_dir, out_mp4, fps=25)
                                    if assembled:
                                        st.success("Assembled annotated poaching video.")
                                        final_video_path = Path(assembled)
                                        display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                        with open(assembled, "rb") as fh:
                                            st.download_button("Download annotated poaching video", fh.read(), file_name=assembled.name)
                                    else:
                                        st.error("Failed to assemble frames into a video. Try turning off Fast mode or install ffmpeg.")
                                else:
                                    st.error("Annotated video not found and no frames detected. Check the out directory:")
                                    files_info = list_out_dir(out_dir, limit=200)
                                    if files_info:
                                        out_text_holder.text("\n".join([f"{p} | {s} | {size} bytes" for (p, s, size) in files_info[:100]]))
                                    else:
                                        out_text_holder.text("No files written to output directory (out_dir may be empty).")
                        else:
                            st.info("Running slower frame-by-frame poaching annotation (writing new video).")
                            t_out = out_dir / (Path(video_src).stem + "_annotated.mp4")
                            cap = cv2.VideoCapture(str(video_src))
                            if not cap.isOpened():
                                st.error("Cannot open video.")
                            else:
                                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(str(t_out), fourcc, fps, (w, h))
                                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                                pbar = st.progress(0)
                                i = 0
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    # convert to rgb for predict
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    res = poach_model.predict(frame_rgb, conf=conf_th, imgsz=imgsz, device=device, save=False, verbose=False)
                                    r = res[0]
                                    xyxy, xywha, scores, classes = extract_obb(r)
                                    annotated = frame.copy()
                                    if xywha is not None and xywha.shape[0] > 0:
                                        annotated = draw_rotated_obb(annotated, xywha, scores, classes, poach_names)
                                    elif xyxy is not None and xyxy.shape[0] > 0:
                                        annotated = draw_boxes_cv2(annotated, xyxy, scores, classes, poach_names)
                                    writer.write(annotated)
                                    i += 1
                                    if frames > 0:
                                        pbar.progress(min(1.0, i / frames))
                                writer.release()
                                cap.release()
                                st.success("Poaching video annotated successfully!")
                                final_video_path = Path(t_out)
                                display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                with open(t_out, "rb") as fh:
                                    st.download_button("Download annotated poaching video (mp4)", fh.read(), file_name=t_out.name)
                    finally:
                        if uploaded_vid is not None:
                            try:
                                os.unlink(tfile.name)
                            except Exception:
                                pass

st.markdown("---")
st.caption("Streamlit demo — now includes both Wildlife and Poaching detection tabs. Adapt class filters, add post-processing, or push detections to a DB as needed.")
