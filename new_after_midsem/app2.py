# app.py
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile, os, time, subprocess, shutil
import cv2
import numpy as np
import pandas as pd
from typing import List, Optional

# -------------------------
# Config / Defaults
# -------------------------
# A sample image uploaded earlier (use as default pasteable URL in the app)
DEFAULT_SAMPLE_FILE = "file:///mnt/data/b0cbf3d6-d0a0-43a8-bf55-255790044f95.png"

# -------------------------
# Helpers
# -------------------------


from pathlib import Path
import base64
import mimetypes

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

    # 3) fallback: embed HTML5 video with base64 (slower, but often works)
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
        st.warning(f"classes.txt not found at {p}. Provide path via sidebar.")
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

def draw_boxes_cv2(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes_idx: np.ndarray, class_names: List[str]):
    out = img.copy()
    font_scale = max(0.4, min(1.0, img.shape[1] / 1000.0))
    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes_idx):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (255, 180, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        label = f"{class_names[int(cls)] if class_names else int(cls)} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, y0), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out

def auto_select_weights() -> Optional[str]:
    runs_root = Path("runs") / "train"
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
    """Return a list of file info (relative path, suffix, size) for the most recent files."""
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

# -------------------------
# Frame -> video assembly and AVI->MP4 conversion helpers
# -------------------------
def ffmpeg_assemble_frames(frames_dir: Path, out_file: Path, fps: int = 25) -> Optional[Path]:
    """Try to assemble frames into mp4 with ffmpeg (glob). Returns out_file on success, else None."""
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
    """Assemble frames into mp4 using OpenCV (fallback)."""
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
    """Use ffmpeg to convert AVI -> MP4 with H.264 + AAC audio."""
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
    """Convert AVI -> MP4 using OpenCV (slower but no ffmpeg required)."""
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
# Cached model loader
# -------------------------
@st.cache_resource
def get_model(weights_path: str):
    # If weights_path doesn't exist, YOLO will try to load hub weights (internet needed).
    return YOLO(weights_path)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Wildlife Detection", layout="wide")
st.title("Wildlife detection (YOLOv8) — Image / Video / Batch")

# Sidebar controls
st.sidebar.header("Settings")
classes_default = r"C:\Users\vedan\Machine Learning\SWE PROJECT\dataset\classes.txt"
classes_path = st.sidebar.text_input("classes.txt path", classes_default)
names = load_class_names(classes_path)
st.sidebar.write(f"Loaded {len(names)} classes." if names else "No classes loaded.")

weights_default = auto_select_weights() or r"C:\Users\vedan\Machine Learning\SWE PROJECT\new_after_midsem\first_run\weights\best.pt"
weights_path = st.sidebar.text_input("Weights (best.pt / last.pt)", weights_default)
if not Path(weights_path).exists() and weights_path != "yolov8x.pt":
    st.sidebar.warning("Weights path doesn't exist. App will try to load fallback 'yolov8x.pt' from the hub (internet).")

imgsz = st.sidebar.select_slider("Image size (px)", options=[384, 512, 640, 768, 1024], value=1024)
conf_th = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
device = st.sidebar.text_input("Device ('0' for GPU index 0 or 'cpu')", "0")
if device != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

st.sidebar.markdown("---")
st.sidebar.write("Model will be cached after first load (keeps GPU memory).")
if st.sidebar.button("Reload model"):
    # clear cache and reload
    get_model.clear()
    st.experimental_rerun()

# load model (cached)
with st.spinner("Loading model..."):
    try:
        model = get_model(weights_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Tabs for image/video/batch
tab1, tab2, tab3 = st.tabs(["Image", "Video", "Batch"])

# ---------- Image tab ----------
with tab1:
    st.subheader("Image inference")
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        url_input = st.text_input("Or paste image URL or local file URL (file://...)", DEFAULT_SAMPLE_FILE)
        run_button = st.button("Run inference on image")
    with col2:
        st.write("Result")
        placeholder = st.empty()

    if run_button:
        # select image source
        if uploaded_img is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_img.name).suffix)
            tfile.write(uploaded_img.getvalue()); tfile.flush(); tfile.close()
            img_path = tfile.name
        elif url_input:
            if url_input.startswith("file://"):
                img_path = url_input.replace("file://", "")
            else:
                # try to download remote URL
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
                    with st.spinner("Running model..."):
                        res = model.predict(source=img_bgr, conf=conf_th, imgsz=imgsz, device=device, save=False, verbose=False)
                        r = res[0]
                        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.array([])
                        scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.array([])
                        cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.array([])
                        annotated = draw_boxes_cv2(img_bgr, boxes, scores, cls_idx, names)
                        # Convert BGR->RGB for Streamlit
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        placeholder.image(annotated_rgb, use_column_width=True)
                        # show table
                        df = pd.DataFrame([{"class_idx": int(c), "class": (names[int(c)] if names else int(c)), "conf": float(sc),
                                            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                                           for (x1, y1, x2, y2), sc, c in zip(boxes, scores, cls_idx)])
                        if not df.empty:
                            st.dataframe(df)
                        else:
                            st.info("No detections above threshold.")
            finally:
                # cleanup temp file if created
                if uploaded_img is not None:
                    try:
                        os.unlink(tfile.name)
                    except:
                        pass

# ---------- Video tab ----------
with tab2:
    st.subheader("Video inference")
    colv1, colv2 = st.columns([1, 2])
    with colv1:
        uploaded_vid = st.file_uploader("Upload video (mp4, avi, mov)", type=["mp4", "avi", "mov", "avi"])
        vid_url = st.text_input("Or paste local path / URL to video (file:///...)", "")
        fast_mode = st.checkbox("Fast (Ultralytics save=True) - quicker", value=True)
        run_vid = st.button("Run inference on video")
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
                # download remote url
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
            # use a stable output directory (not ephemeral tmp) so we can inspect files if needed
            out_dir = Path(r"C:\Users\vedan\Machine Learning\SWE PROJECT\new_after_midsem\runs\detect\predict3")
            try:
                # ---------------- FAST MODE (save=True) ----------------
                if fast_mode:
                    st.info("Running fast video annotate (Ultralytics). This may take a while depending on video length.")
                    with st.spinner("Annotating video..."):
                        _ = model.predict(
                            source=video_src,
                            conf=conf_th,
                            imgsz=imgsz,
                            device=device,
                            save=True,
                            save_dir=str(out_dir)
                        )

                    # Recursively search for any saved video file (newest first)
                    ann_vid = None
                    for f in sorted(out_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
                        if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                            ann_vid = f
                            break

                    if ann_vid:
                        # If it's an AVI, convert to MP4 for browser compatibility
                        if ann_vid.suffix.lower() == ".avi":
                            out_text_holder.info("Annotated .avi found — converting to mp4 for browser playback...")
                            mp4_out = ann_vid.with_name(ann_vid.stem + "_converted.mp4")
                            converted = ffmpeg_convert_avi_to_mp4(ann_vid, mp4_out)
                            if converted is None:
                                converted = python_convert_avi_to_mp4(ann_vid, mp4_out)
                            if converted:
                                st.success("Annotated video converted to MP4.")
                                final_video_path = Path(converted)  # or ann_vid if mp4
                                display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                # out_vid_holder.video(str(converted))
                                with open(converted, "rb") as fh:
                                    st.download_button("Download annotated video (mp4)", fh.read(), file_name=converted.name)
                            else:
                                st.error("Failed to convert .avi to .mp4. You can download the .avi and convert locally.")
                                with open(ann_vid, "rb") as fh:
                                    st.download_button("Download annotated video (avi)", fh.read(), file_name=ann_vid.name)
                        else:
                            st.success("Video annotated successfully!")
                            final_video_path = Path(ann_vid)  # or ann_vid if mp4
                            display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                            #out_vid_holder.video(str(ann_vid))
                            with open(ann_vid, "rb") as fh:
                                st.download_button("Download annotated video", fh.read(), file_name=ann_vid.name)
                    else:
                        # no video file found - maybe frames were saved -> attempt assembly
                        frames = sorted([p for p in out_dir.rglob("*.jpg")])
                        if frames:
                            st.info(f"Detected {len(frames)} annotated frames — attempting to assemble into mp4...")
                            assembled = None
                            out_mp4 = out_dir / (Path(video_src).stem + "_annotated_combined.mp4")
                            assembled = ffmpeg_assemble_frames(out_dir, out_mp4, fps=25)
                            if assembled is None:
                                assembled = python_assemble_frames(out_dir, out_mp4, fps=25)
                            if assembled:
                                st.success("Assembled annotated video.")
                                final_video_path = Path(assembled)  # or ann_vid if mp4
                                display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                                #out_vid_holder.video(str(assembled))
                                with open(assembled, "rb") as fh:
                                    st.download_button("Download annotated video", fh.read(), file_name=assembled.name)
                            else:
                                st.error("Failed to assemble frames into a video. Try turning off Fast mode or install ffmpeg.")
                        else:
                            st.error("Annotated video not found and no frames detected. Check the out directory:")
                            files_info = list_out_dir(out_dir, limit=200)
                            if files_info:
                                out_text_holder.text("\n".join([f"{p} | {s} | {size} bytes" for (p, s, size) in files_info[:100]]))
                            else:
                                out_text_holder.text("No files written to output directory (out_dir may be empty).")

                # ---------------- SLOW MODE (frame-by-frame) ----------------
                else:
                    st.info("Running slower frame-by-frame annotation (writing new video).")
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

                        st.success("Video annotated successfully!")
                        final_video_path = Path(t_out)  # or ann_vid if mp4
                        display_video_robust(out_vid_holder, final_video_path, text_holder=out_text_holder)
                        #out_vid_holder.video(str(t_out))
                        with open(t_out, "rb") as fh:
                            st.download_button("Download annotated video (mp4)", fh.read(), file_name=t_out.name)

            finally:
                # cleanup temp upload
                if uploaded_vid is not None:
                    try:
                        os.unlink(tfile.name)
                    except Exception:
                        pass

# ---------- Batch tab ----------
with tab3:
    st.subheader("Batch image folder inference")
    folder_path = st.text_input("Enter folder path on server (e.g. /dataset/test/images) or upload a zip", "")
    uploaded_zip = st.file_uploader("Or upload a zip of images (optional)", type=["zip"])
    run_batch_btn = st.button("Run batch inference")
    if run_batch_btn:
        # handle zip uploaded case
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
                    rows.append({"image": p.name, "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": float(sc), "cls_idx": int(cl), "cls_name": names[int(cl)] if names else int(cl)})
                pbar.progress((i + 1) / len(files))
            df = pd.DataFrame(rows)
            csv_p = outdir / "detections.csv"
            df.to_csv(csv_p, index=False)
            st.success("Batch complete.")
            st.write("Annotated images saved to server temp folder:", outdir)
            st.download_button("Download detections CSV", csv_p.read_bytes(), file_name="detections.csv")

st.markdown("---")
st.caption("Streamlit demo — you can adapt layout, add class filters, confidence slider live control, or integrate this with a REST API / Streamlit sharing.")
