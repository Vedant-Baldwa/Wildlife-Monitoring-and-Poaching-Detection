# app.py
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile, os, time
import cv2
import numpy as np
import pandas as pd
from typing import List

# -------------------------
# Helpers
# -------------------------
DEFAULT_SAMPLE_FILE = "file:///mnt/data/03916b41-481c-43a5-8eb3-e7d5c17d242a.png"

def load_class_names(classes_path: str) -> List[str]:
    p = Path(classes_path)
    if not p.exists():
        st.warning(f"classes.txt not found at {p}. Provide path via sidebar.")
        return []
    names=[]
    with open(p, "r", encoding="utf8") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            parts=s.split(None,1)
            if len(parts)==1:
                names.append(parts[0])
            else:
                if parts[0].isdigit():
                    names.append(parts[1].strip())
                else:
                    names.append(s)
    return names

def draw_boxes_cv2(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes_idx: np.ndarray, class_names: List[str]):
    out = img.copy()
    font_scale = max(0.4, min(1.0, img.shape[1]/1000.0))
    for (x1,y1,x2,y2), conf, cls in zip(boxes, scores, classes_idx):
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        color = (255,180,0)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2, lineType=cv2.LINE_AA)
        label = f"{class_names[int(cls)] if class_names else int(cls)} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1,y0), (x1+tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, lineType=cv2.LINE_AA)
    return out

def auto_select_weights():
    runs_root = Path("runs") / "train"
    if not runs_root.exists():
        return None
    for r in sorted(runs_root.iterdir(), key=lambda p:p.stat().st_mtime, reverse=True):
        best = r / "weights" / "best.pt"
        last = r / "weights" / "last.pt"
        if best.exists(): return str(best)
        if last.exists(): return str(last)
    return None

# -------------------------
# Cached model loader
# -------------------------
@st.cache_resource
def get_model(weights_path: str):
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

weights_default = r"C:\Users\vedan\Machine Learning\SWE PROJECT\new_after_midsem\first_run\weights\best.pt"
weights_path = st.sidebar.text_input("Weights (best.pt / last.pt)", weights_default)
if not Path(weights_path).exists() and weights_path != "yolov8x.pt":
    st.sidebar.warning("Weights path doesn't exist. App will try to load fallback 'yolov8x.pt' from the hub (internet).")

imgsz = st.sidebar.select_slider("Image size (px)", options=[384,512,640,768,1024], value=1024)
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
    col1, col2 = st.columns([1,2])
    with col1:
        uploaded_img = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
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
                img_path = url_input.replace("file://","")
            else:
                # try to download remote URL
                import requests
                try:
                    r = requests.get(url_input, timeout=10)
                    if r.status_code==200:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        tfile.write(r.content); tfile.flush(); tfile.close()
                        img_path = tfile.name
                    else:
                        st.error("Could not download URL: status="+str(r.status_code)); img_path=None
                except Exception as e:
                    st.error(f"Failed to download URL: {e}"); img_path=None
        else:
            st.error("Provide an uploaded file or a URL"); img_path=None

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
                        df = pd.DataFrame([{"class_idx":int(c),"class": (names[int(c)] if names else int(c)), "conf":float(sc),
                                            "x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2)}
                                           for (x1,y1,x2,y2), sc, c in zip(boxes, scores, cls_idx)])
                        if not df.empty:
                            st.dataframe(df)
                        else:
                            st.info("No detections above threshold.")
            finally:
                # cleanup temp file if created
                if uploaded_img is not None:
                    try: os.unlink(tfile.name)
                    except: pass

# ---------- Video tab ----------
# ---------- Video tab ----------
with tab2:
    st.subheader("Video inference")
    colv1, colv2 = st.columns([1,2])
    with colv1:
        uploaded_vid = st.file_uploader("Upload video (mp4, avi)", type=["mp4","avi","mov"])
        vid_url = st.text_input("Or paste local path / URL to video (file:///...)", "")
        fast_mode = st.checkbox("Fast (Ultralytics save=True) - quicker", value=True)
        run_vid = st.button("Run inference on video")
    with colv2:
        out_vid_holder = st.empty()

    if run_vid:
        # choose source
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_vid.name).suffix)
            tfile.write(uploaded_vid.getvalue()); tfile.flush(); tfile.close()
            video_src = tfile.name
        elif vid_url:
            if vid_url.startswith("file://"):
                video_src = vid_url.replace("file://","")
            else:
                # download remote url
                import requests
                try:
                    r = requests.get(vid_url, stream=True, timeout=10)
                    if r.status_code==200:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                tfile.write(chunk)
                        tfile.flush(); tfile.close()
                        video_src = tfile.name
                    else:
                        st.error("Could not download video: status="+str(r.status_code)); video_src=None
                except Exception as e:
                    st.error(f"Failed to download video: {e}"); video_src=None
        else:
            st.error("Provide a video upload or file:// path"); video_src=None

        if video_src:
            out_dir = Path(r"C:\Users\vedan\Machine Learning\SWE PROJECT\new_after_midsem\runs\detect\predict2")
            try:
                # ---------------- FAST MODE (save=True) ----------------
                if fast_mode:
                    st.info("Running fast video annotate (Ultralytics). This may take a while depending on video length.")
                    with st.spinner("Annotating video..."):
                        results = model.predict(
                            source=video_src,
                            conf=conf_th,
                            imgsz=imgsz,
                            device=device,
                            save=True,
                            save_dir=str(out_dir)
                        )

                    # find annotated video recursively
                    ann_vid = None
                    for f in out_dir.rglob("*"):
                        if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                            ann_vid = f
                            break

                    if ann_vid:
                        st.success("Video annotated successfully!")
                        out_vid_holder.video(str(ann_vid))
                    else:
                        st.error("Annotated video not found. Ultralytics may have saved frames instead of video.")
                        frames = list(out_dir.rglob("*.jpg"))
                        if frames:
                            st.warning("Detected annotated frames instead of video. Turn OFF fast mode to force video creation.")

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
                            if not ret: break

                            res = model.predict(frame, conf=conf_th, imgsz=imgsz, device=device, save=False, verbose=False)
                            r = res[0]
                            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes,"xyxy") else []
                            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes,"conf") else []
                            cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes,"cls") else []

                            annotated = draw_boxes_cv2(frame, boxes, scores, cls_idx, names)
                            writer.write(annotated)

                            i += 1
                            if frames > 0:
                                pbar.progress(min(1.0, i / frames))

                        writer.release()
                        cap.release()

                        st.success("Video annotated successfully!")
                        out_vid_holder.video(str(t_out))

            finally:
                if uploaded_vid is not None:
                    try: os.unlink(tfile.name)
                    except: pass


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
            with zipfile.ZipFile(zipf, 'r') as z:
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
            files = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
            rows=[]
            for i,p in enumerate(files):
                res = model.predict(source=str(p), conf=conf_th, imgsz=imgsz, device=device, save=False, verbose=False)
                r=res[0]
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes,"xyxy") else []
                scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes,"conf") else []
                cls_idx = r.boxes.cls.cpu().numpy() if hasattr(r.boxes,"cls") else []
                img = cv2.imread(str(p))
                annotated = draw_boxes_cv2(img, boxes, scores, cls_idx, names)
                outp = outdir / p.name
                cv2.imwrite(str(outp), annotated)
                for (x1,y1,x2,y2), sc, cl in zip(boxes, scores, cls_idx):
                    rows.append({"image":p.name,"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),"conf":float(sc),"cls_idx":int(cl),"cls_name":names[int(cl)] if names else int(cl)})
                pbar.progress((i+1)/len(files))
            df = pd.DataFrame(rows)
            csv_p = outdir / "detections.csv"
            df.to_csv(csv_p, index=False)
            st.success("Batch complete.")
            st.write("Annotated images saved to server temp folder:", outdir)
            st.download_button("Download detections CSV", csv_p.read_bytes(), file_name="detections.csv")

st.markdown("---")
st.caption("Streamlit demo — you can adapt layout, add class filters, confidence slider live control, or integrate this with a REST API / Streamlit sharing.")