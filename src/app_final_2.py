# app_final_B.py — Clean & improved version with shared inference helpers
# Based on your reference app: /mnt/data/app2.py
# Save and run with: streamlit run app_final_B.py

import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile, os, subprocess, mimetypes, base64, yaml
import cv2
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

ORIGINAL_UPLOADED = "/mnt/data/app2.py"
DEFAULT_SAMPLE_FILE = r"C:\Users\vedan\Machine Learning\SWE PROJECT\poaching\dataset\test\images\9c8a8884-5-2f1c_jpg.rf.834654bc5d9407eb11056f0ca121fd50.jpg"

# -------------------------
# Utility helpers
# -------------------------
def normalize_device(device_input: str) -> str:
    d = str(device_input).strip().lower()
    if d in ("cpu", "none", ""):
        return "cpu"
    if d.isdigit():
        return f"cuda:{int(d)}"
    if d.startswith("cuda"):
        return d
    return d

def save_img_temp_bgr(img_bgr: np.ndarray, suffix: str = ".jpg") -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    cv2.imwrite(tf.name, img_bgr)
    tf.close()
    return tf.name

def read_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except Exception:
        return None

def ffmpeg_convert_avi_to_mp4(avi_path: Path, mp4_path: Path) -> Optional[Path]:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = ["ffmpeg","-y","-i",str(avi_path),"-c:v","libx264","-preset","fast","-crf","23","-c:a","aac","-b:a","128k",str(mp4_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp4_path
    except Exception:
        return None

def display_video_robust(holder, video_path: Path, text_holder=None):
    video_path = Path(video_path)
    if text_holder is None:
        text_holder = holder
    if not video_path.exists():
        text_holder.error(f"Video not found: {video_path}")
        return
    size_mb = video_path.stat().st_size / (1024*1024)
    text_holder.info(f"Found annotated video: {video_path} ({size_mb:.2f} MB)")
    try:
        holder.video(str(video_path))
        text_holder.success("Displayed video via st.video(path).")
        return
    except Exception:
        pass
    try:
        with open(video_path,"rb") as f:
            holder.video(f.read())
            text_holder.success("Displayed video via st.video(bytes).")
            return
    except Exception:
        pass
    try:
        b = read_bytes(video_path)
        mime,_ = mimetypes.guess_type(str(video_path))
        if mime is None: mime = "video/mp4"
        b64 = base64.b64encode(b).decode()
        html = f'<video controls width="800"><source src="data:{mime};base64,{b64}" type="{mime}">Your browser does not support the video tag.</video>'
        holder.markdown(html, unsafe_allow_html=True)
        text_holder.success("Displayed via HTML5 video (base64).")
    except Exception as e:
        text_holder.error(f"Unable to display video: {e}")

# -------------------------
# Class name loading (yaml/classes.txt/model.names)
# -------------------------
def load_class_names_txt(path: Path) -> List[str]:
    if not path or not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf8").splitlines():
        s = line.strip()
        if not s: continue
        parts = s.split(None,1)
        if len(parts)==1: out.append(parts[0])
        else:
            if parts[0].isdigit(): out.append(parts[1].strip())
            else: out.append(s)
    return out

def load_classes_from_yaml(yaml_path: Path) -> List[str]:
    if not yaml_path or not yaml_path.exists(): return []
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf8"))
    except Exception:
        return []
    names = data.get("names") or data.get("labels") or data.get("label")
    if isinstance(names, list): return names
    if isinstance(names, dict):
        max_k = max(int(k) for k in names.keys())
        out = [None]*(max_k+1)
        for k,v in names.items(): out[int(k)] = v
        return out
    return []

# -------------------------
# auto_select_weights helper (was missing earlier)
# -------------------------
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

# -------------------------
# Drawing + OBB extraction helpers (shared)
# -------------------------
def extract_obb(r) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if r is None or not hasattr(r,"obb") or r.obb is None:
        return np.zeros((0,4)), np.zeros((0,5)), np.zeros((0,)), np.zeros((0,),dtype=int)
    obb = r.obb
    try:
        xyxy = obb.xyxy.cpu().numpy()
    except Exception:
        xyxy = np.zeros((0,4))
    try:
        xywha = obb.xywha.cpu().numpy()
    except Exception:
        xywha = np.zeros((0,5))
    try:
        scores = obb.conf.cpu().numpy()
    except Exception:
        scores = np.zeros((0,))
    try:
        classes = obb.cls.cpu().numpy().astype(int)
    except Exception:
        classes = np.zeros((0,),dtype=int)
    return xyxy, xywha, scores, classes

def draw_rotated_obb(img_bgr: np.ndarray, xywha: np.ndarray, scores: np.ndarray, classes: np.ndarray, class_names: List[str]=None):
    out = img_bgr.copy()
    if xywha is None or len(xywha) == 0:
        return out
    font_scale = max(0.4, min(1.0, img_bgr.shape[1]/1000.0))
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
    for (cx,cy,w,h,a), sc, cl in zip(xywha, scores, classes):
        angle = float(a)
        if abs(angle) <= 2*np.pi and abs(angle) < 10:
            angle_deg = np.degrees(angle)
        else:
            angle_deg = angle
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
        pts = cv2.boxPoints(rect).astype(int)
        color = (0,255,0)
        cv2.polylines(out, [pts], True, color, 2, lineType=cv2.LINE_AA)
        tlx, tly = pts.min(axis=0)
        label_cls = int(cl)
        if class_names and isinstance(class_names, list) and 0 <= label_cls < len(class_names):
            class_label = class_names[label_cls]
        else:
            class_label = str(label_cls)
        try:
            conf_float = float(sc)
        except Exception:
            conf_float = 0.0
        label = f"{class_label} {conf_float:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y0 = max(0, tly - th - 6)
        cv2.rectangle(out, (tlx, y0), (tlx + tw, tly), color, -1)
        cv2.putText(out, label, (tlx, tly - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, lineType=cv2.LINE_AA)
    return out

def draw_axis_aligned(img_bgr: np.ndarray, xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray, class_names: List[str]=None):
    out = img_bgr.copy()
    if xyxy is None or len(xyxy) == 0:
        return out
    font_scale = max(0.4, min(1.0, img_bgr.shape[1]/1000.0))
    n = xyxy.shape[0]
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
    for (x1,y1,x2,y2), sc, cl in zip(xyxy, scores, classes):
        x1,y1,x2,y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        color = (255,180,0)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2, lineType=cv2.LINE_AA)
        label_cls = int(cl)
        if class_names and isinstance(class_names, list) and 0 <= label_cls < len(class_names):
            class_label = class_names[label_cls]
        else:
            class_label = str(label_cls)
        try:
            conf_float = float(sc)
        except Exception:
            conf_float = 0.0
        label = f"{class_label} {conf_float:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, y0), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, lineType=cv2.LINE_AA)
    return out

# -------------------------
# Inference helper (single func for image/frame)
# -------------------------
def infer_and_annotate(model: YOLO, img_bgr: np.ndarray, conf: float, imgsz: int, device: str, class_names: List[str], prefer_rotated=True):
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = img_bgr
    try:
        res = model.predict(img_rgb, conf=conf, imgsz=imgsz, device=device, augment=False, save=False, verbose=False)
        r = res[0]
    except Exception:
        tmp = save_img_temp_bgr(img_bgr)
        try:
            res = model.predict(tmp, conf=conf, imgsz=imgsz, device=device, augment=False, save=False, verbose=False)
            r = res[0]
        finally:
            try: os.unlink(tmp)
            except: pass
    if r is None:
        return img_bgr.copy(), pd.DataFrame(columns=["class_idx","class","conf","x1","y1","x2","y2"])
    xyxy, xywha, scores, classes = extract_obb(r)
    annotated = img_bgr.copy()
    if prefer_rotated and xywha is not None and xywha.shape[0] > 0:
        annotated = draw_rotated_obb(annotated, xywha, scores, classes, class_names)
    elif xyxy is not None and xyxy.shape[0] > 0:
        annotated = draw_axis_aligned(annotated, xyxy, scores, classes, class_names)
    rows = []
    if xyxy is not None and xyxy.shape[0] > 0:
        for (x1,y1,x2,y2), sc, cl in zip(xyxy, scores, classes):
            rows.append({"class_idx":int(cl), "class":(class_names[int(cl)] if class_names and int(cl)<len(class_names) else int(cl)),
                         "conf":float(sc),"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2)})
    elif xywha is not None and xywha.shape[0] > 0:
        for (cx,cy,w,h,a), sc, cl in zip(xywha, scores, classes):
            x1 = float(cx) - float(w)/2.0
            y1 = float(cy) - float(h)/2.0
            x2 = float(cx) + float(w)/2.0
            y2 = float(cy) + float(h)/2.0
            rows.append({"class_idx":int(cl), "class":(class_names[int(cl)] if class_names and int(cl)<len(class_names) else int(cl)),
                         "conf":float(sc),"x1":x1,"y1":y1,"x2":x2,"y2":y2})
    df = pd.DataFrame(rows)
    return annotated, df

# -------------------------
# Cached model loader
# -------------------------
@st.cache_resource
def get_model(weights_path: str):
    return YOLO(weights_path)

# -------------------------
# Streamlit UI (improved)
# -------------------------
st.set_page_config(page_title="Wildlife & Poaching Detection (improved)", layout="wide")
st.title("Wildlife & Poaching Detection — Improved (Wildlife + OBB Poaching)")

# Sidebar
st.sidebar.header("Model paths & settings")
wild_classes_default = r"C:\Users\vedan\Machine Learning\SWE PROJECT\dataset_2\classes.txt"
wild_classes_path = Path(st.sidebar.text_input("Wildlife classes.txt (optional)", wild_classes_default))
wild_weights_default = auto_select_weights() or r"C:\Users\vedan\Machine Learning\SWE PROJECT\new_after_midsem\african_wildlife_runs\detect\wildlife_yolov8x_african\weights\best.pt"
wild_weights_path = st.sidebar.text_input("Wildlife weights", wild_weights_default)

st.sidebar.markdown("---")
poach_yaml_default = r"C:\Users\vedan\Machine Learning\SWE PROJECT\poaching\dataset\data.yaml"
poach_yaml_path = Path(st.sidebar.text_input("Poaching data.yaml (optional)", poach_yaml_default))
poach_classes_txt = st.sidebar.text_input("Poaching classes.txt (optional)", "")
poach_weights_default = r"C:\Users\vedan\Machine Learning\SWE PROJECT\poaching\poacher_obb\exp1\weights\best.pt"
poach_weights_path = st.sidebar.text_input("Poaching weights", poach_weights_default)

st.sidebar.markdown("---")
imgsz = st.sidebar.select_slider("Image size", [384,512,640,768,1024], value=1024)
wild_conf = st.sidebar.slider("Wildlife conf", 0.0,1.0,0.25,0.01)
poach_conf = st.sidebar.slider("Poaching conf", 0.0,1.0,0.25,0.01)
device_input = st.sidebar.text_input("Device ('cpu' or GPU index like '0')", "0")
device = normalize_device(device_input)
st.sidebar.write(f"Using device: {device}")
st.sidebar.markdown("---")
if st.sidebar.button("Reload models & clear cache"):
    get_model.clear()
    st.experimental_rerun()

# Load models
with st.spinner("Loading wildlife model..."):
    try:
        wild_model = get_model(wild_weights_path)
    except Exception as e:
        st.error(f"Failed to load wildlife model: {e}")
        st.stop()

poach_model = None
if poach_weights_path:
    with st.spinner("Loading poaching model..."):
        try:
            poach_model = get_model(poach_weights_path)
        except Exception as e:
            st.warning(f"Failed to load poaching model: {e}")
            poach_model = None
else:
    st.sidebar.info("No poaching weights provided")

# class names
wild_names = load_class_names_txt(wild_classes_path) if wild_classes_path else []
poach_names = []
if poach_classes_txt:
    poach_names = load_class_names_txt(Path(poach_classes_txt))
if not poach_names and poach_yaml_path.exists():
    poach_names = load_classes_from_yaml(poach_yaml_path)
if not poach_names and poach_model:
    mn = getattr(poach_model, "names", None)
    if mn:
        if isinstance(mn, dict):
            max_k = max(int(k) for k in mn.keys()); out = [None]*(max_k+1)
            for k,v in mn.items(): out[int(k)] = v
            poach_names = out
        elif isinstance(mn, list):
            poach_names = mn

# top-level tabs
tab1, tab2 = st.tabs(["Wildlife Detection", "Poaching Detection"])

# Wildlife tab (image sample)
with tab1:
    st.header("Wildlife Detection")
    w1, w2, w3 = st.tabs(["Image","Video","Batch"])
    with w1:
        st.subheader("Image — Wildlife")
        c1,c2 = st.columns([1,2])
        with c1:
            up = st.file_uploader("Upload wildlife image", type=["jpg","png","jpeg"], key="w_img")
            url = st.text_input("Or file:// path or URL", DEFAULT_SAMPLE_FILE, key="w_img_url")
            run = st.button("Run wildlife image", key="w_img_run")
        with c2:
            out = st.empty()
        if run:
            img_path = None
            if up:
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
                tf.write(up.getvalue()); tf.close(); img_path = tf.name
            elif url:
                if url.startswith("file://"): img_path = url.replace("file://","")
                else:
                    import requests
                    try:
                        r = requests.get(url, timeout=10)
                        if r.status_code==200:
                            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            tf.write(r.content); tf.close(); img_path = tf.name
                        else:
                            st.error("Failed download")
                    except Exception as e:
                        st.error(e)
            if img_path:
                img_bgr = cv2.imread(img_path)
                res = wild_model.predict(img_bgr, conf=wild_conf, imgsz=imgsz, device=device, save=False, verbose=False)
                r = res[0]
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes,"xyxy") else np.zeros((0,4))
                scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes,"conf") else np.zeros((0,))
                cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes,"cls") else np.zeros((0,),dtype=int)
                ann = draw_axis_aligned(img_bgr, boxes, scores, cls, wild_names)
                out.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                rows = [{"class_idx":int(c),"class":(wild_names[int(c)] if wild_names and int(c)<len(wild_names) else int(c)),
                         "conf":float(sc),"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2)}
                        for (x1,y1,x2,y2), sc, c in zip(boxes, scores, cls)]
                if rows:
                    st.dataframe(pd.DataFrame(rows))
                else:
                    st.info("No wildlife detections above threshold")
            if up:
                try: os.unlink(img_path)
                except: pass

with tab2:
    st.header("Poaching Detection (OBB support)")
    p1,p2 = st.tabs(["Image","Video"])
    with p1:
        st.subheader("Image — Poaching (OBB)")
        c1,c2 = st.columns([1,2])
        with c1:
            up = st.file_uploader("Upload poaching image", type=["jpg","jpeg","png"], key="p_img")
            url = st.text_input("Or file:// path or URL", DEFAULT_SAMPLE_FILE, key="p_img_url")
            run = st.button("Run poaching image", key="p_img_run")
            draw_mode = st.selectbox("Draw mode", ["Rotated (xywha)", "Axis-aligned (xyxy)"], index=0)
        with c2:
            out = st.empty()
        if run:
            if poach_model is None:
                st.error("Poaching model not loaded. Set weights in sidebar.")
            else:
                img_path = None
                if up:
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
                    tf.write(up.getvalue()); tf.close(); img_path = tf.name
                elif url:
                    if url.startswith("file://"):
                        img_path = url.replace("file://","")
                    else:
                        import requests
                        try:
                            r = requests.get(url, timeout=10)
                            if r.status_code==200:
                                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                                tf.write(r.content); tf.close(); img_path = tf.name
                            else:
                                st.error("Failed download")
                        except Exception as e:
                            st.error(e)
                if img_path:
                    img_bgr = cv2.imread(img_path)
                    ann, df = infer_and_annotate(poach_model, img_bgr, conf=poach_conf, imgsz=imgsz, device=device, class_names=poach_names, prefer_rotated=(draw_mode=="Rotated (xywha)"))
                    out.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    if not df.empty:
                        st.dataframe(df)
                    else:
                        st.info("No poaching detections above threshold")
                if up:
                    try: os.unlink(img_path)
                    except: pass

    with p2:
        st.subheader("Video — Poaching (OBB)")
        c1,c2 = st.columns([1,2])
        with c1:
            upv = st.file_uploader("Upload video (mp4/avi/mov)", type=["mp4","avi","mov"], key="p_vid")
            urlv = st.text_input("Or file:// path or URL", "", key="p_vid_url")
            fast = st.checkbox("Fast (Ultralytics save=True)", value=True, key="p_fast")
            runv = st.button("Run poaching video", key="p_vid_run")
            draw_mode = st.selectbox("Video draw mode", ["Rotated (xywha)", "Axis-aligned (xyxy)"], index=0, key="p_vid_draw")
        with c2:
            v_holder = st.empty(); t_holder = st.empty()
        if runv:
            if poach_model is None:
                st.error("Poaching model not loaded.")
            else:
                video_src = None
                if upv:
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upv.name).suffix)
                    tf.write(upv.getvalue()); tf.close(); video_src = tf.name
                elif urlv:
                    if urlv.startswith("file://"):
                        video_src = urlv.replace("file://","")
                    else:
                        import requests
                        try:
                            r = requests.get(urlv, stream=True, timeout=10)
                            if r.status_code==200:
                                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk: tf.write(chunk)
                                tf.close(); video_src = tf.name
                            else:
                                st.error("Failed download")
                        except Exception as e:
                            st.error(e)
                if video_src:
                    out_dir = Path(tempfile.mkdtemp(prefix="poach_vid_out_"))
                    try:
                        if fast:
                            t_holder.info("Running fast Ultralytics annotate (save=True)...")
                            _ = poach_model.predict(source=video_src, conf=poach_conf, imgsz=imgsz, device=device, save=True, save_dir=str(out_dir))
                            ann_vid = None
                            for f in sorted(out_dir.rglob("*"), key=lambda p:p.stat().st_mtime, reverse=True):
                                if f.suffix.lower() in [".mp4",".avi",".mov",".mkv"]:
                                    ann_vid = f; break
                            if ann_vid:
                                if ann_vid.suffix.lower()==".avi":
                                    t_holder.info("Converting .avi -> .mp4")
                                    mp4 = ann_vid.with_name(ann_vid.stem + "_conv.mp4")
                                    conv = ffmpeg_convert_avi_to_mp4(ann_vid, mp4) or None
                                    if conv: display_video_robust(v_holder, conv, t_holder)
                                    else: display_video_robust(v_holder, ann_vid, t_holder)
                                else:
                                    display_video_robust(v_holder, ann_vid, t_holder)
                            else:
                                frames = sorted(out_dir.rglob("*.jpg"))
                                if frames:
                                    t_holder.info(f"Found {len(frames)} frames — assembling...")
                                    out_mp4 = out_dir / (Path(video_src).stem + "_annotated.mp4")
                                    try:
                                        cmd = ["ffmpeg","-y","-pattern_type","glob","-i",str(out_dir/ "*.jpg"),"-c:v","libx264","-pix_fmt","yuv420p",str(out_mp4)]
                                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    except Exception:
                                        imgs = sorted([p for p in out_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
                                        first = cv2.imread(str(imgs[0])); h,w = first.shape[:2]
                                        fourcc = cv2.VideoWriter_fourcc(*"mp4v"); writer = cv2.VideoWriter(str(out_mp4), fourcc, 25, (w,h))
                                        for p in imgs:
                                            im = cv2.imread(str(p))
                                            if im.shape[:2] != (h,w): im = cv2.resize(im,(w,h))
                                            writer.write(im)
                                        writer.release()
                                    display_video_robust(v_holder, out_mp4, t_holder)
                                else:
                                    t_holder.error("No annotated video or frames found")
                        else:
                            t_holder.info("Processing frame-by-frame (slower)...")
                            cap = cv2.VideoCapture(str(video_src))
                            if not cap.isOpened(): t_holder.error("Cannot open video")
                            else:
                                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                out_file = Path(tempfile.mkdtemp()) / (Path(video_src).stem + "_annotated.mp4")
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w,h))
                                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                                pbar = st.progress(0)
                                i = 0
                                while True:
                                    ret, frame = cap.read()
                                    if not ret: break
                                    ann, df = infer_and_annotate(poach_model, frame, conf=poach_conf, imgsz=imgsz, device=device, class_names=poach_names, prefer_rotated=(draw_mode=="Rotated (xywha)"))
                                    writer.write(ann)
                                    i += 1
                                    if frames > 0: pbar.progress(min(1.0, i/frames))
                                writer.release(); cap.release()
                                display_video_robust(v_holder, out_file, t_holder)
                    finally:
                        if upv:
                            try: os.unlink(video_src)
                            except: pass

st.markdown("---")
st.caption("Improved app: shared inference helpers + OBB support for Poaching model. Use the sidebar to tune settings.")
