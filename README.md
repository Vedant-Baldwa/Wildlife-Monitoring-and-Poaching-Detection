
# 🐾 Wildlife Monitoring and Poaching Detection  
*A Dual-Model Computer Vision System using YOLOv8 & YOLOv8-OBB*
### 📄 Find full project report here [Project Report](LICENSE).

## 📌 Introduction
This project presents a robust real-time wildlife monitoring and poaching detection system using two deep-learning models:

1. **Wildlife Detection Model — YOLOv8x**  
2. **Poaching Detection Model — YOLOv8-OBB (Oriented Bounding Boxes)**

A fully interactive **Streamlit interface** is provided for testing image/video inputs and visualizing detections.

---

## ⭐ Features

### 🐘 Wildlife Detection (YOLOv8x)
- Detects 4 key species: **Buffalo, Elephant, Rhinoceros, Zebra**
- High performance:
  - Precision: 0.968
  - Recall: 0.925
  - mAP50: 0.973
  - mAP50-95: 0.844

### 🎯 Poaching Detection (YOLOv8-OBB)
- **71 threat classes**, including:
  - Weapons (rifles, pistols, crossbows)
  - Tools (nets, ropes, traps)
  - Vehicles (jeep, truck, motorbike)
  - Human intruders (hunters/poachers)
- Designed for rotated/tilted objects using OBB

### 💻 Streamlit Interface
- Image, video, and batch-folder inference  
- Real-time bounding box & OBB visualization  
- GPU-optimized inference

Run using:
```bash
streamlit run app.py
```

---

## 🔧 Installation

```bash
git clone https://github.com/Vedant-Baldwa/Wildlife-Monitoring-and-Poaching-Detection
cd Wildlife-Monitoring-and-Poaching-Detection
pip install -r requirements.txt
```

---

## ▶️ Usage

### Start the Streamlit App
```bash
streamlit run app.py
```

### Inference Options
- Upload **images**  
- Upload **videos**  
- Select model  
- Save annotated output  

---
## 📄 License

This project is licensed under the [MIT License](LICENSE).
