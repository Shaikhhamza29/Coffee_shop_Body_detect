# YOLOv8 Human Tracking (Raspberry Pi / Python)

A simple human detection and tracking system using **YOLOv8n** and OpenCV in Python.

This project detects persons in a video file, assigns them unique IDs,  
and displays each person’s:

- Top‑Left (TL) coordinates `(x1, y1)`
- Bottom‑Right (BR) coordinates `(x2, y2)`
- Width and Height of bounding box
- Person ID on the frame

It is optimized to run reasonably well on Raspberry Pi by scaling input and reducing detection frequency.

---

## 📦 Requirements

This project uses Python and the following packages:

✔ ultralytics (YOLOv8)  
✔ opencv‑python  
✔ numpy

If you already have a `requirements.txt`, you can install everything at once:

```bash
# Create & activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
