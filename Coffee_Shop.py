import cv2
import time
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATH = "CCTV.mp4"

DETECT_INTERVAL = 0.5          # seconds
BOX_THICKNESS = 2
IOU_THRESHOLD = 0.3

YOLO_IMGSZ = 960               # better for far people
CONF_THRESHOLD = 0.25

# ================= MODEL =================
person_model = YOLO("yolov8s.pt")
print("🔥 YOLO loaded")

# ================= STATE =================
next_person_id = 1
last_detect_time = 0
persons = {}

# ================= IOU FUNCTION =================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

print("🔥 Body Tracking Started")

# ================= MAIN LOOP =================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ ERROR: Video not opened")
    exit()

# ---------- FPS SYNC ----------
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
frame_delay = int(1000 / fps)
print(f"📽 Video FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    now = time.time()

    # ================= DETECTION =================
    if now - last_detect_time >= DETECT_INTERVAL:
        results = person_model(
            frame,
            imgsz=YOLO_IMGSZ,
            conf=CONF_THRESHOLD,
            classes=[0],
            verbose=False
        )

        active_ids = set()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)

                best_id = None
                best_score = 0.0

                for pid, pdata in persons.items():
                    score = iou(bbox, pdata["bbox"])
                    if score > best_score:
                        best_score = score
                        best_id = pid

                if best_id is None or best_score < IOU_THRESHOLD:
                    pid = next_person_id
                    next_person_id += 1
                    persons[pid] = {"bbox": bbox}
                    print(f"➡️ ID {pid} ENTERED | TL=({x1},{y1}) BR=({x2},{y2})")
                else:
                    pid = best_id
                    persons[pid]["bbox"] = bbox

                active_ids.add(pid)

        # Remove disappeared persons
        for pid in list(persons.keys()):
            if pid not in active_ids:
                del persons[pid]

        last_detect_time = now

    # ================= DRAW =================
    for pid, pdata in persons.items():
        x1, y1, x2, y2 = pdata["bbox"]
        w = x2 - x1
        h = y2 - y1

        # ---- PRINT FULL COORDINATES ----
        print(f"ID {pid} | TL=({x1},{y1}) BR=({x2},{y2}) W={w} H={h}")

        # ---- DRAW BOX ----
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0), BOX_THICKNESS)

        # ---- DRAW TEXT ----
        cv2.putText(frame, f"ID {pid}",
                    (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"TL({x1},{y1}) BR({x2},{y2})",
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1)

        cv2.putText(frame, f"W:{w} H:{h}",
                    (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 0), 1)

    cv2.imshow("Human Tracking - CCTV", frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
