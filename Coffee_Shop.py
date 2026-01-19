import cv2
import time
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATH = "video3.mp4"
DETECT_INTERVAL = 1.0     # Run YOLO every ~1 second
BOX_THICKNESS = 2
IOU_THRESHOLD = 0.3
FRAME_SCALE = 0.5         # Scale down frames for faster detection

# ================= BODY MODEL =================
person_model = YOLO("yolov8n.pt")
print("🔥 YOLO loaded")

# ================= STATE =================
next_person_id = 1
last_detect_time = 0
persons = {}

# ================= UTILS =================
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
while True:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ ERROR: Video not opened")
        break

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video → restart
            break

        # Create a smaller frame for YOLO (faster inference)
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        now = time.time()

        # ================= DETECTION =================
        if now - last_detect_time >= DETECT_INTERVAL:
            results = person_model(small_frame, conf=0.4, classes=[0], verbose=False)

            active_ids = set()
            for r in results:
                for box in r.boxes:
                    # Get original coordinates (scaled back)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = int(x1 / FRAME_SCALE)
                    y1 = int(y1 / FRAME_SCALE)
                    x2 = int(x2 / FRAME_SCALE)
                    y2 = int(y2 / FRAME_SCALE)

                    bbox = (x1, y1, x2, y2)

                    # ---------------- Detect / Match ----------------
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
                        persons[pid] = {"bbox": bbox, "last_seen": now}
                        print(f"➡️ Person {pid} entered: TL=({x1},{y1}) BR=({x2},{y2})")
                    else:
                        pid = best_id
                        persons[pid]["bbox"] = bbox
                        persons[pid]["last_seen"] = now

                    active_ids.add(pid)

            # Remove lost persons
            for pid in list(persons.keys()):
                if pid not in active_ids:
                    del persons[pid]

            last_detect_time = now

        # ================= DRAW =================
        for pid, pdata in persons.items():
            x1, y1, x2, y2 = pdata["bbox"]
            w = x2 - x1
            h = y2 - y1

            # Add bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)

            # Draw labels with coordinates
            cv2.putText(frame, f"ID {pid}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"TL:({x1},{y1})", (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            cv2.putText(frame, f"BR:({x2},{y2})", (x1 + 5, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            cv2.putText(frame, f"W:{w} H:{h}", (x1 + 5, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            print(f"ID {pid}: TL=({x1},{y1}), BR=({x2},{y2}), W={w}, H={h}")

        cv2.imshow("Human Tracking - Raspberry Pi", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()  # restart video automatically
