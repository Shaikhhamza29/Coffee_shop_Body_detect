import cv2
from ultralytics import YOLO


class HumanTracker:
    """
    YOLOv8 Human Tracker
    - Frame-by-frame processing
    - Stable IDs
    - Smoothed bounding boxes
    - Reusable as a package
    """

    DEFAULT_IOU_THRESHOLD = 0.5
    DEFAULT_CONF_THRESHOLD = 0.5
    DEFAULT_IMGSZ = 832
    DEFAULT_BOX_THICKNESS = 1

    DETECT_EVERY_N_FRAMES = 7   # set 1 = detect every frame
    SMOOTH_ALPHA = 0.5

    def __init__(
        self,
        model_path="yolov8s.pt",
        iou_threshold=DEFAULT_IOU_THRESHOLD,
        conf_threshold=DEFAULT_CONF_THRESHOLD,
        imgsz=DEFAULT_IMGSZ,
        box_thickness=DEFAULT_BOX_THICKNESS,
    ):
        self.iou_threshold = iou_threshold
        self.box_thickness = box_thickness

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf_threshold

        self.tracks = {}          # raw detections
        self.smoothed = {}        # smoothed boxes
        self.cached_persons = {}  # public output

        self.next_id = 1
        self.frame_count = 0

        print("🔥 HumanTracker initialized (FRAME-BY-FRAME MODE)")

    # ---------- IOU ----------
    @staticmethod
    def _iou(a, b):
        ax2, ay2 = a["x1"] + a["w"], a["y1"] + a["h"]
        bx2, by2 = b["x1"] + b["w"], b["y1"] + b["h"]

        xA = max(a["x1"], b["x1"])
        yA = max(a["y1"], b["y1"])
        xB = min(ax2, bx2)
        yB = min(ay2, by2)

        inter = max(0, xB - xA) * max(0, yB - yA)
        union = (a["w"] * a["h"]) + (b["w"] * b["h"]) - inter
        return inter / union if union > 0 else 0

    # ---------- DETECT ----------
    def _detect(self, frame):
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[0],  # person
            verbose=False
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "w": x2 - x1,
                    "h": y2 - y1
                })
        return detections

    # ---------- TRACK UPDATE ----------
    def _update(self, detections):
        active = set()

        for det in detections:
            best_id = None
            best_score = 0

            for pid, track in self.tracks.items():
                score = self._iou(det, track)
                if score > best_score:
                    best_score = score
                    best_id = pid

            if best_id is None or best_score < self.iou_threshold:
                pid = self.next_id
                self.next_id += 1
                self.tracks[pid] = det
                self.smoothed[pid] = det.copy()
            else:
                pid = best_id
                self.tracks[pid] = det

            active.add(pid)

        self.tracks = {pid: t for pid, t in self.tracks.items() if pid in active}
        self.smoothed = {pid: s for pid, s in self.smoothed.items() if pid in active}

    # ---------- SMOOTH ----------
    def _smooth(self):
        if not self.tracks:
            self.cached_persons = {}
            return

        a = self.SMOOTH_ALPHA
        for pid, target in self.tracks.items():
            s = self.smoothed.get(pid, target.copy())

            s["x1"] += int(a * (target["x1"] - s["x1"]))
            s["y1"] += int(a * (target["y1"] - s["y1"]))
            s["w"]  += int(a * (target["w"]  - s["w"]))
            s["h"]  += int(a * (target["h"]  - s["h"]))

            self.smoothed[pid] = s

        self.cached_persons = {
            pid: {
                "x1": s["x1"],
                "y1": s["y1"],
                "w": s["w"],
                "h": s["h"],
                "x2": s["x1"] + s["w"],
                "y2": s["y1"] + s["h"]
            }
            for pid, s in self.smoothed.items()
        }

    # ---------- MAIN API ----------
    def process_frame(self, frame):
        """
        Pass ONE frame → get people data
        """
        self.frame_count += 1

        if self.frame_count % self.DETECT_EVERY_N_FRAMES == 0:
            detections = self._detect(frame)
            self._update(detections)

        self._smooth()
        return self.cached_persons.copy()

    # ---------- OPTIONAL DRAW ----------
    def draw(self, frame):
        # ---- PEOPLE COUNT ON SCREEN ----
        cv2.putText(
            frame,
            f"People: {len(self.cached_persons)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        # ---- DRAW EACH PERSON ----
        for pid, p in self.cached_persons.items():
            x1, y1 = p["x1"], p["y1"]
            w, h = p["w"], p["h"]
            x2, y2 = p["x2"], p["y2"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                self.box_thickness
            )

            # ID
            cv2.putText(
                frame,
                f"ID {pid}",
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Coordinates x1, y1
            cv2.putText(
                frame,
                f"x1:{x1} y1:{y1}",
                (x1, y1 - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1
            )

            # Width, Height
            cv2.putText(
                frame,
                f"w:{w} h:{h}",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1
            )


    # ---------- SINGLE CALL API ----------
    def analyze_frame(self, frame, draw=True):
        """
        ONE function call.
        Pass frame → get people count + bounding boxes.
        """
        persons = self.process_frame(frame)

        if draw:
            self.draw(frame)

        return {
            "count": len(persons),
            "persons": persons
        }
