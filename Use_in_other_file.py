import cv2
from Coffee_Shop import HumanTracker

cap = cv2.VideoCapture("Videos/CCTV.mp4")
tracker = HumanTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    persons = tracker.process_frame(frame)

    print("People count:", len(persons))
    print("Bounding boxes:", persons)

    tracker.draw(frame)

    cv2.imshow("Human Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
