from ultralytics import YOLO
import cv2
import os
import csv
from collections import deque

VIDEO_PATH = r"C:\Users\Admin\Downloads\backpack\ajay.mp4"
POSE_MODEL = "yolo11n-pose.pt"
DETECTION_MODEL = "yolo11n.pt"

PERSON_CONF = 0.45
BACKPACK_CONF = 0.15          # lowered further — dark backpack against dark clothing tanks confidence while carried

# ROI padding — asymmetric. Extra headroom on top because a hand reaching
# DOWN to grab a handle/strap sits above the backpack's own bounding box.
ROI_PAD_TOP = 130
ROI_PAD_SIDES = 70
ROI_PAD_BOTTOM = 60

# Rolling-window stability instead of "N in a row".
# A single missed detection frame no longer wipes out all progress.
WINDOW_SIZE = 10
WINDOW_REQUIRED = 6           # need True in at least 6 of the last 10 frames

# How many frames to keep trusting the last known backpack box if
# detection drops out for a moment (handles confidence flicker AND
# occlusion — e.g. a hand/leg blocking the pack while it's carried,
# which can last well over a second, not just a frame or two).
BACKPACK_MEMORY_FRAMES = 45

PERSON_MISSING_LIMIT = 15     # frames of no person before we call them "gone"

BACKPACK_CLASS_ID = 24

OUTPUT_VIDEO = "pick_place_output.mp4"
OUTPUT_CSV = "pick_place_log.csv"
OUTPUT_EVENTS = "pick_place_events.txt"


def inside_roi(point, roi, margin=0):
    if point is None or roi is None:
        return False
    x, y = point
    x1, y1, x2, y2 = roi
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def roi_overlap_pct(box, roi):
    """% of `box` (the backpack box) that overlaps with `roi`.
    0% = fully outside, 100% = backpack box fully inside the ROI.
    This is intersection-over-backpack-area, not full IoU — we care about
    how much of the BACKPACK is inside the zone, not how much of the
    (much larger) ROI the backpack fills."""
    if box is None or roi is None:
        return 0.0

    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi

    ix1, iy1 = max(bx1, rx1), max(by1, ry1)
    ix2, iy2 = min(bx2, rx2), min(by2, ry2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    box_area = max(1, (bx2 - bx1) * (by2 - by1))

    return round((inter_area / box_area) * 100, 1)


BACKPACK_OVERLAP_THRESHOLD = 40.0  # % of the backpack box that must sit inside the ROI to count as "IN"


def center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def text(frame, value, pos, size=0.6, color=(0, 255, 255)):
    cv2.putText(
        frame, value, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        size, color, 2, cv2.LINE_AA
    )


class RollingCheck:
    """Tracks the last WINDOW_SIZE frame results for a condition and says
    whether it has been true 'enough' recently — tolerant of flicker,
    unlike a strict consecutive-frame counter."""

    def __init__(self, window_size=WINDOW_SIZE, required=WINDOW_REQUIRED):
        self.buf = deque(maxlen=window_size)
        self.required = required

    def update(self, value):
        self.buf.append(bool(value))
        return sum(self.buf) >= self.required

    def reset(self):
        self.buf.clear()


if not os.path.exists(VIDEO_PATH):
    print("Video not found:", VIDEO_PATH)
    exit()

pose_model = YOLO(POSE_MODEL)
detect_model = YOLO(DETECTION_MODEL)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame", "timestamp",
    "person_detected", "person_id",
    "person_x1", "person_y1", "person_x2", "person_y2",
    "left_wrist_x", "left_wrist_y",
    "right_wrist_x", "right_wrist_y",
    "backpack_detected", "backpack_from_memory", "backpack_confidence",
    "backpack_x1", "backpack_y1",
    "backpack_x2", "backpack_y2",
    "backpack_center_x", "backpack_center_y",
    "backpack_roi_overlap_pct", "roi_value",
    "wrist_inside_roi", "backpack_inside_roi",
    "state", "event"
])

event_file = open(OUTPUT_EVENTS, "w", encoding="utf-8")

state = "PLACED"
previous_state = state

roi = None
roi_created = False

person_missing_counter = 0
frame_number = 0

# backpack memory (survives short detection gaps)
last_backpack_box = None
last_backpack_size = None
last_backpack_seen_frame = -999

# one rolling window per transition
placed_to_picking = RollingCheck()
picking_to_picked = RollingCheck()
leaves_to_placing_setup = RollingCheck()
placing_to_placed = RollingCheck()


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    timestamp = frame_number / fps

    person_found = False
    person_box = None
    person_id = None

    left_wrist = None
    right_wrist = None

    backpack_found_raw = False
    backpack_from_memory = False
    backpack_box = None
    backpack_center = None
    backpack_confidence = 0

    event = ""

    # ---------------- Person + pose ----------------
    pose_results = pose_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=PERSON_CONF,
        tracker="bytetrack.yaml",
        verbose=False
    )

    if len(pose_results) > 0:
        result = pose_results[0]

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                person_found = True
                coords = box.xyxy[0].cpu().numpy()
                person_box = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]

                if box.id is not None:
                    person_id = int(box.id[0].cpu().numpy())

                if result.keypoints is not None:
                    keypoints = result.keypoints.xy[i].cpu().numpy()

                    if len(keypoints) > 10:
                        lx, ly = keypoints[9]
                        rx, ry = keypoints[10]

                        if lx > 0 and ly > 0:
                            left_wrist = (int(lx), int(ly))
                        if rx > 0 and ry > 0:
                            right_wrist = (int(rx), int(ry))
                break

    # ---------------- Backpack detection ----------------
    detect_results = detect_model(frame, conf=BACKPACK_CONF, verbose=False)

    if len(detect_results) > 0:
        result = detect_results[0]
        best_conf = 0

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                if class_id == BACKPACK_CLASS_ID and confidence > best_conf:
                    best_conf = confidence
                    coords = box.xyxy[0].cpu().numpy()
                    backpack_box = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                    backpack_confidence = confidence
                    backpack_found_raw = True

    if backpack_found_raw:
        last_backpack_box = backpack_box
        last_backpack_size = (backpack_box[2] - backpack_box[0], backpack_box[3] - backpack_box[1])
        last_backpack_seen_frame = frame_number
        backpack_found = True
    else:
        within_memory = (frame_number - last_backpack_seen_frame) <= BACKPACK_MEMORY_FRAMES
        carrying_wrist = left_wrist if left_wrist is not None else right_wrist

        if within_memory and carrying_wrist is not None and last_backpack_size is not None:
            # Detection dropped (occlusion / dark-on-dark while carried) — instead of
            # freezing the box at its old resting spot, follow the wrist. A carried
            # backpack moves with the hand holding it, so this tracks the REAL
            # position instead of replaying a stale one that never leaves the ROI.
            w, h = last_backpack_size
            cx, cy = carrying_wrist
            backpack_box = [cx - w // 2, cy, cx + w // 2, cy + h]  # hangs below the hand
            backpack_found = True
            backpack_from_memory = True
        elif within_memory and last_backpack_box is not None:
            # no wrist available either — fall back to the last known box, but this
            # should only bridge very brief gaps (e.g. a blink-length occlusion
            # while the pack is still sitting still, not while it's being carried)
            backpack_box = last_backpack_box
            backpack_found = True
            backpack_from_memory = True
        else:
            backpack_found = False

    if backpack_found:
        backpack_center = center(backpack_box)

    # ---------------- Create fixed ROI (once) ----------------
    if not roi_created and backpack_found_raw and state == "PLACED":
        x1, y1, x2, y2 = backpack_box
        roi = (
            max(0, x1 - ROI_PAD_SIDES),
            max(0, y1 - ROI_PAD_TOP),
            min(width, x2 + ROI_PAD_SIDES),
            min(height, y2 + ROI_PAD_BOTTOM)
        )
        roi_created = True
        print("Fixed ROI created:", roi)

    # ---------------- ROI checks ----------------
    # small margin tolerance on wrist point — keypoints jitter a few px at boundaries
    wrist_inside = inside_roi(left_wrist, roi, margin=15) or inside_roi(right_wrist, roi, margin=15)
    backpack_overlap = roi_overlap_pct(backpack_box, roi)   # 0-100, one clean number
    backpack_inside = backpack_overlap >= BACKPACK_OVERLAP_THRESHOLD
    # ROI Value: inverse of overlap. 0 = fully inside/placed, 100 = fully picked/away.
    # Not a fixed/default number — it's live-computed every frame from the actual box position.
    roi_value = round(100 - backpack_overlap, 1) if backpack_found else 100.0

    # ---------------- Person visibility ----------------
    if person_found:
        person_missing_counter = 0
    else:
        person_missing_counter += 1

    # ========================================================
    # STATE MACHINE (rolling-window based, tolerant of flicker)
    # ========================================================

    if state == "PLACED":
        cond = person_found and backpack_found and backpack_inside and wrist_inside
        if placed_to_picking.update(cond):
            state = "PICKING"
            placed_to_picking.reset()
            picking_to_picked.reset()

    elif state == "PICKING":
        cond = person_found and backpack_found and not backpack_inside
        if picking_to_picked.update(cond):
            state = "PICKED"
            picking_to_picked.reset()

    elif state == "PICKED":
        if person_missing_counter >= PERSON_MISSING_LIMIT:
            state = "PERSON LEAVES"
            leaves_to_placing_setup.reset()

    elif state == "PERSON LEAVES":
        cond = person_found and backpack_found and backpack_inside and wrist_inside
        if leaves_to_placing_setup.update(cond):
            state = "PLACING"
            leaves_to_placing_setup.reset()
            placing_to_placed.reset()

    elif state == "PLACING":
        cond = backpack_found and backpack_inside and not wrist_inside
        if placing_to_placed.update(cond):
            state = "PLACED"
            placing_to_placed.reset()

    # ---------------- State change logging ----------------
    if state != previous_state:
        event = f"{previous_state} -> {state}"
        print(f"{timestamp:.2f}s : {event}")
        event_file.write(f"{timestamp:.2f}s : {event}\n")
        previous_state = state

    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        text(frame, "FIXED ROI", (x1, max(25, y1 - 10)))

    if backpack_found:
        x1, y1, x2, y2 = backpack_box
        color = (0, 165, 255) if backpack_from_memory else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not backpack_from_memory:
            label = f"Backpack {backpack_confidence:.2f}"
        elif left_wrist is not None or right_wrist is not None:
            label = "Backpack (following wrist)"
        else:
            label = "Backpack (last known)"
        text(frame, label, (x1, max(25, y1 - 10)), color=color)

    if person_box is not None:
        x1, y1, x2, y2 = person_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = "Person"
        if person_id is not None:
            label += f" ID:{person_id}"
        text(frame, label, (x1, max(25, y1 - 10)))

    if left_wrist is not None:
        cv2.circle(frame, left_wrist, 6, (0, 0, 255), -1)
        text(frame, "Left Wrist", (left_wrist[0] + 10, left_wrist[1] - 10), size=0.45, color=(0, 0, 255))
    if right_wrist is not None:
        cv2.circle(frame, right_wrist, 6, (0, 0, 255), -1)
        text(frame, "Right Wrist", (right_wrist[0] + 10, right_wrist[1] - 10), size=0.45, color=(0, 0, 255))

    # ---------------- Status panel ----------------
    cv2.rectangle(frame, (10, 10), (330, 175), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (330, 175), (255, 0, 0), 2)

    cv2.putText(frame, "ROI STATUS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Wrist: {'IN' if wrist_inside else 'OUT'}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    backpack_status = f"{backpack_overlap:.0f}% ({'IN' if backpack_inside else 'OUT'})"
    if backpack_found and backpack_from_memory:
        backpack_status += " mem"    # flags that this isn't a live detection this frame
    elif not backpack_found:
        backpack_status = "0% (OUT) lost"  # no detection AND memory window expired
    cv2.putText(frame, f"Backpack: {backpack_status}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"ROI Value: {roi_value:.0f}  (0=placed, 100=picked)", (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"State: {state}", (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_number}", (20, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ---------------- CSV row ----------------
    csv_writer.writerow([
        frame_number,
        f"{timestamp:.3f}",
        person_found,
        person_id if person_id is not None else "",

        person_box[0] if person_box else "",
        person_box[1] if person_box else "",
        person_box[2] if person_box else "",
        person_box[3] if person_box else "",

        left_wrist[0] if left_wrist else "",
        left_wrist[1] if left_wrist else "",

        right_wrist[0] if right_wrist else "",
        right_wrist[1] if right_wrist else "",

        backpack_found,
        backpack_from_memory,
        f"{backpack_confidence:.4f}" if backpack_found_raw else "",

        backpack_box[0] if backpack_box else "",
        backpack_box[1] if backpack_box else "",
        backpack_box[2] if backpack_box else "",
        backpack_box[3] if backpack_box else "",

        backpack_center[0] if backpack_center else "",
        backpack_center[1] if backpack_center else "",
        backpack_overlap,
        roi_value,

        wrist_inside,
        backpack_inside,

        state,
        event
    ])

    out.write(frame)

    cv2.imshow("Pick Place Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
csv_file.close()
event_file.close()
cv2.destroyAllWindows()

print("\nProcessing completed.")
print("Final state:", state)
print("Output video:", OUTPUT_VIDEO)
print("CSV log:", OUTPUT_CSV)
print("Event log:", OUTPUT_EVENTS)
