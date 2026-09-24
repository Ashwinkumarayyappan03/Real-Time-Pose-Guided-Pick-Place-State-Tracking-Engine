"""
Multi-person, multi-backpack pick/place detector.

CHANGES vs the original version (see comments tagged [FIX] below):

  [FIX 1] Raw YOLO/ByteTrack IDs are no longer trusted as permanent bag/person
          identity. ByteTrack's own internal track buffer (~30 frames) is
          shorter than BACKPACK_MEMORY_FRAMES (45), so on any occlusion longer
          than ~1s, ByteTrack silently drops the old track and hands back a
          brand-new id -- which the old script turned into a brand-new
          BackpackTrack, a brand-new ROI, and a permanently stacked dict entry.
          A new match_existing_backpack()/match_existing_person() step now
          re-attaches a "new" raw id to an existing bag/person if it's
          spatially close to one we already know, before any state/ROI logic
          runs. Genuinely new objects still get a genuinely new id.

  [FIX 2] BACKPACK_CONF was 0.15 -- low enough that a single physical bag can
          produce two overlapping detections in the SAME frame, which
          ByteTrack then tracks as two different ids simultaneously. Raised to
          0.30, plus a same-frame IoU dedup (dedup_overlapping) that collapses
          overlapping raw detections before they ever reach identity
          resolution.

  [FIX 3] Same churn problem existed for persons (pose_model.track ids), just
          less visible. Same identity-resolution pattern applied.

  ROI creation, the per-bag state machine, CSV logging, drawing and eviction
  are UNCHANGED -- they already assumed "one id = one physical object", which
  is now actually true.
"""

from ultralytics import YOLO
import cv2
import os
import csv
import math
from collections import deque

VIDEO_PATH = r"C:\Users\Admin\Downloads\backpack\both2.mp4"
POSE_MODEL = "yolo11n-pose.pt"
DETECTION_MODEL = "yolo11n.pt"

PERSON_CONF = 0.45
BACKPACK_CONF = 0.30          # [FIX 2] was 0.15

ROI_PAD_TOP = 130
ROI_PAD_SIDES = 70
ROI_PAD_BOTTOM = 60

WINDOW_SIZE = 10
WINDOW_REQUIRED = 6

BACKPACK_MEMORY_FRAMES = 45     # how long to keep trusting a bag's identity through a dropout
PERSON_MISSING_LIMIT = 15       # frames of the OWNER being gone before "PERSON LEAVES"

# max pixel distance for a wrist to be considered "belonging to" a given backpack
# (used both for picking an owner and for follow-during-occlusion fallback)
WRIST_MATCH_MAX_DIST = 250

# evict a tracked person/backpack from memory if unseen this long (housekeeping only,
# does not affect state logic beyond the owner-presence check above)
EVICT_AFTER_FRAMES = 300

BACKPACK_CLASS_ID = 24
BACKPACK_OVERLAP_THRESHOLD = 40.0  # % of backpack box that must sit inside its ROI to count as "IN"

# ---- [FIX 1/2] identity-resolution tunables --------------------------------
BACKPACK_DEDUP_IOU = 0.5            # same-frame duplicate boxes -> keep highest conf
BACKPACK_MATCH_RECALL_FRAMES = 150  # how long a "gone" bag stays eligible for re-matching
BACKPACK_MATCH_IOU = 0.12           # prefer overlap match
BACKPACK_MATCH_DIST = 300           # else nearest-center fallback (pixels)

PERSON_MATCH_RECALL_FRAMES = 90
PERSON_MATCH_IOU = 0.15
PERSON_MATCH_DIST = 400
# ------------------------------------------------------------------------------

OUTPUT_VIDEO = "pick_place_output_multi.mp4"
OUTPUT_CSV = "pick_place_log_multi.csv"
OUTPUT_EVENTS = "pick_place_events_multi.txt"

# distinct colors so multiple backpacks are visually distinguishable
PALETTE = [
    (0, 255, 0), (255, 0, 255), (0, 165, 255), (255, 255, 0),
    (255, 0, 0), (0, 255, 255), (128, 0, 255), (0, 128, 255),
]


def color_for_id(idx):
    return PALETTE[idx % len(PALETTE)]


def inside_roi(point, roi, margin=0):
    if point is None or roi is None:
        return False
    x, y = point
    x1, y1, x2, y2 = roi
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def roi_overlap_pct(box, roi):
    """% of `box` that overlaps with `roi` (intersection / box area)."""
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


def center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def distance(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def iou_boxes(a, b):
    """[FIX 1/2] plain box IoU, used for same-frame dedup and identity matching."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedup_overlapping(dets, iou_thr=BACKPACK_DEDUP_IOU):
    """[FIX 2] Collapse duplicate same-frame boxes (one physical bag firing
    two low-confidence detections) down to the single best one."""
    items = sorted(dets.items(), key=lambda kv: -kv[1][1])
    kept = {}
    kept_boxes = []
    for rid, (bbox, conf) in items:
        if all(iou_boxes(bbox, kb) < iou_thr for kb in kept_boxes):
            kept[rid] = (bbox, conf)
            kept_boxes.append(bbox)
    return kept


def match_existing_backpack(bbox, backpacks, frame_number):
    """[FIX 1] Try to re-attach a raw detection to a bag we already know,
    instead of minting a new BackpackTrack (and a new ROI) for it."""
    candidates = []
    for bid, bt in backpacks.items():
        ref_box = bt.last_known_box
        if ref_box is None or bt.last_known_frame < 0:
            continue
        if (frame_number - bt.last_known_frame) > BACKPACK_MATCH_RECALL_FRAMES:
            continue
        candidates.append((bid, iou_boxes(bbox, ref_box), distance(center(bbox), center(ref_box))))
    iou_hits = [c for c in candidates if c[1] >= BACKPACK_MATCH_IOU]
    if iou_hits:
        return max(iou_hits, key=lambda c: c[1])[0]
    dist_hits = [c for c in candidates if c[2] <= BACKPACK_MATCH_DIST]
    if dist_hits:
        return min(dist_hits, key=lambda c: c[2])[0]
    return None


def match_existing_person(bbox, persons, frame_number):
    """[FIX 3] Same idea as match_existing_backpack, for people."""
    candidates = []
    for pid, pt in persons.items():
        ref_box = pt.box
        if ref_box is None:
            continue
        if (frame_number - pt.last_seen_frame) > PERSON_MATCH_RECALL_FRAMES:
            continue
        candidates.append((pid, iou_boxes(bbox, ref_box), distance(center(bbox), center(ref_box))))
    iou_hits = [c for c in candidates if c[1] >= PERSON_MATCH_IOU]
    if iou_hits:
        return max(iou_hits, key=lambda c: c[1])[0]
    dist_hits = [c for c in candidates if c[2] <= PERSON_MATCH_DIST]
    if dist_hits:
        return min(dist_hits, key=lambda c: c[2])[0]
    return None


def text(frame, value, pos, size=0.55, color=(0, 255, 255)):
    cv2.putText(frame, value, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)


class RollingCheck:
    def __init__(self, window_size=WINDOW_SIZE, required=WINDOW_REQUIRED):
        self.buf = deque(maxlen=window_size)
        self.required = required

    def update(self, value):
        self.buf.append(bool(value))
        return sum(self.buf) >= self.required

    def reset(self):
        self.buf.clear()


class PersonTrack:
    def __init__(self, pid):
        self.id = pid
        self.box = None
        self.left_wrist = None
        self.right_wrist = None
        self.last_seen_frame = -1

    def wrists(self):
        out = []
        if self.left_wrist is not None:
            out.append(("L", self.left_wrist))
        if self.right_wrist is not None:
            out.append(("R", self.right_wrist))
        return out


class BackpackTrack:
    def __init__(self, bid):
        self.id = bid
        self.box = None
        self.confidence = 0.0
        self.from_memory = False
        self.last_seen_frame = -1
        self.last_real_size = None      # (w, h) from the last RAW detection

        # [FIX 1] persists longer than `box`/`last_seen_frame` so a bag can
        # still be re-identified after BACKPACK_MEMORY_FRAMES has lapsed.
        self.last_known_box = None
        self.last_known_frame = -1

        self.roi = None
        self.roi_created = False

        self.state = "PLACED"
        self.prev_state = "PLACED"
        self.owner_id = None
        self.candidate_owner_id = None
        self.owner_missing_counter = 0

        self.w_placed_to_picking = RollingCheck()
        self.w_picking_to_picked = RollingCheck()
        self.w_leaves_to_placing = RollingCheck()
        self.w_placing_to_placed = RollingCheck()


def find_nearest_wrist(point, all_wrists, max_dist=WRIST_MATCH_MAX_DIST, prefer_person_id=None):
    """all_wrists: list of (person_id, side, wrist_point).
    Returns (person_id, wrist_point) or (None, None).
    If prefer_person_id is given, only wrists from that person are considered."""
    best = None
    best_d = max_dist
    for pid, side, wp in all_wrists:
        if prefer_person_id is not None and pid != prefer_person_id:
            continue
        d = distance(point, wp)
        if d <= best_d:
            best_d = d
            best = (pid, wp)
    return best if best else (None, None)


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
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame", "timestamp", "num_persons_visible",
    "backpack_id", "backpack_state", "event",
    "backpack_confidence", "backpack_from_memory",
    "backpack_x1", "backpack_y1", "backpack_x2", "backpack_y2",
    "backpack_roi_overlap_pct", "roi_value",
    "wrist_inside_roi", "backpack_inside_roi",
    "owner_person_id"
])

event_file = open(OUTPUT_EVENTS, "w", encoding="utf-8")

persons = {}     # person_id (canonical) -> PersonTrack
backpacks = {}   # backpack_id (canonical) -> BackpackTrack

# [FIX 1/3] raw tracker id -> canonical id, plus canonical id counters
raw_to_canonical_person = {}
raw_to_canonical_backpack = {}
next_person_id = 0
next_backpack_id = 0

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    timestamp = frame_number / fps

    # ---------------- Persons (multi) ----------------
    pose_results = pose_model.track(
        frame, persist=True, classes=[0],
        conf=PERSON_CONF, tracker="bytetrack.yaml", verbose=False
    )

    seen_person_ids = set()

    if len(pose_results) > 0:
        result = pose_results[0]
        if result.boxes is not None:
            # [FIX 3] collect raw boxes first, resolve identity, THEN update tracks
            raw_person_boxes = {}
            for i, box in enumerate(result.boxes):
                if box.id is None:
                    continue  # unconfirmed track, skip until it has a stable ID
                raw_pid = int(box.id[0].cpu().numpy())
                coords = box.xyxy[0].cpu().numpy()
                bbox = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                raw_person_boxes[raw_pid] = (bbox, i)

            for raw_pid, (bbox, kp_idx) in raw_person_boxes.items():
                if raw_pid in raw_to_canonical_person:
                    pid = raw_to_canonical_person[raw_pid]
                else:
                    pid = match_existing_person(bbox, persons, frame_number)
                    if pid is None:
                        next_person_id += 1
                        pid = next_person_id
                    raw_to_canonical_person[raw_pid] = pid

                seen_person_ids.add(pid)

                if pid not in persons:
                    persons[pid] = PersonTrack(pid)
                p = persons[pid]

                p.box = bbox
                p.last_seen_frame = frame_number
                p.left_wrist = None
                p.right_wrist = None

                if result.keypoints is not None:
                    keypoints = result.keypoints.xy[kp_idx].cpu().numpy()
                    if len(keypoints) > 10:
                        lx, ly = keypoints[9]
                        rx, ry = keypoints[10]
                        if lx > 0 and ly > 0:
                            p.left_wrist = (int(lx), int(ly))
                        if rx > 0 and ry > 0:
                            p.right_wrist = (int(rx), int(ry))

    all_wrists = []  # (person_id, side, point) across every visible person, this frame
    for pid in seen_person_ids:
        for side, wp in persons[pid].wrists():
            all_wrists.append((pid, side, wp))

    # ---------------- Backpacks (multi, tracked by ID) ----------------
    detect_results = detect_model.track(
        frame, persist=True, classes=[BACKPACK_CLASS_ID],
        conf=BACKPACK_CONF, tracker="bytetrack.yaml", verbose=False
    )

    raw_detections = {}  # raw tracker id -> (box, confidence)
    if len(detect_results) > 0:
        result = detect_results[0]
        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                raw_bid = int(box.id[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                coords = box.xyxy[0].cpu().numpy()
                bbox = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                raw_detections[raw_bid] = (bbox, confidence)

    # [FIX 2] collapse duplicate overlapping boxes from the same frame
    raw_detections = dedup_overlapping(raw_detections)

    # [FIX 1] resolve raw ids -> stable canonical bag ids before anything
    # else (state machine, ROI creation, CSV) ever sees them
    resolved_detections = {}
    for raw_bid, (bbox, conf) in raw_detections.items():
        if raw_bid in raw_to_canonical_backpack:
            bid = raw_to_canonical_backpack[raw_bid]
        else:
            bid = match_existing_backpack(bbox, backpacks, frame_number)
            if bid is None:
                next_backpack_id += 1
                bid = next_backpack_id
            raw_to_canonical_backpack[raw_bid] = bid

        if bid in resolved_detections and conf <= resolved_detections[bid][1]:
            continue  # two raw ids collided onto the same bag this frame -> keep higher conf
        resolved_detections[bid] = (bbox, conf)

    raw_detections = resolved_detections  # everything below is unchanged from here on

    for bid, (bbox, conf) in raw_detections.items():
        if bid not in backpacks:
            backpacks[bid] = BackpackTrack(bid)
        bt = backpacks[bid]
        bt.box = bbox
        bt.confidence = conf
        bt.from_memory = False
        bt.last_seen_frame = frame_number
        bt.last_real_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        bt.last_known_box = bbox           # [FIX 1]
        bt.last_known_frame = frame_number  # [FIX 1]

    # apply occlusion memory for tracked backpacks not seen raw this frame
    for bid, bt in backpacks.items():
        if bid in raw_detections:
            continue
        within_memory = (frame_number - bt.last_seen_frame) <= BACKPACK_MEMORY_FRAMES
        if not within_memory or bt.last_real_size is None:
            bt.box = None
            bt.from_memory = False
            continue

        w, h = bt.last_real_size
        follow_point = None

        # prefer the known owner's wrist
        if bt.owner_id is not None and bt.owner_id in seen_person_ids:
            _, wp = find_nearest_wrist(bt.box and center(bt.box) or (0, 0), all_wrists,
                                        prefer_person_id=bt.owner_id)
            follow_point = wp

        # otherwise nearest wrist overall to the bag's last known position
        if follow_point is None and bt.box is not None:
            _, wp = find_nearest_wrist(center(bt.box), all_wrists)
            follow_point = wp

        if follow_point is not None:
            cx, cy = follow_point
            bt.box = [cx - w // 2, cy, cx + w // 2, cy + h]  # hangs below the hand
            bt.from_memory = True
        elif bt.box is not None:
            # no wrist to follow -- freeze at last known box (bridges blink-length gaps
            # while the bag is sitting still, not while being carried)
            bt.from_memory = True
        # else: box stays None -> treated as fully lost this frame

        if bt.box is not None:
            bt.last_known_box = bt.box            # [FIX 1] keep re-id reference fresh
            bt.last_known_frame = frame_number     # [FIX 1]

    # ---------------- Per-backpack ROI creation ----------------
    for bid, (bbox, conf) in raw_detections.items():
        bt = backpacks[bid]
        if not bt.roi_created and bt.state == "PLACED":
            x1, y1, x2, y2 = bbox
            bt.roi = (
                max(0, x1 - ROI_PAD_SIDES),
                max(0, y1 - ROI_PAD_TOP),
                min(width, x2 + ROI_PAD_SIDES),
                min(height, y2 + ROI_PAD_BOTTOM),
            )
            bt.roi_created = True
            print(f"Backpack {bid}: fixed ROI created: {bt.roi}")

    # ---------------- Per-backpack state machine ----------------
    for bid, bt in backpacks.items():
        backpack_found = bt.box is not None
        backpack_overlap = roi_overlap_pct(bt.box, bt.roi) if backpack_found else 0.0
        backpack_inside = backpack_overlap >= BACKPACK_OVERLAP_THRESHOLD
        roi_value = round(100 - backpack_overlap, 1) if backpack_found else 100.0

        wrist_inside = any(inside_roi(wp, bt.roi, margin=15) for _, _, wp in all_wrists)

        if bt.state in ("PLACED", "PICKING") and backpack_found:
            pid, wp = find_nearest_wrist(center(bt.box), all_wrists)
            if pid is not None and inside_roi(wp, bt.roi, margin=15):
                bt.candidate_owner_id = pid

        person_present_for_owner = (
            bt.owner_id in seen_person_ids if bt.owner_id is not None else len(seen_person_ids) > 0
        )

        event = ""

        if bt.state == "PLACED":
            cond = backpack_found and backpack_inside and wrist_inside and len(seen_person_ids) > 0
            if bt.w_placed_to_picking.update(cond):
                bt.state = "PICKING"
                bt.w_placed_to_picking.reset()
                bt.w_picking_to_picked.reset()

        elif bt.state == "PICKING":
            cond = backpack_found and not backpack_inside
            if bt.w_picking_to_picked.update(cond):
                bt.state = "PICKED"
                bt.owner_id = bt.candidate_owner_id
                bt.owner_missing_counter = 0
                bt.w_picking_to_picked.reset()

        elif bt.state == "PICKED":
            if not person_present_for_owner:
                bt.owner_missing_counter += 1
            else:
                bt.owner_missing_counter = 0
            if bt.owner_missing_counter >= PERSON_MISSING_LIMIT:
                bt.state = "PERSON LEAVES"
                bt.w_leaves_to_placing.reset()

        elif bt.state == "PERSON LEAVES":
            cond = backpack_found and backpack_inside and wrist_inside and len(seen_person_ids) > 0
            if bt.w_leaves_to_placing.update(cond):
                bt.state = "PLACING"
                bt.w_leaves_to_placing.reset()
                bt.w_placing_to_placed.reset()

        elif bt.state == "PLACING":
            cond = backpack_found and backpack_inside and not wrist_inside
            if bt.w_placing_to_placed.update(cond):
                bt.state = "PLACED"
                bt.owner_id = None
                bt.candidate_owner_id = None
                bt.w_placing_to_placed.reset()

        if bt.state != bt.prev_state:
            event = f"Backpack {bid}: {bt.prev_state} -> {bt.state}"
            print(f"{timestamp:.2f}s : {event}")
            event_file.write(f"{timestamp:.2f}s : {event}\n")
            bt.prev_state = bt.state

        csv_writer.writerow([
            frame_number, f"{timestamp:.3f}", len(seen_person_ids),
            bid, bt.state, event,
            f"{bt.confidence:.4f}" if backpack_found and not bt.from_memory else "",
            bt.from_memory,
            bt.box[0] if backpack_found else "",
            bt.box[1] if backpack_found else "",
            bt.box[2] if backpack_found else "",
            bt.box[3] if backpack_found else "",
            backpack_overlap, roi_value,
            wrist_inside, backpack_inside,
            bt.owner_id if bt.owner_id is not None else ""
        ])

        col = color_for_id(bid)
        if bt.roi is not None:
            rx1, ry1, rx2, ry2 = bt.roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col, 2)
            text(frame, f"ROI #{bid}", (rx1, max(20, ry1 - 8)), size=0.5, color=col)
        if backpack_found:
            bx1, by1, bx2, by2 = bt.box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)
            label = f"Bag {bid} {bt.state}"
            if bt.from_memory:
                label += " (mem)"
            if bt.owner_id is not None:
                label += f" owner:{bt.owner_id}"
            text(frame, label, (bx1, max(20, by1 - 8)), size=0.5, color=col)

    # evict stale backpacks/persons (housekeeping)
    for bid in [b for b, bt in backpacks.items() if frame_number - bt.last_seen_frame > EVICT_AFTER_FRAMES]:
        del backpacks[bid]
    for pid in [p for p, pt in persons.items() if frame_number - pt.last_seen_frame > EVICT_AFTER_FRAMES]:
        del persons[pid]

    # ---------------- Draw persons + wrists ----------------
    for pid in seen_person_ids:
        p = persons[pid]
        x1, y1, x2, y2 = p.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        text(frame, f"Person {pid}", (x1, max(20, y1 - 8)), size=0.55, color=(0, 255, 255))
        if p.left_wrist is not None:
            cv2.circle(frame, p.left_wrist, 6, (0, 0, 255), -1)
        if p.right_wrist is not None:
            cv2.circle(frame, p.right_wrist, 6, (0, 0, 255), -1)

    # ---------------- Status panel ----------------
    panel_h = 40 + 20 * max(1, len(backpacks))
    cv2.rectangle(frame, (10, 10), (380, 10 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (380, 10 + panel_h), (255, 255, 255), 2)
    cv2.putText(frame, f"Persons: {len(seen_person_ids)}  Bags: {len(backpacks)}  Frame: {frame_number}",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y = 55
    for bid, bt in backpacks.items():
        line = f"Bag {bid}: {bt.state}"
        if bt.owner_id is not None:
            line += f" (owner {bt.owner_id})"
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_for_id(bid), 1)
        y += 20

    out.write(frame)
    cv2.imshow("Multi Pick Place Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
csv_file.close()
event_file.close()
cv2.destroyAllWindows()

print("\nProcessing completed.")
print("Tracked backpacks:", list(backpacks.keys()))
print("Output video:", OUTPUT_VIDEO)
print("CSV log:", OUTPUT_CSV)
print("Event log:", OUTPUT_EVENTS)