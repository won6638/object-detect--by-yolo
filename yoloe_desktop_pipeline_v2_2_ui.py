from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


WINDOW_NAME = "YOLOE-26 Desktop Pipeline v2.2 - VLM HUD"
MIN_DRAG_PIXELS = 12
SCHEMA_VERSION = "2.0.0"


@dataclass
class AutoTrack:
    box: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    track_id: int
    display_id: int
    mask_polygon: np.ndarray | None = None


@dataclass
class ManualROI:
    roi_id: int
    object_id: str
    box: tuple[int, int, int, int]


@dataclass
class ResultCard:
    object_id: str
    request_id: str
    capture_id: str
    record_path: Path
    track_id: int | None = None
    manual_roi_id: int | None = None
    status: str = "pending"
    response: dict[str, Any] | None = None
    harness: dict[str, Any] | None = None
    mtime_ns: int = -1
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass
class UIState:
    current_tracks: list[AutoTrack] = field(default_factory=list)
    manual_rois: list[ManualROI] = field(default_factory=list)

    selected_auto_ids: set[int] = field(default_factory=set)
    selected_manual_ids: set[int] = field(default_factory=set)

    # App-level object identity for selected auto tracks.
    track_object_ids: dict[int, str] = field(default_factory=dict)

    # UI/display tracking aids.
    seen_streak: dict[int, int] = field(default_factory=dict)
    lost_frame_counts: dict[int, int] = field(default_factory=dict)

    next_manual_id: int = 1

    drag_start: tuple[int, int] | None = None
    drag_current: tuple[int, int] | None = None
    is_dragging: bool = False

    frame_width: int = 0
    frame_height: int = 0

    status_message: str = ""
    status_until: float = 0.0

    show_detector_label: bool = False
    show_selected_mask: bool = True
    show_result_cards: bool = True

    # VLM/UI integration.
    query_on_select: bool = True
    pending_auto_query_ids: set[int] = field(default_factory=set)
    pending_manual_query_ids: set[int] = field(default_factory=set)
    result_cards_by_object: dict[str, ResultCard] = field(default_factory=dict)
    last_result_poll: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "YOLOE-26 prompt-free discovery + BoT-SORT tracking + "
            "capture/record.json + queue.jsonl/events.jsonl."
        )
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--scan", action="store_true")

    parser.add_argument("--model", default="yoloe-26s-seg-pf.pt")
    parser.add_argument("--tracker", default="botsort_yoloe_desktop.yaml")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)

    # Requested change: use capture/ as the single output root.
    parser.add_argument("--output", default="capture")

    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--llm-jpeg-quality", type=int, default=88)
    parser.add_argument("--llm-max-side", type=int, default=512)

    parser.add_argument(
        "--lost-after",
        type=int,
        default=60,
        help="Frames absent before selected auto object becomes LOST.",
    )
    parser.add_argument(
        "--min-track-age",
        type=int,
        default=3,
        help="Frames a proposal must survive before being shown.",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=0.80,
        help="Hide scene-level proposals larger than this frame ratio.",
    )
    parser.add_argument(
        "--query-on-select",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically create a VLM request the first time an object/ROI is selected. "
            "Use --no-query-on-select to keep the old C-key-only behavior."
        ),
    )
    parser.add_argument(
        "--result-poll-ms",
        type=int,
        default=100,
        help="How often the camera UI checks record.json for VLM results.",
    )

    return parser.parse_args()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"


def timestamp_values() -> tuple[str, str]:
    now = datetime.now().astimezone()
    file_stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    iso_stamp = now.isoformat(timespec="milliseconds")
    return file_stamp, iso_stamp


def set_status(state: UIState, message: str, seconds: float = 2.5) -> None:
    state.status_message = message
    state.status_until = time.monotonic() + seconds
    print(message)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")
        f.flush()


def rel_posix(path: Path, root: Path) -> str:
    """Portable JSON path: always use forward slashes."""
    return path.relative_to(root).as_posix()


def write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError(f"Failed to save JPG: {path}")


def try_open_camera(index: int) -> tuple[cv2.VideoCapture | None, str]:
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY),
    ]

    for backend_name, backend in backends:
        cap = cv2.VideoCapture(index, backend)

        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, backend_name

        cap.release()

    return None, "Unavailable"


def scan_cameras() -> None:
    print("Scanning camera indices 0-9...")
    found: list[int] = []

    for index in range(10):
        cap, backend = try_open_camera(index)

        if cap is None:
            continue

        ok, frame = cap.read()

        if ok and frame is not None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(
                f"  Camera {index}: available, "
                f"{w}x{h}, FPS={fps:.1f}, backend={backend}"
            )
            found.append(index)

        cap.release()

    print(
        f"\nAvailable camera indices: {found}"
        if found
        else "\nNo usable camera found."
    )


def clamp_box(
    coords: Any,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in coords)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def normalize_drag_box(
    start: tuple[int, int],
    end: tuple[int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(start[0], end[0]))
    y1 = max(0, min(start[1], end[1]))
    x2 = min(frame_width, max(start[0], end[0]))
    y2 = min(frame_height, max(start[1], end[1]))

    if x2 - x1 < MIN_DRAG_PIXELS or y2 - y1 < MIN_DRAG_PIXELS:
        return None

    return x1, y1, x2, y2


def expand_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    mx = int(round(bw * max(0.0, margin_ratio)))
    my = int(round(bh * max(0.0, margin_ratio)))

    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(w, x2 + mx),
        min(h, y2 + my),
    )


def box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_area_ratio(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> float:
    h, w = frame_shape[:2]
    return box_area(box) / float(max(1, w * h))


def point_inside_box(
    x: int,
    y: int,
    box: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_dict(box: tuple[int, int, int, int]) -> dict[str, int]:
    x1, y1, x2, y2 = box
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def normalized_bbox(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> dict[str, float]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box

    return {
        "x1": round(x1 / w, 6),
        "y1": round(y1 / h, 6),
        "x2": round(x2 / w, 6),
        "y2": round(y2 / h, 6),
    }


def resize_for_llm(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)

    # Never upscale.
    if longest <= max_side:
        return image.copy()

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def mask_from_polygon(
    frame_shape: tuple[int, ...],
    polygon: np.ndarray | None,
) -> np.ndarray | None:
    if polygon is None or len(polygon) < 3:
        return None

    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    poly = np.asarray(polygon, dtype=np.float32).copy()
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    poly = np.round(poly).astype(np.int32)

    cv2.fillPoly(mask, [poly], 255)
    return mask


def save_masked_crop(
    path: Path,
    frame: np.ndarray,
    full_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> tuple[int, int]:
    x1, y1, x2, y2 = crop_box

    crop_bgr = frame[y1:y2, x1:x2].copy()
    crop_mask = full_mask[y1:y2, x1:x2].copy()

    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = crop_mask

    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), rgba):
        raise RuntimeError(f"Failed to save masked PNG: {path}")

    return int(np.count_nonzero(crop_mask)), int(crop_mask.size)


def update_seen_streak(state: UIState, raw_ids: set[int]) -> None:
    for track_id in list(state.seen_streak):
        if track_id not in raw_ids:
            state.seen_streak[track_id] = 0

    for track_id in raw_ids:
        state.seen_streak[track_id] = state.seen_streak.get(track_id, 0) + 1


def extract_tracks(
    result: Any,
    frame_shape: tuple[int, ...],
    track_to_display: dict[int, int],
    next_display_id: int,
    state: UIState,
    min_track_age: int,
    max_box_area_ratio: float,
) -> tuple[list[AutoTrack], int]:
    boxes = result.boxes

    if boxes is None or len(boxes) == 0 or boxes.id is None:
        update_seen_streak(state, set())
        return [], next_display_id

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    track_ids = boxes.id.detach().cpu().numpy().astype(int)

    polygons = result.masks.xy if result.masks is not None else None

    raw_ids = {int(tid) for tid in track_ids.tolist()}
    update_seen_streak(state, raw_ids)

    tracks: list[AutoTrack] = []

    for index, (coords, confidence, class_id, track_id) in enumerate(
        zip(xyxy, confidences, class_ids, track_ids)
    ):
        box = clamp_box(coords, frame_shape)

        if box is None:
            continue

        tid = int(track_id)

        # Hide obvious scene-level proposals unless already selected.
        if (
            box_area_ratio(box, frame_shape) > max_box_area_ratio
            and tid not in state.selected_auto_ids
        ):
            continue

        # Do not clutter UI with one-frame proposals.
        if (
            state.seen_streak.get(tid, 0) < min_track_age
            and tid not in state.selected_auto_ids
        ):
            continue

        if tid not in track_to_display:
            track_to_display[tid] = next_display_id
            next_display_id += 1

        polygon = None

        if polygons is not None and index < len(polygons):
            polygon = np.asarray(polygons[index], dtype=np.float32).copy()

        tracks.append(
            AutoTrack(
                box=box,
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=str(result.names[int(class_id)]),
                track_id=tid,
                display_id=track_to_display[tid],
                mask_polygon=polygon,
            )
        )

    tracks.sort(key=lambda item: item.display_id)
    return tracks, next_display_id


def find_smallest_auto_at(
    tracks: list[AutoTrack],
    x: int,
    y: int,
) -> AutoTrack | None:
    candidates = [
        item for item in tracks
        if point_inside_box(x, y, item.box)
    ]

    return min(
        candidates,
        key=lambda item: box_area(item.box),
        default=None,
    )


def find_smallest_manual_at(
    rois: list[ManualROI],
    x: int,
    y: int,
) -> ManualROI | None:
    candidates = [
        item for item in rois
        if point_inside_box(x, y, item.box)
    ]

    return min(
        candidates,
        key=lambda item: box_area(item.box),
        default=None,
    )


def ensure_auto_object_id(state: UIState, track_id: int) -> str:
    object_id = state.track_object_ids.get(track_id)

    if object_id is None:
        object_id = new_id("obj")
        state.track_object_ids[track_id] = object_id

    return object_id


def on_mouse(
    event: int,
    x: int,
    y: int,
    flags: int,
    state: UIState,
) -> None:
    del flags

    if event == cv2.EVENT_LBUTTONDOWN:
        state.drag_start = (x, y)
        state.drag_current = (x, y)
        state.is_dragging = True
        return

    if event == cv2.EVENT_MOUSEMOVE and state.is_dragging:
        state.drag_current = (x, y)
        return

    if event != cv2.EVENT_LBUTTONUP or not state.is_dragging:
        return

    start = state.drag_start
    state.drag_current = (x, y)
    state.is_dragging = False

    if start is None:
        return

    distance = math.hypot(x - start[0], y - start[1])

    # Drag = manual ROI.
    if distance >= MIN_DRAG_PIXELS:
        box = normalize_drag_box(
            start,
            (x, y),
            state.frame_width,
            state.frame_height,
        )

        if box is None:
            set_status(state, "Manual ROI is too small.")
            return

        roi = ManualROI(
            roi_id=state.next_manual_id,
            object_id=new_id("obj"),
            box=box,
        )

        state.next_manual_id += 1
        state.manual_rois.append(roi)
        state.selected_manual_ids.add(roi.roi_id)

        set_status(
            state,
            f"Manual ROI M{roi.roi_id} registered and selected.",
        )

        if state.query_on_select and roi.object_id not in state.result_cards_by_object:
            state.pending_manual_query_ids.add(roi.roi_id)

        return

    # Short-click: automatic proposal first.
    auto_target = find_smallest_auto_at(state.current_tracks, x, y)

    if auto_target is not None:
        if auto_target.track_id in state.selected_auto_ids:
            state.selected_auto_ids.remove(auto_target.track_id)
            state.lost_frame_counts.pop(auto_target.track_id, None)

            set_status(
                state,
                f"A{auto_target.display_id} deselected.",
            )
        else:
            object_id = ensure_auto_object_id(
                state,
                auto_target.track_id,
            )

            state.selected_auto_ids.add(auto_target.track_id)
            state.lost_frame_counts[auto_target.track_id] = 0

            set_status(
                state,
                f"A{auto_target.display_id} selected "
                f"({object_id[:12]}...).",
            )

            if (
                state.query_on_select
                and object_id not in state.result_cards_by_object
            ):
                state.pending_auto_query_ids.add(auto_target.track_id)

        return

    # If no auto box clicked, toggle a manual ROI.
    manual_target = find_smallest_manual_at(state.manual_rois, x, y)

    if manual_target is not None:
        if manual_target.roi_id in state.selected_manual_ids:
            state.selected_manual_ids.remove(manual_target.roi_id)
            set_status(state, f"M{manual_target.roi_id} deselected.")
        else:
            state.selected_manual_ids.add(manual_target.roi_id)
            set_status(state, f"M{manual_target.roi_id} selected.")


def source_metadata(
    args: argparse.Namespace,
    frame: np.ndarray,
) -> dict[str, Any]:
    h, w = frame.shape[:2]

    return {
        "device_type": "desktop_webcam",
        "device_name": "prototype_desktop",
        "camera_id": f"camera_{args.camera}",
        "frame_width": w,
        "frame_height": h,
    }


def detector_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": Path(args.model).name,
        "family": "YOLOE-26",
        "mode": "prompt_free_open_vocabulary_instance_segmentation",
        "inference_size": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "precision": "fp16" if torch.cuda.is_available() else "fp32",
        "semantic_label_policy": "hint_only_not_ground_truth",
    }


def create_capture_directory(
    output_root: Path,
    object_id: str,
    capture_id: str,
    file_stamp: str,
) -> Path:
    # Simple object -> capture structure.
    folder = (
        output_root
        / "objects"
        / object_id
        / f"{file_stamp}_{capture_id}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_images(
    *,
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    full_mask: np.ndarray | None,
    args: argparse.Namespace,
    output_root: Path,
    capture_dir: Path,
) -> tuple[
    tuple[int, int, int, int],
    dict[str, Any],
    dict[str, Any] | None,
]:
    expanded = expand_box(box, frame.shape, args.margin)
    x1, y1, x2, y2 = expanded

    original_crop = frame[y1:y2, x1:x2].copy()

    if original_crop.size == 0:
        raise RuntimeError("Empty capture crop.")

    original_path = capture_dir / "original.jpg"
    write_jpg(
        original_path,
        original_crop,
        args.jpeg_quality,
    )

    llm_image = resize_for_llm(
        original_crop,
        args.llm_max_side,
    )

    llm_path = capture_dir / "llm.jpg"
    write_jpg(
        llm_path,
        llm_image,
        args.llm_jpeg_quality,
    )

    image_metadata: dict[str, Any] = {
        "original": {
            "file": rel_posix(original_path, output_root),
            "width": int(original_crop.shape[1]),
            "height": int(original_crop.shape[0]),
            "format": "jpg",
            "jpeg_quality": args.jpeg_quality,
        },
        "llm": {
            "file": rel_posix(llm_path, output_root),
            "width": int(llm_image.shape[1]),
            "height": int(llm_image.shape[0]),
            "format": "jpg",
            "jpeg_quality": args.llm_jpeg_quality,
            "max_side_policy": args.llm_max_side,
            "upscaled": False,
        },
    }

    segmentation = None

    if full_mask is not None:
        mask_crop = full_mask[y1:y2, x1:x2].copy()

        mask_path = capture_dir / "mask.png"

        if not cv2.imwrite(str(mask_path), mask_crop):
            raise RuntimeError(f"Failed to save mask: {mask_path}")

        masked_path = capture_dir / "masked.png"

        nonzero, total = save_masked_crop(
            masked_path,
            frame,
            full_mask,
            expanded,
        )

        segmentation = {
            "available": True,
            "mask_file": rel_posix(mask_path, output_root),
            "masked_file": rel_posix(masked_path, output_root),
            "mask_area_px": nonzero,
            "crop_area_px": total,
            "mask_coverage_ratio": round(
                nonzero / max(1, total),
                6,
            ),
        }

    return expanded, image_metadata, segmentation


def enqueue_record(
    *,
    output_root: Path,
    record_path: Path,
    request_id: str,
    object_id: str,
    capture_id: str,
    iso_stamp: str,
) -> None:
    # Only a lightweight pointer is appended.
    # No duplicate request JSON is created.
    append_jsonl(
        output_root / "queue.jsonl",
        {
            "request_id": request_id,
            "object_id": object_id,
            "capture_id": capture_id,
            "record": rel_posix(record_path, output_root),
            "status": "pending",
            "queued_at": iso_stamp,
        },
    )


def build_llm_section(
    *,
    request_id: str,
    yolo_hint: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "pending",
        "request": {
            "request_id": request_id,
            "task": "identify_object",
            "yolo_hint": yolo_hint,
            "requested_fields": [
                "recognition_status",
                "object_name",
                "category",
                "brand",
                "product_name",
                "model",
                "self_reported_confidence",
                "visual_evidence",
                "alternatives",
            ],
        },
        "response": None,
        "harness": {
            "provider": None,
            "model": None,
            "tier": None,
            "retry_count": 0,
            "schema_valid": None,
            "latency_ms": None,
            "cross_model_agreement": None,
        },
    }


def save_auto_capture(
    *,
    frame: np.ndarray,
    track: AutoTrack,
    state: UIState,
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    object_id = ensure_auto_object_id(state, track.track_id)
    capture_id = new_id("cap")
    request_id = new_id("req")
    file_stamp, iso_stamp = timestamp_values()

    capture_dir = create_capture_directory(
        output_root,
        object_id,
        capture_id,
        file_stamp,
    )

    full_mask = mask_from_polygon(
        frame.shape,
        track.mask_polygon,
    )

    expanded, image_meta, segmentation = save_images(
        frame=frame,
        box=track.box,
        full_mask=full_mask,
        args=args,
        output_root=output_root,
        capture_dir=capture_dir,
    )

    yolo_hint = {
        "class_name": track.class_name,
        "confidence": round(track.confidence, 6),
        "policy": "hint_only_not_ground_truth",
        "instruction": (
            "Detector label is a hint only. "
            "The Vision LLM must judge the image independently."
        ),
    }

    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "object_capture",
        "object_id": object_id,
        "capture_id": capture_id,
        "request_id": request_id,
        "timestamp": iso_stamp,

        "source": source_metadata(args, frame),

        "selection": {
            "method": "yoloe_prompt_free",
            "track_id": track.track_id,
            "display_id": track.display_id,
            "tracking_status": "TRACKING",
            "bbox_px": bbox_dict(track.box),
            "bbox_normalized": normalized_bbox(
                track.box,
                frame.shape,
            ),
            "bbox_with_margin_px": bbox_dict(expanded),
            "crop_margin_ratio": args.margin,
        },

        "detector": {
            **detector_metadata(args),
            "class_id": track.class_id,
            "label_hint": track.class_name,
            "confidence": round(track.confidence, 6),
        },

        "segmentation": segmentation,

        "image": image_meta,

        "llm": build_llm_section(
            request_id=request_id,
            yolo_hint=yolo_hint,
        ),
    }

    record_path = capture_dir / "record.json"
    write_json(record_path, record)

    enqueue_record(
        output_root=output_root,
        record_path=record_path,
        request_id=request_id,
        object_id=object_id,
        capture_id=capture_id,
        iso_stamp=iso_stamp,
    )

    return record_path


def save_manual_capture(
    *,
    frame: np.ndarray,
    roi: ManualROI,
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    capture_id = new_id("cap")
    request_id = new_id("req")
    file_stamp, iso_stamp = timestamp_values()

    capture_dir = create_capture_directory(
        output_root,
        roi.object_id,
        capture_id,
        file_stamp,
    )

    expanded, image_meta, _ = save_images(
        frame=frame,
        box=roi.box,
        full_mask=None,
        args=args,
        output_root=output_root,
        capture_dir=capture_dir,
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "object_capture",
        "object_id": roi.object_id,
        "capture_id": capture_id,
        "request_id": request_id,
        "timestamp": iso_stamp,

        "source": source_metadata(args, frame),

        "selection": {
            "method": "manual_roi",
            "track_id": None,
            "display_id": None,
            "manual_roi_id": roi.roi_id,
            "tracking_status": "NOT_TRACKED",
            "bbox_px": bbox_dict(roi.box),
            "bbox_normalized": normalized_bbox(
                roi.box,
                frame.shape,
            ),
            "bbox_with_margin_px": bbox_dict(expanded),
            "crop_margin_ratio": args.margin,
        },

        "detector": None,
        "segmentation": None,
        "image": image_meta,

        "llm": build_llm_section(
            request_id=request_id,
            yolo_hint=None,
        ),
    }

    record_path = capture_dir / "record.json"
    write_json(record_path, record)

    enqueue_record(
        output_root=output_root,
        record_path=record_path,
        request_id=request_id,
        object_id=roi.object_id,
        capture_id=capture_id,
        iso_stamp=iso_stamp,
    )

    return record_path



def read_json_safe(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def register_result_card(
    *,
    state: UIState,
    record_path: Path,
    track_id: int | None = None,
    manual_roi_id: int | None = None,
) -> ResultCard | None:
    record = read_json_safe(record_path)
    if not record:
        return None

    card = ResultCard(
        object_id=str(record["object_id"]),
        request_id=str(record["request_id"]),
        capture_id=str(record["capture_id"]),
        record_path=record_path.resolve(),
        track_id=track_id,
        manual_roi_id=manual_roi_id,
        status=str(record.get("llm", {}).get("status", "pending")),
        response=record.get("llm", {}).get("response"),
        harness=record.get("llm", {}).get("harness"),
    )

    try:
        card.mtime_ns = card.record_path.stat().st_mtime_ns
    except OSError:
        pass

    # This assignment is the stale-response protection for the UI:
    # only the latest request registered for this object is renderable.
    state.result_cards_by_object[card.object_id] = card
    return card


def poll_result_cards(
    *,
    state: UIState,
    poll_ms: int,
) -> None:
    now = time.monotonic()
    interval = max(0.03, poll_ms / 1000.0)

    if now - state.last_result_poll < interval:
        return

    state.last_result_poll = now

    for object_id, card in list(state.result_cards_by_object.items()):
        try:
            mtime_ns = card.record_path.stat().st_mtime_ns
        except OSError:
            continue

        if mtime_ns == card.mtime_ns and card.status not in {
            "pending", "processing"
        }:
            continue

        record = read_json_safe(card.record_path)
        if not record:
            continue

        # Never attach an old response to a newer card.
        if str(record.get("request_id")) != card.request_id:
            continue

        card.mtime_ns = mtime_ns
        llm = record.get("llm", {})
        card.status = str(llm.get("status", "pending"))
        response = llm.get("response")
        card.response = response if isinstance(response, dict) else None

        harness = llm.get("harness")
        card.harness = harness if isinstance(harness, dict) else None


def process_pending_queries(
    *,
    frame: np.ndarray,
    state: UIState,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    if state.pending_auto_query_ids:
        active_by_id = {
            item.track_id: item
            for item in state.current_tracks
        }

        for track_id in list(state.pending_auto_query_ids):
            track = active_by_id.get(track_id)
            if track is None:
                continue

            try:
                record_path = save_auto_capture(
                    frame=frame,
                    track=track,
                    state=state,
                    args=args,
                    output_root=output_root,
                )
                card = register_result_card(
                    state=state,
                    record_path=record_path,
                    track_id=track_id,
                )
                state.pending_auto_query_ids.discard(track_id)

                if card is not None:
                    set_status(
                        state,
                        f"A{track.display_id}: VLM request queued.",
                        seconds=1.8,
                    )
            except Exception as exc:
                state.pending_auto_query_ids.discard(track_id)
                set_status(
                    state,
                    f"A{track.display_id}: query failed to queue: {exc}",
                    seconds=4.0,
                )

    if state.pending_manual_query_ids:
        manual_by_id = {
            item.roi_id: item
            for item in state.manual_rois
        }

        for roi_id in list(state.pending_manual_query_ids):
            roi = manual_by_id.get(roi_id)
            if roi is None:
                state.pending_manual_query_ids.discard(roi_id)
                continue

            try:
                record_path = save_manual_capture(
                    frame=frame,
                    roi=roi,
                    args=args,
                    output_root=output_root,
                )
                card = register_result_card(
                    state=state,
                    record_path=record_path,
                    manual_roi_id=roi_id,
                )
                state.pending_manual_query_ids.discard(roi_id)

                if card is not None:
                    set_status(
                        state,
                        f"M{roi_id}: VLM request queued.",
                        seconds=1.8,
                    )
            except Exception as exc:
                state.pending_manual_query_ids.discard(roi_id)
                set_status(
                    state,
                    f"M{roi_id}: query failed to queue: {exc}",
                    seconds=4.0,
                )


def _short_text(value: Any, limit: int = 38) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


def result_card_lines(card: ResultCard) -> tuple[str, list[str], str]:
    status = card.status.lower()

    if status == "pending":
        return "QUEUED", ["Waiting for VLM worker..."], "WAIT"

    if status == "processing":
        return "ANALYZING", ["Identifying selected object..."], "WORK"

    if status == "timeout":
        return "VLM TIMEOUT", ["Identification timed out."], "ERROR"

    if status == "failed":
        return "VLM ERROR", ["Identification failed."], "ERROR"

    if status == "cancelled":
        return "REQUEST CANCELLED", ["A newer request replaced this one."], "STALE"

    response = card.response or {}
    recognition = str(response.get("recognition_status", "")).lower()

    if status == "unknown" or recognition == "unknown":
        return "UNABLE TO IDENTIFY", ["Not enough visual evidence."], "UNKNOWN"

    object_name = _short_text(response.get("object_name")) or "Object"
    category = _short_text(response.get("category"))
    brand = _short_text(response.get("brand"))
    product = _short_text(response.get("product_name"))
    model = _short_text(response.get("model"))

    details: list[str] = []

    if category:
        details.append(f"Category: {category}")
    if brand:
        details.append(f"Brand: {brand}")
    if product:
        details.append(f"Product: {product}")
    if model:
        details.append(f"Model: {model}")

    # Keep the live HUD compact. Detailed evidence remains in record.json.
    details = details[:4]

    if status == "uncertain" or recognition == "uncertain":
        return f"LIKELY: {object_name}", details or ["Identification uncertain."], "UNCERTAIN"

    return object_name, details, "IDENTIFIED"


def draw_info_card(
    *,
    frame: np.ndarray,
    anchor_box: tuple[int, int, int, int],
    card: ResultCard | None,
    waiting_for_capture: bool,
) -> None:
    if card is None and not waiting_for_capture:
        return

    x1, y1, x2, y2 = anchor_box
    frame_h, frame_w = frame.shape[:2]

    if card is None:
        title = "SELECTED"
        lines = ["Press C to identify."]
        badge = "READY"
    else:
        title, lines, badge = result_card_lines(card)

    card_w = min(360, max(270, frame_w // 4))
    header_h = 30
    line_h = 22
    padding = 12
    card_h = header_h + padding + 28 + max(1, len(lines)) * line_h + padding

    gap = 16

    if x2 + gap + card_w <= frame_w - 8:
        cx = x2 + gap
        connector_target_x = cx
        connector_source_x = x2
    else:
        cx = max(8, x1 - gap - card_w)
        connector_target_x = cx + card_w
        connector_source_x = x1

    cy = max(8, min(y1 - 4, frame_h - card_h - 8))

    connector_y = max(y1, min((y1 + y2) // 2, y2))
    target_y = cy + header_h // 2

    # HUD connector line.
    cv2.line(
        frame,
        (connector_source_x, connector_y),
        (connector_target_x, target_y),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (connector_source_x, connector_y),
        3,
        (0, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    # Semi-transparent dark panel.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (cx, cy),
        (cx + card_w, cy + card_h),
        (10, 14, 18),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        dst=frame,
    )

    # Thin HUD border/header.
    cv2.rectangle(
        frame,
        (cx, cy),
        (cx + card_w, cy + card_h),
        (90, 110, 120),
        1,
    )
    cv2.line(
        frame,
        (cx, cy + header_h),
        (cx + card_w, cy + header_h),
        (0, 255, 255),
        1,
    )

    cv2.putText(
        frame,
        "AI IDENTIFICATION",
        (cx + 10, cy + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    badge_size = cv2.getTextSize(
        badge,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        1,
    )[0]

    cv2.putText(
        frame,
        badge,
        (cx + card_w - badge_size[0] - 10, cy + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 230, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        _short_text(title, 34),
        (cx + 12, cy + header_h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    ty = cy + header_h + 53
    for line in lines:
        cv2.putText(
            frame,
            _short_text(line, 46),
            (cx + 12, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (205, 215, 220),
            1,
            cv2.LINE_AA,
        )
        ty += line_h


def draw_result_cards(
    *,
    frame: np.ndarray,
    state: UIState,
) -> None:
    if not state.show_result_cards:
        return

    for track in state.current_tracks:
        if track.track_id not in state.selected_auto_ids:
            continue

        object_id = state.track_object_ids.get(track.track_id)
        if not object_id:
            continue

        card = state.result_cards_by_object.get(object_id)

        draw_info_card(
            frame=frame,
            anchor_box=track.box,
            card=card,
            waiting_for_capture=(not state.query_on_select and card is None),
        )

    for roi in state.manual_rois:
        if roi.roi_id not in state.selected_manual_ids:
            continue

        card = state.result_cards_by_object.get(roi.object_id)

        draw_info_card(
            frame=frame,
            anchor_box=roi.box,
            card=card,
            waiting_for_capture=(not state.query_on_select and card is None),
        )


def append_lost_event(
    *,
    output_root: Path,
    object_id: str,
    track_id: int,
    display_id: int,
) -> None:
    _, iso_stamp = timestamp_values()

    append_jsonl(
        output_root / "events.jsonl",
        {
            "event_id": new_id("evt"),
            "timestamp": iso_stamp,
            "object_id": object_id,
            "event": "LOST",
            "previous_track_id": track_id,
            "display_id": display_id,
            "reason": "target_missing_beyond_buffer",
            "automatic_reassignment": False,
            "user_reselection_required": True,
        },
    )


def update_lost_targets(
    state: UIState,
    display_id_by_track: dict[int, int],
    lost_after_frames: int,
    output_root: Path,
) -> None:
    active_ids = {
        item.track_id
        for item in state.current_tracks
    }

    lost_now: list[int] = []

    for track_id in list(state.selected_auto_ids):
        if track_id in active_ids:
            state.lost_frame_counts[track_id] = 0
            continue

        count = state.lost_frame_counts.get(track_id, 0) + 1
        state.lost_frame_counts[track_id] = count

        if count > lost_after_frames:
            lost_now.append(track_id)

    for track_id in lost_now:
        display_id = display_id_by_track.get(
            track_id,
            track_id,
        )

        object_id = state.track_object_ids.get(track_id)

        if object_id is not None:
            append_lost_event(
                output_root=output_root,
                object_id=object_id,
                track_id=track_id,
                display_id=display_id,
            )

        state.selected_auto_ids.discard(track_id)
        state.lost_frame_counts.pop(track_id, None)

        # Conservative identity policy:
        # after LOST, user must select again and gets a new object_id.
        state.track_object_ids.pop(track_id, None)

        set_status(
            state,
            f"A{display_id} LOST. User reselection is required.",
            seconds=4.0,
        )


def save_full_frame(
    frame: np.ndarray,
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    file_stamp, iso_stamp = timestamp_values()

    folder = output_root / "full_frame"
    folder.mkdir(parents=True, exist_ok=True)

    image_path = folder / f"{file_stamp}.jpg"
    write_jpg(
        image_path,
        frame,
        args.jpeg_quality,
    )

    metadata_path = folder / f"{file_stamp}.json"

    write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "full_frame_capture",
            "timestamp": iso_stamp,
            "source": source_metadata(args, frame),
            "image": {
                "file": rel_posix(image_path, output_root),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "format": "jpg",
                "jpeg_quality": args.jpeg_quality,
            },
        },
    )

    return image_path


def draw_label(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        2,
    )

    label_top = max(
        0,
        y1 - th - baseline - 7,
    )

    cv2.rectangle(
        frame,
        (x1, label_top),
        (
            min(frame.shape[1] - 1, x1 + tw + 8),
            y1,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 4,
            max(th + 1, y1 - baseline - 3),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def overlay_selected_mask(
    frame: np.ndarray,
    track: AutoTrack,
) -> None:
    full_mask = mask_from_polygon(
        frame.shape,
        track.mask_polygon,
    )

    if full_mask is None:
        return

    overlay = frame.copy()
    overlay[full_mask > 0] = (0, 255, 255)

    cv2.addWeighted(
        overlay,
        0.22,
        frame,
        0.78,
        0,
        dst=frame,
    )


def draw_interface(
    frame: np.ndarray,
    state: UIState,
    smoothed_fps: float,
    display_id_by_track: dict[int, int],
    lost_after_frames: int,
) -> None:
    if state.show_selected_mask:
        for track in state.current_tracks:
            if track.track_id in state.selected_auto_ids:
                overlay_selected_mask(frame, track)

    for track in state.current_tracks:
        selected = track.track_id in state.selected_auto_ids

        color = (
            (0, 255, 255)
            if selected
            else (0, 255, 0)
        )

        label = f"A{track.display_id} {track.confidence:.2f}"

        if state.show_detector_label:
            label += f" {track.class_name}"

        draw_label(
            frame,
            track.box,
            label,
            color,
            3 if selected else 2,
        )

    for roi in state.manual_rois:
        selected = roi.roi_id in state.selected_manual_ids

        color = (
            (255, 0, 255)
            if selected
            else (255, 180, 0)
        )

        draw_label(
            frame,
            roi.box,
            f"M{roi.roi_id} manual ROI",
            color,
            3 if selected else 2,
        )

    draw_result_cards(
        frame=frame,
        state=state,
    )

    if (
        state.is_dragging
        and state.drag_start is not None
        and state.drag_current is not None
    ):
        preview = normalize_drag_box(
            state.drag_start,
            state.drag_current,
            max(1, state.frame_width),
            max(1, state.frame_height),
        )

        if preview is not None:
            cv2.rectangle(
                frame,
                (preview[0], preview[1]),
                (preview[2], preview[3]),
                (255, 255, 0),
                2,
            )

    cv2.putText(
        frame,
        (
            f"FPS {smoothed_fps:.1f} | Proposals {len(state.current_tracks)} | "
            f"Selected detected {len(state.selected_auto_ids)} / "
            f"ROI {len(state.selected_manual_ids)}"
        ),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Click: select + identify | Drag: ROI + identify | C: re-identify",
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "R: reset | S: full frame | L: label | V: mask | I: info card | Q/ESC: quit",
        (12, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y = 107

    for track_id in sorted(state.selected_auto_ids):
        lost_count = state.lost_frame_counts.get(track_id, 0)

        if lost_count <= 0:
            continue

        display_id = display_id_by_track.get(
            track_id,
            track_id,
        )

        cv2.putText(
            frame,
            (
                f"A{display_id}: TEMPORARILY_LOST "
                f"{lost_count}/{lost_after_frames}"
            ),
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        y += 24

    if (
        state.status_message
        and time.monotonic() < state.status_until
    ):
        cv2.putText(
            frame,
            state.status_message,
            (12, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def reset_state(state: UIState) -> None:
    state.manual_rois.clear()
    state.selected_auto_ids.clear()
    state.selected_manual_ids.clear()
    state.track_object_ids.clear()
    state.lost_frame_counts.clear()
    state.pending_auto_query_ids.clear()
    state.pending_manual_query_ids.clear()
    state.result_cards_by_object.clear()
    state.next_manual_id = 1


def main() -> None:
    args = parse_args()

    if args.scan:
        scan_cameras()
        return

    tracker_path = Path(args.tracker)

    if not tracker_path.exists():
        raise FileNotFoundError(
            f"Tracker config not found: {tracker_path.resolve()}"
        )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    device: int | str = 0 if cuda_available else "cpu"
    precision = "fp16" if cuda_available else "fp32"

    print(f"Model: {args.model}")
    print(f"Inference: {args.imgsz}px / conf={args.conf}")
    print(f"Tracker: {tracker_path.resolve()}")
    print(f"Output root: {output_root.resolve()}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Precision: FP16")
    else:
        print("CUDA GPU not detected. CPU FP32 will be used.")

    model = YOLO(args.model)

    cap, backend = try_open_camera(args.camera)

    if cap is None:
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Run --scan or close other camera applications."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(
        f"Camera {args.camera} opened with {backend}: "
        f"{actual_width}x{actual_height}, FPS={actual_fps:.1f}"
    )

    state = UIState()
    state.query_on_select = args.query_on_select

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(
        WINDOW_NAME,
        on_mouse,
        state,
    )

    track_to_display: dict[int, int] = {}
    next_display_id = 1
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("Failed to read camera frame.")
                break

            state.frame_height, state.frame_width = frame.shape[:2]
            original_frame = frame.copy()

            start = time.perf_counter()

            results = model.track(
                source=original_frame,
                persist=True,
                tracker=str(tracker_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                quantize=precision,
                max_det=args.max_det,
                verbose=False,
            )

            state.current_tracks, next_display_id = extract_tracks(
                result=results[0],
                frame_shape=original_frame.shape,
                track_to_display=track_to_display,
                next_display_id=next_display_id,
                state=state,
                min_track_age=args.min_track_age,
                max_box_area_ratio=args.max_box_area_ratio,
            )

            display_id_by_track = {
                tid: did
                for tid, did in track_to_display.items()
            }

            update_lost_targets(
                state=state,
                display_id_by_track=display_id_by_track,
                lost_after_frames=args.lost_after,
                output_root=output_root,
            )

            # Auto query is intentionally handled in the camera loop rather than
            # inside the mouse callback so the capture uses a current clean frame.
            process_pending_queries(
                frame=original_frame,
                state=state,
                args=args,
                output_root=output_root,
            )

            # Non-blocking record.json polling. The VLM worker remains a separate
            # process; this loop only reads small JSON files at a low frequency.
            poll_result_cards(
                state=state,
                poll_ms=args.result_poll_ms,
            )

            elapsed = time.perf_counter() - start
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0

            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else smoothed_fps * 0.90 + instant_fps * 0.10
            )

            display_frame = original_frame.copy()

            draw_interface(
                frame=display_frame,
                state=state,
                smoothed_fps=smoothed_fps,
                display_id_by_track=display_id_by_track,
                lost_after_frames=args.lost_after,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("l"), ord("L")):
                state.show_detector_label = not state.show_detector_label
                set_status(
                    state,
                    "Detector labels ON."
                    if state.show_detector_label
                    else "Detector labels OFF.",
                )
                continue

            if key in (ord("v"), ord("V")):
                state.show_selected_mask = not state.show_selected_mask
                set_status(
                    state,
                    "Selected mask preview ON."
                    if state.show_selected_mask
                    else "Selected mask preview OFF.",
                )
                continue

            if key in (ord("i"), ord("I")):
                state.show_result_cards = not state.show_result_cards
                set_status(
                    state,
                    "VLM info cards ON."
                    if state.show_result_cards
                    else "VLM info cards OFF.",
                )
                continue

            if key in (ord("r"), ord("R")):
                reset_state(state)
                set_status(
                    state,
                    "Selections and manual ROIs reset.",
                )
                continue

            if key in (ord("s"), ord("S")):
                path = save_full_frame(
                    original_frame,
                    args,
                    output_root,
                )

                set_status(
                    state,
                    f"Full frame saved: {path.name}",
                )
                continue

            if key not in (ord("c"), ord("C")):
                continue

            active_by_id = {
                item.track_id: item
                for item in state.current_tracks
            }

            manual_by_id = {
                item.roi_id: item
                for item in state.manual_rois
            }

            saved = 0
            skipped = 0

            for track_id in sorted(state.selected_auto_ids):
                track = active_by_id.get(track_id)

                if track is None:
                    skipped += 1
                    continue

                record_path = save_auto_capture(
                    frame=original_frame,
                    track=track,
                    state=state,
                    args=args,
                    output_root=output_root,
                )

                register_result_card(
                    state=state,
                    record_path=record_path,
                    track_id=track_id,
                )
                print(f"Record: {record_path}")
                saved += 1

            for roi_id in sorted(state.selected_manual_ids):
                roi = manual_by_id.get(roi_id)

                if roi is None:
                    skipped += 1
                    continue

                record_path = save_manual_capture(
                    frame=original_frame,
                    roi=roi,
                    args=args,
                    output_root=output_root,
                )

                register_result_card(
                    state=state,
                    record_path=record_path,
                    manual_roi_id=roi_id,
                )
                print(f"Record: {record_path}")
                saved += 1

            if saved == 0:
                set_status(
                    state,
                    "No visible selected target. "
                    "Select a proposal or drag an ROI first.",
                )
            else:
                msg = f"Queued {saved} identification request(s)."

                if skipped:
                    msg += f" Skipped {skipped} unavailable target(s)."

                set_status(state, msg)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and window closed.")


if __name__ == "__main__":
    main()
