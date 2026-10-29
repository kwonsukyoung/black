import base64
import math
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionConfig:
    """블랙박스 접촉·스크래치·근접 체류 정밀 감지 설정"""
    # 내 차 왼쪽 뒷 필러 등 집중 감시 구역 (정규화 0.0 ~ 1.0)
    roi_x: float = 0.0
    roi_y: float = 0.12
    roi_w: float = 0.55
    roi_h: float = 0.80
    # 초당 분석할 프레임 수 (궤적 및 감속 추적을 위해 기본 5 FPS)
    sample_fps: float = 5.0
    # 감도 (1 ~ 100)
    sensitivity: int = 65
    # 이벤트 병합 간격(초)
    merge_gap_sec: float = 2.0
    # 단순 통과 차량 자동 제외 여부
    exclude_pass_through: bool = True
    # 최소 체류 시간(초) - 이보다 짧게 휙 지나가면서 차체 흔들림도 없으면 단순 통과로 판정
    min_dwell_sec: float = 2.2
    # 차체 미세 진동(충격) 감지 활성화
    detect_shake: bool = True


@dataclass
class DetectedEvent:
    """감지된 단일 의심 장면 정보"""
    event_id: str
    file_name: str
    file_index: int
    start_time: float
    end_time: float
    peak_time: float
    duration: float
    score: float                  # 진짜 접촉/스크래치 의심도 (0 ~ 99)
    category: str                 # "impact" | "scrape_maneuver" | "person_linger" | "pass_through"
    label: str                    # 화면 표시용 분류명
    reason: str                   # 왜 이 점수가 나왔는지 구체적 판정 근거
    dwell_sec: float              # 감시 구역 내 체류 시간(초)
    jitter_px: float              # 정적 배경 기준 실제 차체 흔들림(픽셀)
    deceleration_ratio: float     # 근접 시 감속/방향전환 지수
    bbox: Dict[str, float]
    thumbnail_data_url: str       # 피크 시점 + 이동 궤적 스냅샷
    strip_data_url: str           # [접근 -> 밀착 피크 -> 직후] 3분할 비교 스냅샷


def _compute_static_bg_jitter(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    motion_mask: np.ndarray,
) -> float:
    """
    지나가는 차량 자체의 움직임을 제외하고,
    화면 상단/원경의 '정지된 배경(기둥, 건물, 주차선 등)'이 순간적으로 흔들렸는지 측정합니다.
    (차가 단순히 지나갈 때는 0에 수렴하고, 내 차가 긁히거나 충격받을 때만 상승합니다.)
    """
    h, w = prev_gray.shape[:2]
    # 상단 시계 자막(0~8%)은 제외하고 8%~45% 구간의 정적 배경 영역 중 움직임 마스크가 없는 곳만 추출
    y1, y2 = int(h * 0.08), int(h * 0.55)
    bg_prev = prev_gray[y1:y2, :]
    bg_curr = curr_gray[y1:y2, :]
    bg_mask = motion_mask[y1:y2, :]

    # 움직이는 물체가 화면 대부분을 가린 경우 오탐 방지를 위해 진동 0 반환
    static_ratio = float(np.count_nonzero(bg_mask == 0)) / float(bg_mask.size + 1e-5)
    if static_ratio < 0.45:
        return 0.0

    # 정적 영역에서 코너 특징점 추출 후 옵티컬 플로우로 카메라 자체의 미세 이동량(px) 계산
    static_mask = (bg_mask == 0).astype(np.uint8) * 255
    pts = cv2.goodFeaturesToTrack(
        bg_prev,
        maxCorners=40,
        qualityLevel=0.08,
        minDistance=10,
        mask=static_mask,
    )
    if pts is None or len(pts) < 6:
        return 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(bg_prev, bg_curr, pts, None)
    if next_pts is None or status is None:
        return 0.0

    good_old = pts[status.flatten() == 1]
    good_new = next_pts[status.flatten() == 1]
    if len(good_old) < 6:
        return 0.0

    displacements = np.linalg.norm(good_new - good_old, axis=2).flatten()
    # 중앙값(median) 변위가 0.45px ~ 6.0px 사이로 전체 배경이 동시에 이동했을 때만 카메라 진동으로 인정
    med_disp = float(np.median(displacements))
    std_disp = float(np.std(displacements))

    # 전체 배경이 일관되게 흔들린 경우 (std가 너무 크면 노이즈)
    if 0.42 <= med_disp <= 6.5 and std_disp < med_disp * 1.3:
        return round(med_disp, 2)
    return 0.0


def _analyze_trajectory(
    points: List[Tuple[float, float, float, float]]
) -> Tuple[float, bool, float]:
    """
    기록된 궤적 [(t, cx, cy, area_pct), ...] 을 분석하여:
    1) straightness (직선 통과성: 0.0 ~ 1.0, 1에 가까울수록 일정한 방향으로 휙 지나감)
    2) has_stop_or_reverse (감속 후 정지 또는 전후진 방향 전환 여부)
    3) min_speed_ratio (최고 속도 대비 최저 속도 비율 - 낮을수록 차 옆에서 멈칫거림)
    """
    if len(points) < 3:
        return 1.0, False, 1.0

    start_x, start_y = points[0][1], points[0][2]
    end_x, end_y = points[-1][1], points[-1][2]
    net_dist = math.hypot(end_x - start_x, end_y - start_y)

    speeds: List[float] = []
    dx_list: List[float] = []
    dy_list: List[float] = []
    total_path = 0.0

    for i in range(1, len(points)):
        dt = max(0.05, points[i][0] - points[i - 1][0])
        dx = points[i][1] - points[i - 1][1]
        dy = points[i][2] - points[i - 1][2]
        step_dist = math.hypot(dx, dy)
        total_path += step_dist
        speeds.append(step_dist / dt)
        dx_list.append(dx)
        dy_list.append(dy)

    straightness = net_dist / max(0.001, total_path)

    max_spd = max(speeds) if speeds else 0.0
    min_spd = min(speeds) if speeds else 0.0
    min_speed_ratio = (min_spd / max_spd) if max_spd > 0.01 else 0.0

    # 방향 전환(전진 후 후진 등) 또는 중간 정지 감지
    direction_changes = 0
    for i in range(1, len(dx_list)):
        dot = dx_list[i] * dx_list[i - 1] + dy_list[i] * dy_list[i - 1]
        if dot < -0.0004:
            direction_changes += 1

    has_stop_or_reverse = (direction_changes >= 1) or (min_speed_ratio < 0.22 and len(points) >= 5) or (straightness < 0.58)
    return round(straightness, 3), has_stop_or_reverse, round(min_speed_ratio, 3)


def _encode_main_thumbnail(
    frame_bgr: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    bbox_norm: Dict[str, float],
    trajectory: List[Tuple[float, float, float, float]],
    category: str,
) -> str:
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    rx, ry, rw, rh = roi_rect

    # ROI 영역 표시
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 210, 0), 2)

    # 이동 궤적 선 그리기
    if len(trajectory) >= 2:
        pts_px = [
            (int(pt[1] * w), int(pt[2] * h))
            for pt in trajectory
        ]
        for i in range(1, len(pts_px)):
            cv2.line(vis, pts_px[i - 1], pts_px[i], (0, 255, 180), 2, cv2.LINE_AA)
        cv2.circle(vis, pts_px[0], 5, (255, 200, 0), -1)
        cv2.circle(vis, pts_px[-1], 6, (0, 60, 255), -1)

    # 객체 바운딩 박스
    bx = int(bbox_norm["x"] * w)
    by = int(bbox_norm["y"] * h)
    bw = int(bbox_norm["w"] * w)
    bh = int(bbox_norm["h"] * h)
    color = (0, 50, 255) if category == "impact" else (0, 180, 255)
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 3)

    target_w = 480
    if w > target_w:
        scale = target_w / float(w)
        vis = cv2.resize(vis, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_strip_thumbnail(
    first_frame: np.ndarray,
    peak_frame: np.ndarray,
    last_frame: np.ndarray,
) -> str:
    """[접근 전 -> 최대 밀착 순간 -> 이탈 직후] 3분할 비교 이미지를 생성합니다."""
    cell_w, cell_h = 220, 124
    frames = [first_frame, peak_frame, last_frame]
    labels = ["1. 접근", "2. 밀착/피크", "3. 직후"]
    cells = []

    for idx, f in enumerate(frames):
        resized = cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, cell_h - 22), (cell_w, cell_h), (15, 23, 42), -1)
        cv2.putText(
            resized,
            labels[idx],
            (8, cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(resized)

    strip = np.hstack(cells)
    ok, buf = cv2.imencode(".jpg", strip, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def analyze_video_file(
    video_path: str,
    file_name: str,
    file_index: int,
    config: DetectionConfig,
    progress_callback: Optional[Callable[[float, int], None]] = None,
) -> Tuple[List[DetectedEvent], float]:
    """
    단순 주행 통과 차량은 걸러내고,
    1) 실제 차체 흔들림(접촉 충격)
    2) 내 차 옆 감속/정차/전후진 주차 기동(차로 긁고 가는 패턴)
    3) 보행자/물체의 근접 체류(문콕/스크래치 패턴)
    만 정밀 분류하여 반환합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or math.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    frame_step = max(1, int(round(fps / max(2.0, config.sample_fps))))

    diff_thresh = int(np.clip(52 - (config.sensitivity * 0.42), 10, 45))
    min_area_pct = max(0.15, 0.45 * ((110 - config.sensitivity) / 50.0))

    prev_gray_full: Optional[np.ndarray] = None
    bg_avg_roi: Optional[np.ndarray] = None

    raw_segments: List[Dict] = []
    current_seg: Optional[Dict] = None

    frame_idx = 0
    processed_steps = 0
    analysis_width = 480

    while True:
        if frame_idx > 0 and frame_step > 1:
            for _ in range(frame_step - 1):
                if not cap.grab():
                    break
                frame_idx += 1

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp = frame_idx / fps
        frame_idx += 1
        processed_steps += 1

        orig_h, orig_w = frame.shape[:2]
        scale = min(1.0, analysis_width / float(orig_w))
        small = (
            cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else frame
        )
        sh, sw = small.shape[:2]

        rx = int(np.clip(config.roi_x * sw, 0, sw - 12))
        ry = int(np.clip(config.roi_y * sh, 0, sh - 12))
        rw = int(np.clip(config.roi_w * sw, 12, sw - rx))
        rh = int(np.clip(config.roi_h * sh, 12, sh - ry))

        gray_full = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gray_roi = gray_full[ry : ry + rh, rx : rx + rw]

        if prev_gray_full is None or bg_avg_roi is None:
            prev_gray_full = gray_full
            bg_avg_roi = gray_roi.astype("float32")
            continue

        # 1. ROI 내부 움직임 추출
        cv2.accumulateWeighted(gray_roi, bg_avg_roi, 0.10)
        bg_diff = cv2.absdiff(gray_roi, cv2.convertScaleAbs(bg_avg_roi))
        frame_diff = cv2.absdiff(gray_roi, prev_gray_full[ry : ry + rh, rx : rx + rw])
        combined = cv2.max(bg_diff, frame_diff)

        _, thresh = cv2.threshold(combined, diff_thresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # 전체 화면 기준 움직임 마스크 생성 (정적 배경 진동 계산용)
        full_diff = cv2.absdiff(gray_full, prev_gray_full)
        _, full_motion_mask = cv2.threshold(full_diff, diff_thresh, 255, cv2.THRESH_BINARY)

        # 2. 정적 배경 기준 실제 차체 진동(px) 측정
        jitter_px = 0.0
        if config.detect_shake:
            jitter_px = _compute_static_bg_jitter(prev_gray_full, gray_full, full_motion_mask)

        prev_gray_full = gray_full

        # 3. ROI 내 윤곽선 및 무게중심(궤적) 계산
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(rw * rh)
        min_cnt_area = max(20.0, roi_area * (min_area_pct / 100.0) * 0.25)

        active_area = 0.0
        weighted_cx = 0.0
        weighted_cy = 0.0
        min_x, min_y, max_x, max_y = rw, rh, 0, 0
        valid_count = 0

        for cnt in contours:
            ca = cv2.contourArea(cnt)
            if ca < min_cnt_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 6 or bh < 6:
                continue
            valid_count += 1
            active_area += ca
            cx = bx + bw * 0.5
            cy = by + bh * 0.5
            weighted_cx += cx * ca
            weighted_cy += cy * ca
            min_x = min(min_x, bx)
            min_y = min(min_y, by)
            max_x = max(max_x, bx + bw)
            max_y = max(max_y, by + bh)

        area_pct = (active_area / roi_area) * 100.0 if roi_area > 0 else 0.0
        is_active = (valid_count > 0 and area_pct >= min_area_pct) or (jitter_px >= 0.55)

        if is_active:
            if active_area > 0:
                norm_cx = (rx + (weighted_cx / active_area)) / float(sw)
                norm_cy = (ry + (weighted_cy / active_area)) / float(sh)
                norm_bbox = {
                    "x": round((rx + min_x) / float(sw), 4),
                    "y": round((ry + min_y) / float(sh), 4),
                    "w": round(max(12, max_x - min_x) / float(sw), 4),
                    "h": round(max(12, max_y - min_y) / float(sh), 4),
                }
            else:
                norm_cx = config.roi_x + config.roi_w * 0.5
                norm_cy = config.roi_y + config.roi_h * 0.5
                norm_bbox = {
                    "x": round(config.roi_x, 4),
                    "y": round(config.roi_y, 4),
                    "w": round(config.roi_w, 4),
                    "h": round(config.roi_h, 4),
                }

            inv_scale = 1.0 / scale
            orig_roi_rect = (
                int(rx * inv_scale),
                int(ry * inv_scale),
                int(rw * inv_scale),
                int(rh * inv_scale),
            )

            # 순간 밀착/진동 중요도
            frame_importance = jitter_px * 35.0 + min(35.0, area_pct * 1.4)

            if current_seg is None:
                current_seg = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "peak_time": timestamp,
                    "max_importance": frame_importance,
                    "max_area_pct": area_pct,
                    "max_jitter_px": jitter_px,
                    "first_frame": frame.copy(),
                    "peak_frame": frame.copy(),
                    "last_frame": frame.copy(),
                    "roi_rect": orig_roi_rect,
                    "peak_bbox": norm_bbox,
                    "trajectory": [(timestamp, norm_cx, norm_cy, area_pct)],
                }
            else:
                current_seg["end_time"] = timestamp
                current_seg["last_frame"] = frame.copy()
                current_seg["max_area_pct"] = max(current_seg["max_area_pct"], area_pct)
                current_seg["max_jitter_px"] = max(current_seg["max_jitter_px"], jitter_px)
                current_seg["trajectory"].append((timestamp, norm_cx, norm_cy, area_pct))

                if frame_importance >= current_seg["max_importance"]:
                    current_seg["max_importance"] = frame_importance
                    current_seg["peak_time"] = timestamp
                    current_seg["peak_frame"] = frame.copy()
                    current_seg["peak_bbox"] = norm_bbox
        else:
            if current_seg is not None and (timestamp - current_seg["end_time"]) > config.merge_gap_sec:
                raw_segments.append(current_seg)
                current_seg = None

        if progress_callback and processed_steps % 10 == 0 and total_frames > 0:
            pct = min(99.0, (frame_idx / float(total_frames)) * 100.0)
            progress_callback(pct, len(raw_segments))

    if current_seg is not None:
        raw_segments.append(current_seg)

    cap.release()

    # =========================================================================
    # 세그먼트별 정밀 행동 분석 (단순 통과 차량 배제 & 실제 접촉/체류 점수 산출)
    # =========================================================================
    events: List[DetectedEvent] = []

    for idx, seg in enumerate(raw_segments):
        dwell_sec = round(max(0.2, seg["end_time"] - seg["start_time"]), 2)
        jitter_px = round(seg["max_jitter_px"], 2)
        max_area = round(seg["max_area_pct"], 2)
        traj = seg["trajectory"]

        straightness, has_stop_or_reverse, min_spd_ratio = _analyze_trajectory(traj)
        decel_score = round((1.0 - min_spd_ratio) * 100.0, 1)

        # 판별 로직:
        # 1) 차체 미세 진동(jitter_px >= 0.55px)이 동반된 경우 -> 실제 접촉/충격 강력 의심
        if jitter_px >= 0.55:
            category = "impact"
            label = "🚨 접촉/충격 감지 (차체 미세 흔들림)"
            score = min(98.0, 74.0 + jitter_px * 18.0 + (8.0 if has_stop_or_reverse else 0.0))
            reason = f"배경 기준 차체 미세 흔들림({jitter_px}px) 감지 · 체류 {dwell_sec}초"

        # 2) 단순히 짧은 시간(dwell < min_dwell_sec) 동안 직선(straightness >= 0.65)으로 휙 지나간 경우 -> 단순 통과 차량!
        elif dwell_sec < config.min_dwell_sec and not has_stop_or_reverse and straightness >= 0.62:
            category = "pass_through"
            label = "🚗 단순 통과 차량 (정상 주행)"
            score = round(min(22.0, 6.0 + dwell_sec * 5.0), 1)
            reason = f"일정한 속도로 {dwell_sec}초 만에 직선 통과 (접촉 징후 없음)"
            if config.exclude_pass_through:
                continue

        # 3) 시간은 다소 걸렸더라도 멈칫거림/방향전환 없이 직선으로 통과한 일반 차량
        elif dwell_sec < 3.8 and straightness >= 0.78 and not has_stop_or_reverse:
            category = "pass_through"
            label = "🚗 단순 통과 (일정 속도 이동)"
            score = round(min(28.0, 12.0 + dwell_sec * 3.5), 1)
            reason = f"방향 전환이나 정지 없이 직선 이동 ({dwell_sec}초)"
            if config.exclude_pass_through:
                continue

        # 4) 차량 크기의 객체가 내 차 옆에서 감속/정지/전후진 기동하며 머무른 경우 (차로 긁고 가는 전형적 패턴)
        elif max_area >= 6.0 and (has_stop_or_reverse or dwell_sec >= 3.2):
            category = "scrape_maneuver"
            label = "⚠️ 근접 감속·주차기동 (차량 접촉 의심)"
            dwell_bonus = min(28.0, dwell_sec * 2.8)
            stop_bonus = 18.0 if has_stop_or_reverse else 5.0
            score = min(92.0, 46.0 + dwell_bonus + stop_bonus)
            reason = f"감시 구역 내 {dwell_sec}초 체류 · 근접 감속/방향전환 기동 포착"

        # 5) 보행자 또는 중소형 객체가 차 옆에 다가와 머무르거나 서성거린 경우 (문콕/손·물체 스크래치 패턴)
        else:
            category = "person_linger"
            label = "🕵️ 근접 체류·서성거림 (보행자/문콕 의심)"
            dwell_bonus = min(32.0, dwell_sec * 3.2)
            stop_bonus = 15.0 if has_stop_or_reverse else 0.0
            score = min(89.0, 42.0 + dwell_bonus + stop_bonus)
            reason = f"차량 옆 {dwell_sec}초간 근접 체류 및 반복 움직임 감지"

        thumb_url = _encode_main_thumbnail(
            seg["peak_frame"],
            seg["roi_rect"],
            seg["peak_bbox"],
            traj,
            category,
        )
        strip_url = _encode_strip_thumbnail(
            seg["first_frame"],
            seg["peak_frame"],
            seg["last_frame"],
        )

        start_t = max(0.0, round(seg["start_time"] - 1.5, 2))
        end_t = round(min(duration_sec if duration_sec > 0 else seg["end_time"] + 1.5, seg["end_time"] + 1.5), 2)

        events.append(
            DetectedEvent(
                event_id=f"ev_{file_index}_{idx + 1}",
                file_name=file_name,
                file_index=file_index,
                start_time=start_t,
                end_time=end_t,
                peak_time=round(seg["peak_time"], 2),
                duration=round(max(0.5, end_t - start_t), 2),
                score=round(score, 1),
                category=category,
                label=label,
                reason=reason,
                dwell_sec=dwell_sec,
                jitter_px=jitter_px,
                deceleration_ratio=decel_score,
                bbox=seg["peak_bbox"],
                thumbnail_data_url=thumb_url,
                strip_data_url=strip_url,
            )
        )

    if progress_callback:
        progress_callback(100.0, len(events))

    return events, round(duration_sec, 2)


def event_to_dict(ev: DetectedEvent) -> Dict:
    return asdict(ev)
