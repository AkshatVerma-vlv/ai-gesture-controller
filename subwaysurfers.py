import cv2
import time
import math
import torch
import pydirectinput
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n-pose.pt')
model.to(device)

pydirectinput.PAUSE = 0.0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if device == 'cuda':
    dummy_frame = torch.zeros((1, 3, 320, 320), device=device)
    _ = model(dummy_frame, verbose=False)

# Kinematic thresholds
INVERT_HORIZONTAL = True

FUSED_SWIPE_TRIGGER = 0.118
FUSED_SWIPE_RELEASE = 0.038
H_COOLDOWN = 0.14

FUSED_JUMP_TRIGGER = 0.050
FUSED_DUCK_TRIGGER = -0.060
FUSED_VERT_RELEASE = 0.016
V_COOLDOWN = 0.15

REPEAT_WINDOW = 0.50
RECOIL_REMAINING_RATIO = 0.375
REPEAT_RETRIGGER_RATIO = 0.85
REPEAT_MIN_VELOCITY = 0.055

SHRUG_TRIGGER = 0.038
MAX_HEAD_PITCH_VARIATION = 0.045

DRIFT_VELOCITY_CEILING = 0.08
ADAPTATION_RATE_H = 0.035
ADAPTATION_RATE_V = 0.065

# State variables
base_fused_h = 0.0
base_cranial_pitch = None
base_torso_span = None
base_sh_height = None

prev_nx = None
prev_ny = None

h_state = "IDLE"
h_last_direction = None
h_peak_amplitude = 0.0
h_repeat_deadline = 0.0
last_h_time = 0.0

v_state = "IDLE"
v_last_direction = None
v_peak_amplitude = 0.0
v_repeat_deadline = 0.0
last_v_time = 0.0

hoverboard_latched = False

last_action_display = "RUNNING (NEUTRAL)"
action_display_until = 0.0

infer_ms = 0.0
prev_time = time.time()

print("Subway Surfers Controller initialized. Press 'c' to recalibrate, 'q' to quit.")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        t_start = time.perf_counter()
        results = model(frame, device=device, imgsz=320, verbose=False)
        infer_ms = (time.perf_counter() - t_start) * 1000

        annotated_frame = results[0].plot()

        if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
            keypoints = results[0].keypoints.xyn[0]

            nose = keypoints[0]
            l_ear, r_ear = keypoints[3], keypoints[4]
            l_sh, r_sh = keypoints[5], keypoints[6]
            lwrist, rwrist = keypoints[9], keypoints[10]

            nx, ny = float(nose[0]), float(nose[1])
            lx, ly = float(l_ear[0]), float(l_ear[1])
            rx, ry = float(r_ear[0]), float(r_ear[1])
            ls_x, ls_y = float(l_sh[0]), float(l_sh[1])
            rs_x, rs_y = float(r_sh[0]), float(r_sh[1])
            lw_y, rw_y = float(lwrist[1]), float(rwrist[1])

            if nx > 0 and lx > 0 and rx > 0 and ls_y > 0 and rs_y > 0:
                shoulder_span = math.hypot(rs_x - ls_x, rs_y - ls_y)
                ear_span = math.hypot(rx - lx, ry - ly)

                if shoulder_span > 0.08 and ear_span > 0.03:
                    now = time.time()
                    dt_frame = max(now - prev_time, 0.001)

                    if prev_nx is None:
                        prev_nx, prev_ny = nx, ny

                    nose_raw_vx = (nx - prev_nx) / dt_frame
                    nose_raw_vy = (ny - prev_ny) / dt_frame
                    instant_speed = math.hypot(nose_raw_vx, nose_raw_vy) / shoulder_span

                    dist_to_left_ear = math.hypot(nx - lx, ny - ly)
                    dist_to_right_ear = math.hypot(rx - nx, ry - ny)
                    ear_diff_ratio = (dist_to_left_ear - dist_to_right_ear) / ear_span

                    norm_nose_vx = (nose_raw_vx / shoulder_span) * 0.03
                    raw_fused_h = ear_diff_ratio + norm_nose_vx
                    if INVERT_HORIZONTAL:
                        raw_fused_h = -raw_fused_h

                    ear_mid_y = (ly + ry) / 2.0
                    sh_mid_y = (ls_y + rs_y) / 2.0

                    raw_cranial_pitch = (ear_mid_y - ny) / ear_span
                    raw_torso_span = (sh_mid_y - ny) / shoulder_span
                    norm_nose_vy = (-nose_raw_vy / shoulder_span) * 0.015
                    sh_elevation = sh_mid_y

                    prev_nx, prev_ny = nx, ny

                    if base_cranial_pitch is None:
                        base_fused_h = raw_fused_h
                        base_cranial_pitch = raw_cranial_pitch
                        base_torso_span = raw_torso_span
                        base_sh_height = sh_elevation

                    d_cranial_pitch = raw_cranial_pitch - base_cranial_pitch
                    d_torso_span = raw_torso_span - base_torso_span
                    shrug_delta = (base_sh_height - sh_elevation) / shoulder_span

                    d_fused_h = raw_fused_h - base_fused_h
                    raw_fused_v = (d_cranial_pitch * 0.65) + (d_torso_span * 0.35) + norm_nose_vy

                    # Baseline drift adaptation
                    is_slow_movement = instant_speed < DRIFT_VELOCITY_CEILING
                    near_h_center = abs(d_fused_h) < (FUSED_SWIPE_TRIGGER * 0.70)
                    near_v_center = (FUSED_DUCK_TRIGGER * 0.75) < raw_fused_v < (FUSED_JUMP_TRIGGER * 0.75)
                    not_shrugging_up = shrug_delta < (SHRUG_TRIGGER * 0.65)

                    is_calibrating = False
                    if is_slow_movement and near_h_center and near_v_center and not_shrugging_up and not hoverboard_latched:
                        base_fused_h = (1.0 - ADAPTATION_RATE_H) * base_fused_h + ADAPTATION_RATE_H * raw_fused_h
                        base_cranial_pitch = (1.0 - ADAPTATION_RATE_V) * base_cranial_pitch + ADAPTATION_RATE_V * raw_cranial_pitch
                        base_torso_span = (1.0 - ADAPTATION_RATE_V) * base_torso_span + ADAPTATION_RATE_V * raw_torso_span
                        base_sh_height = (1.0 - ADAPTATION_RATE_V) * base_sh_height + ADAPTATION_RATE_V * sh_elevation
                        is_calibrating = True

                    d_cranial_pitch = raw_cranial_pitch - base_cranial_pitch
                    d_torso_span = raw_torso_span - base_torso_span
                    shrug_delta = (base_sh_height - sh_elevation) / shoulder_span
                    d_fused_h = raw_fused_h - base_fused_h
                    raw_fused_v = (d_cranial_pitch * 0.65) + (d_torso_span * 0.35) + norm_nose_vy

                    # Hoverboard logic
                    shrug_detected = (shrug_delta > SHRUG_TRIGGER) and (abs(d_cranial_pitch) < MAX_HEAD_PITCH_VARIATION)
                    hand_raised = (0 < lw_y < ny) or (0 < rw_y < ny)
                    is_active_shrug = shrug_delta > 0.020

                    if shrug_detected or hand_raised:
                        if not hoverboard_latched:
                            pydirectinput.press('space')
                            hoverboard_latched = True
                            last_action_display = "HOVERBOARD"
                            action_display_until = now + 0.50
                    else:
                        hoverboard_latched = False

                    # Horizontal input logic
                    if abs(d_fused_h) < FUSED_SWIPE_RELEASE:
                        if now > h_repeat_deadline:
                            h_state = "IDLE"
                            h_last_direction = None

                    if h_state == "ARMED" and now > h_repeat_deadline:
                        h_state = "IDLE"

                    if h_state == "LOCKED_OUT":
                        if h_last_direction == "RIGHT" and d_fused_h > h_peak_amplitude:
                            h_peak_amplitude = d_fused_h
                        elif h_last_direction == "LEFT" and d_fused_h < h_peak_amplitude:
                            h_peak_amplitude = d_fused_h

                        h_recoil_target = abs(h_peak_amplitude) * RECOIL_REMAINING_RATIO
                        if abs(d_fused_h) <= max(h_recoil_target, FUSED_SWIPE_RELEASE):
                            h_state = "ARMED"
                            h_repeat_deadline = now + REPEAT_WINDOW

                    if (now - last_h_time) > H_COOLDOWN:
                        if h_state == "IDLE":
                            if d_fused_h > FUSED_SWIPE_TRIGGER:
                                pydirectinput.press('right')
                                last_h_time = now
                                last_action_display = "SWIPE RIGHT"
                                action_display_until = now + 0.35
                                h_state = "LOCKED_OUT"
                                h_last_direction = "RIGHT"
                                h_peak_amplitude = d_fused_h

                            elif d_fused_h < -FUSED_SWIPE_TRIGGER:
                                pydirectinput.press('left')
                                last_h_time = now
                                last_action_display = "SWIPE LEFT"
                                action_display_until = now + 0.35
                                h_state = "LOCKED_OUT"
                                h_last_direction = "LEFT"
                                h_peak_amplitude = d_fused_h

                        elif h_state == "ARMED":
                            h_repeat_req = max(FUSED_SWIPE_TRIGGER * 0.85, abs(h_peak_amplitude) * REPEAT_RETRIGGER_RATIO)
                            is_fast_flick = instant_speed > REPEAT_MIN_VELOCITY

                            if h_last_direction == "RIGHT" and d_fused_h > h_repeat_req and is_fast_flick:
                                pydirectinput.press('right')
                                last_h_time = now
                                last_action_display = "SWIPE RIGHT (REPEAT x2)"
                                action_display_until = now + 0.40
                                h_state = "LOCKED_OUT"
                                h_repeat_deadline = 0.0

                            elif h_last_direction == "LEFT" and d_fused_h < -h_repeat_req and is_fast_flick:
                                pydirectinput.press('left')
                                last_h_time = now
                                last_action_display = "SWIPE LEFT (REPEAT x2)"
                                action_display_until = now + 0.40
                                h_state = "LOCKED_OUT"
                                h_repeat_deadline = 0.0

                            elif h_last_direction == "RIGHT" and d_fused_h < -FUSED_SWIPE_TRIGGER:
                                pydirectinput.press('left')
                                last_h_time = now
                                last_action_display = "SWIPE LEFT"
                                action_display_until = now + 0.35
                                h_state = "LOCKED_OUT"
                                h_last_direction = "LEFT"
                                h_peak_amplitude = d_fused_h

                            elif h_last_direction == "LEFT" and d_fused_h > FUSED_SWIPE_TRIGGER:
                                pydirectinput.press('right')
                                last_h_time = now
                                last_action_display = "SWIPE RIGHT"
                                action_display_until = now + 0.35
                                h_state = "LOCKED_OUT"
                                h_last_direction = "RIGHT"
                                h_peak_amplitude = d_fused_h

                    # Vertical input logic
                    if abs(raw_fused_v) < FUSED_VERT_RELEASE and now > v_repeat_deadline:
                        v_state = "IDLE"
                        v_last_direction = None

                    if v_state == "ARMED" and now > v_repeat_deadline:
                        v_state = "IDLE"

                    if v_state == "LOCKED_OUT":
                        if v_last_direction == "JUMP" and raw_fused_v > v_peak_amplitude:
                            v_peak_amplitude = raw_fused_v
                        elif v_last_direction == "DUCK" and raw_fused_v < v_peak_amplitude:
                            v_peak_amplitude = raw_fused_v

                        v_recoil_target = abs(v_peak_amplitude) * RECOIL_REMAINING_RATIO
                        if abs(raw_fused_v) <= max(v_recoil_target, FUSED_VERT_RELEASE):
                            v_state = "ARMED"
                            v_repeat_deadline = now + REPEAT_WINDOW

                    if (now - last_v_time) > V_COOLDOWN:
                        if v_state == "IDLE":
                            if raw_fused_v > FUSED_JUMP_TRIGGER:
                                pydirectinput.press('up')
                                last_v_time = now
                                last_action_display = "JUMP"
                                action_display_until = now + 0.35
                                v_state = "LOCKED_OUT"
                                v_last_direction = "JUMP"
                                v_peak_amplitude = raw_fused_v

                            elif raw_fused_v < FUSED_DUCK_TRIGGER and not is_active_shrug and not hoverboard_latched:
                                pydirectinput.press('down')
                                last_v_time = now
                                last_action_display = "DUCK"
                                action_display_until = now + 0.35
                                v_state = "LOCKED_OUT"
                                v_last_direction = "DUCK"
                                v_peak_amplitude = raw_fused_v

                        elif v_state == "ARMED":
                            v_repeat_req = max(FUSED_JUMP_TRIGGER * 0.85, abs(v_peak_amplitude) * REPEAT_RETRIGGER_RATIO)
                            is_fast_flick = instant_speed > REPEAT_MIN_VELOCITY

                            if v_last_direction == "JUMP" and raw_fused_v > v_repeat_req and is_fast_flick:
                                pydirectinput.press('up')
                                last_v_time = now
                                last_action_display = "JUMP (REPEAT x2)"
                                action_display_until = now + 0.40
                                v_state = "LOCKED_OUT"
                                v_repeat_deadline = 0.0

                            elif raw_fused_v < FUSED_DUCK_TRIGGER and not is_active_shrug and not hoverboard_latched:
                                pydirectinput.press('down')
                                last_v_time = now
                                last_action_display = "DUCK"
                                action_display_until = now + 0.35
                                v_state = "LOCKED_OUT"
                                v_last_direction = "DUCK"
                                v_peak_amplitude = raw_fused_v

                    if now > action_display_until:
                        last_action_display = "RUNNING (NEUTRAL)"

                    # HUD Gauges rendering
                    bar_cx = int(w / 2)
                    bar_y = int(h - 35)
                    ind_x = int(bar_cx + (d_fused_h * 250))
                    ind_x = max(bar_cx - 100, min(bar_cx + 100, ind_x))

                    h_color = (0, 255, 255) if h_state == "ARMED" else ((0, 255, 0) if h_state == "LOCKED_OUT" else (180, 180, 180))
                    cv2.line(annotated_frame, (bar_cx - 90, bar_y), (bar_cx + 90, bar_y), (60, 60, 60), 4)
                    cv2.line(annotated_frame, (int(bar_cx - FUSED_SWIPE_TRIGGER * 250), bar_y - 8),
                             (int(bar_cx - FUSED_SWIPE_TRIGGER * 250), bar_y + 8), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (int(bar_cx + FUSED_SWIPE_TRIGGER * 250), bar_y - 8),
                             (int(bar_cx + FUSED_SWIPE_TRIGGER * 250), bar_y + 8), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (int(bar_cx - FUSED_SWIPE_RELEASE * 250), bar_y - 5),
                             (int(bar_cx - FUSED_SWIPE_RELEASE * 250), bar_y + 5), (0, 255, 0), 1)
                    cv2.line(annotated_frame, (int(bar_cx + FUSED_SWIPE_RELEASE * 250), bar_y - 5),
                             (int(bar_cx + FUSED_SWIPE_RELEASE * 250), bar_y + 5), (0, 255, 0), 1)
                    cv2.circle(annotated_frame, (ind_x, bar_y), 7, h_color, -1)

                    vbar_x = int(w - 35)
                    vbar_cy = int(h / 2)
                    ind_y = int(vbar_cy - (raw_fused_v * 750))
                    ind_y = max(vbar_cy - 80, min(vbar_cy + 80, ind_y))

                    v_color = (0, 255, 255) if v_state == "ARMED" else ((0, 255, 0) if v_state == "LOCKED_OUT" else (180, 180, 180))
                    cv2.line(annotated_frame, (vbar_x, vbar_cy - 80), (vbar_x, vbar_cy + 80), (60, 60, 60), 4)
                    cv2.line(annotated_frame, (vbar_x - 8, int(vbar_cy - FUSED_JUMP_TRIGGER * 750)),
                             (vbar_x + 8, int(vbar_cy - FUSED_JUMP_TRIGGER * 750)), (0, 255, 0), 2)
                    cv2.line(annotated_frame, (vbar_x - 8, int(vbar_cy - FUSED_DUCK_TRIGGER * 750)),
                             (vbar_x + 8, int(vbar_cy - FUSED_DUCK_TRIGGER * 750)), (0, 0, 255), 2)
                    cv2.circle(annotated_frame, (vbar_x, ind_y), 7, v_color, -1)

                    sbar_x = 35
                    sbar_cy = int(h / 2)
                    s_ind_y = int(sbar_cy - (shrug_delta * 1200))
                    s_ind_y = max(sbar_cy - 80, min(sbar_cy + 80, s_ind_y))

                    shrug_color = (0, 255, 255) if hoverboard_latched else (120, 120, 120)
                    cv2.line(annotated_frame, (sbar_x, sbar_cy - 80), (sbar_x, sbar_cy + 80), (60, 60, 60), 3)
                    cv2.line(annotated_frame, (sbar_x - 8, int(sbar_cy - SHRUG_TRIGGER * 1200)),
                             (sbar_x + 8, int(sbar_cy - SHRUG_TRIGGER * 1200)), (0, 255, 255), 2)
                    cv2.circle(annotated_frame, (sbar_x, s_ind_y), 6, shrug_color, -1)
                    cv2.putText(annotated_frame, "SHRUG", (15, sbar_cy - 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        curr_time = time.time()
        dt = curr_time - prev_time
        fps = 1 / dt if dt > 0 else 30
        prev_time = curr_time

        text_color = (0, 255, 0) if "REPEAT" not in last_action_display else (0, 255, 255)
        if last_action_display == "RUNNING (NEUTRAL)":
            text_color = (200, 200, 200)

        cv2.putText(annotated_frame, f"STATE: {last_action_display}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

        state_hud = f"H:{h_state} | V:{v_state}"
        calib_status = "DRIFT: ACTIVE" if ('is_calibrating' in locals() and is_calibrating) else "DRIFT: LOCKED"
        calib_color = (0, 255, 0) if ('is_calibrating' in locals() and is_calibrating) else (0, 165, 255)

        cv2.putText(annotated_frame, f"CNN: {infer_ms:.1f}ms | Loop: {int(fps)} FPS | {state_hud} | {calib_status}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, calib_color, 1)

        cv2.imshow('Subway Surfers Controller', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if 'raw_fused_h' in locals() and 'raw_cranial_pitch' in locals() and 'sh_elevation' in locals():
                base_fused_h = raw_fused_h
                base_cranial_pitch = raw_cranial_pitch
                base_torso_span = raw_torso_span
                base_sh_height = sh_elevation
                h_state = "IDLE"
                h_last_direction = None
                v_state = "IDLE"
                v_last_direction = None
                hoverboard_latched = False
                last_action_display = "RUNNING (NEUTRAL)"
                print("Baseline re-centered manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
