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
STEER_TRIGGER = 0.068
STEER_RELEASE_FLOOR = 0.025
STEER_INFLECTION_DELTA = 0.015
STEER_REARM_MARGIN = 0.018

BASE_BRAKE_TRIGGER = 0.024
BRAKE_RELEASE_FLOOR = 0.010
NOSE_DROP_DELTA = 0.007
BRAKE_REARM_MARGIN = 0.012

SHRUG_TRIGGER = 0.045

# State variables
base_roll = 0.0
base_pitch = None
base_sh_height = None

peak_roll = 0.0
steer_latched_side = None
rearm_steer_target = 0.0

peak_brake_pitch = 0.0
brake_released_high = False
rearm_target_pitch = 0.0

key_left_down = False
key_right_down = False
key_up_down = False
key_down_down = False
key_space_down = False

last_h_action = "CENTER"
last_v_action = "THROTTLE"
wheelie_active = False

infer_ms = 0.0
prev_time = time.time()

def release_all_keys():
    pydirectinput.keyUp('left')
    pydirectinput.keyUp('right')
    pydirectinput.keyUp('up')
    pydirectinput.keyUp('down')
    pydirectinput.keyUp('space')

print("Bike Racer Controller initialized. Press 'c' to recalibrate, 'q' to quit.")

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

            nx, ny = float(nose[0]), float(nose[1])
            lx, ly = float(l_ear[0]), float(l_ear[1])
            rx, ry = float(r_ear[0]), float(r_ear[1])
            ls_x, ls_y = float(l_sh[0]), float(l_sh[1])
            rs_x, rs_y = float(r_sh[0]), float(r_sh[1])

            if nx > 0 and lx > 0 and rx > 0 and ls_y > 0 and rs_y > 0:
                shoulder_span = math.hypot(rs_x - ls_x, rs_y - ls_y)
                ear_dx = rx - lx
                ear_dy = ry - ly
                ear_span = math.hypot(ear_dx, ear_dy)

                if shoulder_span > 0.08 and ear_span > 0.03:
                    # Differential steering
                    dist_l = math.hypot(lx - ls_x, ly - ls_y)
                    dist_r = math.hypot(rx - rs_x, ry - rs_y)
                    raw_roll = (dist_r - dist_l) / shoulder_span

                    # Anatomical pitch
                    perp_x = -ear_dy / ear_span
                    perp_y = ear_dx / ear_span

                    ear_mid_x = (lx + rx) / 2.0
                    ear_mid_y = (ly + ry) / 2.0
                    v_nose_x = nx - ear_mid_x
                    v_nose_y = ny - ear_mid_y

                    cranial_pitch = -(v_nose_x * perp_x + v_nose_y * perp_y) / ear_span

                    sh_mid_y = (ls_y + rs_y) / 2.0
                    vert_torso_span = (sh_mid_y - ny) / shoulder_span

                    raw_pitch = vert_torso_span + (cranial_pitch * 0.5)
                    sh_elevation = sh_mid_y

                    if base_pitch is None:
                        base_roll = raw_roll
                        base_pitch = raw_pitch
                        base_sh_height = sh_elevation

                    # Baseline adaptation
                    if not key_left_down and not key_right_down and abs(raw_roll - base_roll) < STEER_RELEASE_FLOOR:
                        base_roll = 0.94 * base_roll + 0.06 * raw_roll
                        steer_latched_side = None

                    if not key_down_down and not wheelie_active and not brake_released_high:
                        if abs(raw_pitch - base_pitch) < 0.020:
                            base_pitch = 0.92 * base_pitch + 0.08 * raw_pitch
                        base_sh_height = 0.94 * base_sh_height + 0.06 * sh_elevation

                    d_roll = raw_roll - base_roll
                    d_pitch = raw_pitch - base_pitch
                    shrug_delta = base_sh_height - sh_elevation

                    active_steering = (key_left_down or key_right_down)
                    active_brake_trigger = BASE_BRAKE_TRIGGER * (1.50 if active_steering else 1.0)

                    # Steering logic
                    if abs(d_roll) < STEER_RELEASE_FLOOR:
                        steer_latched_side = None

                    if not key_left_down and not key_right_down:
                        can_turn_right = (steer_latched_side != "RIGHT" and d_roll > STEER_TRIGGER) or \
                                         (steer_latched_side == "RIGHT" and d_roll > rearm_steer_target)
                        can_turn_left = (steer_latched_side != "LEFT" and d_roll < -STEER_TRIGGER) or \
                                        (steer_latched_side == "LEFT" and d_roll < -rearm_steer_target)

                        if can_turn_right:
                            pydirectinput.keyDown('right')
                            key_right_down = True
                            peak_roll = d_roll
                            steer_latched_side = None
                            last_h_action = "STEER RIGHT"
                        elif can_turn_left:
                            pydirectinput.keyDown('left')
                            key_left_down = True
                            peak_roll = d_roll
                            steer_latched_side = None
                            last_h_action = "STEER LEFT"
                        else:
                            last_h_action = "CENTER"

                    elif key_right_down:
                        if d_roll > peak_roll:
                            peak_roll = d_roll

                        returned_center = (d_roll < STEER_RELEASE_FLOOR)
                        reversed_lean = (peak_roll - d_roll) > STEER_INFLECTION_DELTA
                        flick_left = (d_roll < -STEER_TRIGGER)

                        if returned_center or reversed_lean or flick_left:
                            pydirectinput.keyUp('right')
                            key_right_down = False

                            if flick_left:
                                pydirectinput.keyDown('left')
                                key_left_down = True
                                peak_roll = d_roll
                                steer_latched_side = None
                                last_h_action = "STEER LEFT"
                            else:
                                last_h_action = "CENTER"
                                if reversed_lean and not returned_center:
                                    steer_latched_side = "RIGHT"
                                    rearm_steer_target = d_roll + STEER_REARM_MARGIN
                                else:
                                    steer_latched_side = None
                        else:
                            last_h_action = "STEER RIGHT"

                    elif key_left_down:
                        if d_roll < peak_roll:
                            peak_roll = d_roll

                        returned_center = (d_roll > -STEER_RELEASE_FLOOR)
                        reversed_lean = (d_roll - peak_roll) > STEER_INFLECTION_DELTA
                        flick_right = (d_roll > STEER_TRIGGER)

                        if returned_center or reversed_lean or flick_right:
                            pydirectinput.keyUp('left')
                            key_left_down = False

                            if flick_right:
                                pydirectinput.keyDown('right')
                                key_right_down = True
                                peak_roll = d_roll
                                steer_latched_side = None
                                last_h_action = "STEER RIGHT"
                            else:
                                last_h_action = "CENTER"
                                if reversed_lean and not returned_center:
                                    steer_latched_side = "LEFT"
                                    rearm_steer_target = abs(d_roll) + STEER_REARM_MARGIN
                                else:
                                    steer_latched_side = None
                        else:
                            last_h_action = "STEER LEFT"

                    # Throttle and Brake logic
                    if not key_down_down:
                        if d_pitch < BRAKE_RELEASE_FLOOR:
                            brake_released_high = False

                        should_brake = False
                        if not brake_released_high:
                            if d_pitch > active_brake_trigger:
                                should_brake = True
                        else:
                            if d_pitch > rearm_target_pitch:
                                should_brake = True
                                brake_released_high = False

                        if should_brake:
                            if key_up_down:
                                pydirectinput.keyUp('up')
                                key_up_down = False
                            pydirectinput.keyDown('down')
                            key_down_down = True
                            peak_brake_pitch = d_pitch
                            last_v_action = "BRAKE"
                        else:
                            if not key_up_down:
                                pydirectinput.keyDown('up')
                                key_up_down = True
                            last_v_action = "THROTTLE"

                    else:
                        if d_pitch > peak_brake_pitch:
                            peak_brake_pitch = d_pitch

                        returned_to_floor = (d_pitch < BRAKE_RELEASE_FLOOR)
                        nose_dipped_forward = (peak_brake_pitch - d_pitch) > NOSE_DROP_DELTA

                        if returned_to_floor or nose_dipped_forward:
                            pydirectinput.keyUp('down')
                            key_down_down = False
                            pydirectinput.keyDown('up')
                            key_up_down = True
                            last_v_action = "THROTTLE"

                            if nose_dipped_forward and not returned_to_floor:
                                brake_released_high = True
                                rearm_target_pitch = d_pitch + BRAKE_REARM_MARGIN
                            else:
                                brake_released_high = False
                        else:
                            last_v_action = "BRAKE"

                    # Wheelie logic
                    if shrug_delta > SHRUG_TRIGGER and d_pitch < active_brake_trigger:
                        if not key_space_down:
                            pydirectinput.keyDown('space')
                            key_space_down = True
                        wheelie_active = True
                    else:
                        if key_space_down:
                            pydirectinput.keyUp('space')
                            key_space_down = False
                        wheelie_active = False

                    # HUD Gauges rendering
                    bar_cx = int(w / 2)
                    bar_y = int(h - 35)
                    ind_x = int(bar_cx + (d_roll * 350))
                    ind_x = max(bar_cx - 100, min(bar_cx + 100, ind_x))

                    steer_color = (0, 255, 0) if (key_left_down or key_right_down) else (180, 180, 180)
                    cv2.line(annotated_frame, (bar_cx - 90, bar_y), (bar_cx + 90, bar_y), (60, 60, 60), 4)
                    
                    cv2.line(annotated_frame, (int(bar_cx - STEER_TRIGGER * 350), bar_y - 8),
                             (int(bar_cx - STEER_TRIGGER * 350), bar_y + 8), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (int(bar_cx + STEER_TRIGGER * 350), bar_y - 8),
                             (int(bar_cx + STEER_TRIGGER * 350), bar_y + 8), (0, 0, 255), 2)
                    
                    cv2.line(annotated_frame, (int(bar_cx - STEER_RELEASE_FLOOR * 350), bar_y - 5),
                             (int(bar_cx - STEER_RELEASE_FLOOR * 350), bar_y + 5), (0, 255, 0), 1)
                    cv2.line(annotated_frame, (int(bar_cx + STEER_RELEASE_FLOOR * 350), bar_y - 5),
                             (int(bar_cx + STEER_RELEASE_FLOOR * 350), bar_y + 5), (0, 255, 0), 1)
                    cv2.circle(annotated_frame, (ind_x, bar_y), 7, steer_color, -1)

                    if key_left_down or key_right_down:
                        peak_x = int(bar_cx + (peak_roll * 350))
                        cv2.line(annotated_frame, (peak_x, bar_y - 10), (peak_x, bar_y + 10), (255, 255, 0), 2)

                    vbar_x = int(w - 35)
                    vbar_cy = int(h / 2)
                    ind_y = int(vbar_cy + (d_pitch * 1800))
                    ind_y = max(vbar_cy - 80, min(vbar_cy + 80, ind_y))

                    pedal_color = (0, 0, 255) if key_down_down else (0, 255, 0)
                    cv2.line(annotated_frame, (vbar_x, vbar_cy - 80), (vbar_x, vbar_cy + 80), (60, 60, 60), 4)
                    cv2.line(annotated_frame, (vbar_x - 8, int(vbar_cy + active_brake_trigger * 1800)),
                             (vbar_x + 8, int(vbar_cy + active_brake_trigger * 1800)), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (vbar_x - 6, int(vbar_cy + BRAKE_RELEASE_FLOOR * 1800)),
                             (vbar_x + 6, int(vbar_cy + BRAKE_RELEASE_FLOOR * 1800)), (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (vbar_x, ind_y), 7, pedal_color, -1)

                    if key_down_down:
                        peak_y = int(vbar_cy + (peak_brake_pitch * 1800))
                        cv2.line(annotated_frame, (vbar_x - 10, peak_y), (vbar_x + 10, peak_y), (255, 255, 0), 2)

        curr_time = time.time()
        dt = curr_time - prev_time
        fps = 1 / dt if dt > 0 else 30
        prev_time = curr_time

        steer_flag = " [INFLECTION]" if steer_latched_side else ""
        brake_flag = " [INFLECTION]" if brake_released_high else ""
        wheelie_txt = " | WHEELIE!" if wheelie_active else ""

        cv2.putText(annotated_frame, f"{last_h_action}{steer_flag} | {last_v_action}{brake_flag}{wheelie_txt}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"CNN: {infer_ms:.1f}ms | Loop: {int(fps)} FPS", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

        cv2.imshow('Highway Bike Racer', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if 'raw_roll' in locals() and 'raw_pitch' in locals() and 'sh_elevation' in locals():
                base_roll = raw_roll
                base_pitch = raw_pitch
                base_sh_height = sh_elevation
                steer_latched_side = None
                brake_released_high = False

finally:
    release_all_keys()
    cap.release() 
    cv2.destroyAllWindows()