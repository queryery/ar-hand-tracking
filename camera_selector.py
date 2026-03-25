import time
import threading

import cv2
import numpy as np

WINDOW = "Camera Selector"
WIDTH, HEIGHT = 520, 400
BG = (18, 18, 24)
ACCENT = (0, 255, 100)
DIM = (100, 100, 100)
WHITE = (210, 210, 210)
ORANGE = (0, 180, 255)


def _draw_text(img, text, pos, scale=0.5, color=WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _make_panel():
    return np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)


def _probe_cameras(max_index=10):
    found = []
    seen = set()
    for backend_name, backend in [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]:
        for i in range(max_index):
            if i in seen:
                continue
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    h, w = frame.shape[:2]
                    found.append({"index": i, "label": f"Camera {i}  -  {w}x{h}", "preview": frame})
                else:
                    found.append({"index": i, "label": f"Camera {i}  -  no preview", "preview": None})
                seen.add(i)
            else:
                cap.release()
    return found


def select_camera(max_index=10):
    cameras = []
    scanning = True
    scan_done = False
    selected = 0
    typed_index = ""
    status_msg = ""
    status_time = 0

    def _scan():
        nonlocal cameras, scanning, scan_done
        cameras = _probe_cameras(max_index)
        scanning = False
        scan_done = True

    scan_thread = threading.Thread(target=_scan, daemon=True)
    scan_thread.start()

    while True:
        panel = _make_panel()

        _draw_text(panel, "Camera Selector", (15, 32), 0.75, ACCENT, 2)
        cv2.line(panel, (15, 45), (WIDTH - 15, 45), (40, 40, 50), 1)

        if scanning:
            dots = "." * (int(time.time() * 3) % 4)
            _draw_text(panel, f"Scanning for cameras{dots}", (15, 80), 0.55, ORANGE)
            _draw_text(panel, "Make sure your camera / virtual cam is running", (15, 110), 0.4, DIM)

        elif cameras:
            _draw_text(panel, f"{len(cameras)} camera(s) found", (15, 75), 0.45, DIM)

            list_y = 100
            for i, cam in enumerate(cameras):
                y = list_y + i * 50
                if y + 50 > HEIGHT - 80:
                    break

                is_sel = i == selected
                box_color = ACCENT if is_sel else (40, 40, 50)
                cv2.rectangle(panel, (15, y), (WIDTH - 15, y + 40), box_color, 2 if is_sel else 1)

                label_color = ACCENT if is_sel else WHITE
                _draw_text(panel, cam["label"], (25, y + 26), 0.5, label_color)

                if is_sel and cam["preview"] is not None:
                    thumb = cv2.resize(cam["preview"], (80, 50))
                    ty = max(y - 5, 0)
                    tx = WIDTH - 110
                    th, tw = thumb.shape[:2]
                    if ty + th < HEIGHT and tx + tw < WIDTH:
                        panel[ty:ty + th, tx:tx + tw] = thumb

        else:
            _draw_text(panel, "No cameras detected", (15, 80), 0.55, (0, 100, 255))
            _draw_text(panel, "Connect your camera and press R to rescan", (15, 115), 0.45, WHITE)
            _draw_text(panel, "Or type a camera index (0-9) and press Enter", (15, 145), 0.45, WHITE)

        if typed_index:
            _draw_text(panel, f"Manual index: {typed_index}_", (15, HEIGHT - 55), 0.5, ORANGE)

        if status_msg and time.time() - status_time < 3:
            _draw_text(panel, status_msg, (15, HEIGHT - 30), 0.4, (0, 100, 255))

        controls = "Up/Down: navigate | Enter: select | R: rescan | 0-9: manual | Esc: quit"
        _draw_text(panel, controls, (15, HEIGHT - 10), 0.33, DIM)

        cv2.imshow(WINDOW, panel)
        key = cv2.waitKey(50) & 0xFF

        if key == 27:
            cv2.destroyWindow(WINDOW)
            return None

        elif key == ord("r") or key == ord("R"):
            if not scanning:
                cameras = []
                scanning = True
                scan_done = False
                typed_index = ""
                scan_thread = threading.Thread(target=_scan, daemon=True)
                scan_thread.start()

        elif key in (ord("\r"), ord("\n"), 13):
            if typed_index:
                idx = int(typed_index)
                typed_index = ""
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        cv2.destroyWindow(WINDOW)
                        return idx
                    else:
                        status_msg = f"Camera {idx} opened but gave no frames"
                        status_time = time.time()
                else:
                    cap.release()
                    status_msg = f"Camera {idx} failed to open"
                    status_time = time.time()
            elif cameras:
                cv2.destroyWindow(WINDOW)
                return cameras[selected]["index"]

        elif ord("0") <= key <= ord("9"):
            typed_index = (typed_index + chr(key))[-2:]

        elif key == 8:
            typed_index = typed_index[:-1]

        elif not scanning and cameras:
            if key == 0:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == 72:
                    selected = (selected - 1) % len(cameras)
                elif key2 == 80:
                    selected = (selected + 1) % len(cameras)
            elif key in (82, 2):
                selected = (selected - 1) % len(cameras)
            elif key in (84, 3):
                selected = (selected + 1) % len(cameras)

    return None
