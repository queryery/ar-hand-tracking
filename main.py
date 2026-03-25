import os
import subprocess
import sys
import time

import config


def _ensure_deps():
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if not os.path.exists(req):
        return
    print("Checking dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", req],
        stdout=subprocess.DEVNULL,
    )


_ensure_deps()

import cv2  # noqa: E402
from camera import Camera
from camera_selector import select_camera
from detector import HandDetector
from engine3d import Cube, Ball
from gestures import GestureEngine
from renderer import Renderer

CUBE_COLORS = [(0, 180, 255), (255, 100, 50), (180, 50, 255), (50, 255, 180)]
BALL_COLORS = [(50, 200, 100), (200, 50, 150), (100, 200, 255), (255, 200, 50)]


def _draw_pip(frame, raw_frame, fh, fw):
    """Draw picture-in-picture raw camera feed."""
    pip_h = int(fh * config.PIP_SCALE)
    pip_w = int(fw * config.PIP_SCALE)
    pip = cv2.resize(raw_frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

    m = config.PIP_MARGIN
    y1, x1 = fh - pip_h - m, m

    # Border
    cv2.rectangle(
        frame, (x1 - 2, y1 - 2), (x1 + pip_w + 1, y1 + pip_h + 1),
        config.PIP_BORDER, 1, cv2.LINE_AA,
    )
    frame[y1:y1 + pip_h, x1:x1 + pip_w] = pip

    # LIVE indicator
    cv2.circle(frame, (x1 + 8, y1 + 11), 3, (0, 0, 220), -1, cv2.LINE_AA)
    cv2.putText(
        frame, "LIVE", (x1 + 15, y1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1, cv2.LINE_AA,
    )


def main():
    cam_index = select_camera()
    if cam_index is None:
        print("No camera selected.")
        return

    camera = Camera(cam_index)
    detector = HandDetector().start()
    renderer = Renderer()

    objects = [
        Cube(0.35, 0.5, color=(0, 180, 255)),
        Ball(0.65, 0.5, color=(50, 200, 100)),
    ]
    objects[0].rot[0] = 0.4
    objects[0].rot[1] = 0.6
    gestures = GestureEngine(objects)

    cube_idx, ball_idx = 1, 1
    pip_visible = config.PIP_ENABLED

    if not camera.is_opened:
        print("Failed to open camera.")
        return

    start_time = time.perf_counter()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            raw_frame = frame.copy()

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            result = detector.detect(frame, timestamp_ms)

            fh, fw = frame.shape[:2]
            gestures.update(result, fw, fh)

            for obj in objects:
                obj.render(frame)

            frame = renderer.draw(frame, result)

            if pip_visible:
                _draw_pip(frame, raw_frame, fh, fw)

            hint = "C: cube | B: ball | X: remove | V: cam | Q: quit"
            tw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            cv2.putText(
                frame, hint, ((fw - tw) // 2, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA,
            )

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("c"):
                c = CUBE_COLORS[cube_idx % len(CUBE_COLORS)]
                objects.append(Cube(0.5, 0.5, color=c))
                gestures.objects = objects
                cube_idx += 1
            elif key == ord("b"):
                c = BALL_COLORS[ball_idx % len(BALL_COLORS)]
                objects.append(Ball(0.5, 0.5, color=c))
                gestures.objects = objects
                ball_idx += 1
            elif key == ord("v"):
                pip_visible = not pip_visible
            elif key == ord("x") and objects:
                objects.pop()
                gestures.objects = objects
    finally:
        detector.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
