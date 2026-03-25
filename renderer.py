import cv2
import numpy as np

import config

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}


class Renderer:
    def __init__(self):
        pass

    def draw(self, frame: np.ndarray, result) -> np.ndarray:
        if result is None or not result.hand_landmarks:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        for hand_landmarks in result.hand_landmarks:
            points = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks],
                dtype=np.int32,
            )
            
            for a, b in HAND_CONNECTIONS:
                cv2.line(
                    overlay, points[a], points[b],
                    config.SKEL_GLOW, config.SKEL_GLOW_WIDTH, cv2.LINE_AA,
                )

            for a, b in HAND_CONNECTIONS:
                cv2.line(
                    overlay, points[a], points[b],
                    config.SKEL_COLOR, config.SKEL_WIDTH, cv2.LINE_AA,
                )

            for idx, pt in enumerate(points):
                r = config.SKEL_TIP_RADIUS if idx in FINGERTIP_INDICES else config.SKEL_DOT_RADIUS
                cv2.circle(overlay, pt, r, config.SKEL_GLOW, -1, cv2.LINE_AA)
                cv2.circle(overlay, pt, max(1, r - 1), config.SKEL_COLOR, -1, cv2.LINE_AA)

            thumb, index = points[4], points[8]
            pdist = float(np.linalg.norm(thumb - index))
            if pdist < 50:
                mid = (thumb + index) // 2
                alpha = max(0.0, 1.0 - pdist / 50)
                pr = int(4 + alpha * 8)
                cv2.circle(
                    overlay, tuple(mid), pr,
                    config.SKEL_PINCH_COLOR, 1, cv2.LINE_AA,
                )

        cv2.addWeighted(overlay, config.SKEL_ALPHA, frame, 1.0 - config.SKEL_ALPHA, 0, frame)
        return frame
