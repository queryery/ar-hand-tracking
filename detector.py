import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

import config

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

DETECT_WIDTH = 320
DETECT_HEIGHT = 240


def ensure_model():
    if os.path.exists(config.MODEL_PATH):
        return
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
    print("Model downloaded.")


class HandDetector:
    def __init__(self):
        ensure_model()
        self._landmarker = None
        self._result = None
        self._frame_count = 0
        self._detect_interval = 2

    def start(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        return self

    def detect(self, frame: np.ndarray, timestamp_ms: int):
        self._frame_count += 1
        if self._frame_count % self._detect_interval != 0 and self._result is not None:
            return self._result

        small = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._result

    def stop(self):
        if self._landmarker is not None:
            self._landmarker.close()
