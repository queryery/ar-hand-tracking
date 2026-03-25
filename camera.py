import cv2
import config


class Camera:
    def __init__(self, index=None):
        if index is None:
            index = config.CAMERA_INDEX
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        self._cap.grab()
        ret, frame = self._cap.retrieve()
        if not ret:
            return None
        return frame

    @property
    def is_opened(self):
        return self._cap.isOpened()

    def stop(self):
        self._cap.release()
