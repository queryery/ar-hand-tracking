import math

import cv2
import numpy as np


def _rot_mat(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,     sx * cy,                cx * cy               ],
    ])


class Object3D:
    def __init__(self, x, y, scale=1.0, color=(255, 255, 255)):
        self.pos = np.array([x, y], dtype=np.float64)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.scale = float(scale)
        self.color = color
        self.size = 60
        self.grabbed = False

    def hit_test(self, px, py, fw, fh):
        raise NotImplementedError

    def render(self, frame):
        raise NotImplementedError


class Cube(Object3D):
    _V = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)

    _EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    def __init__(self, x, y, scale=1.0, color=(0, 180, 255)):
        super().__init__(x, y, scale, color)

    def _proj(self, fw, fh):
        s = self.size * self.scale * 0.5
        v = self._V * s
        R = _rot_mat(*self.rot)
        v = (R @ v.T).T
        cx, cy = self.pos[0] * fw, self.pos[1] * fh
        return [
            (cx + p[0] / (1 + p[2] * 0.002),
             cy + p[1] / (1 + p[2] * 0.002),
             p[2])
            for p in v
        ]

    def hit_test(self, px, py, fw, fh):
        pts = self._proj(fw, fh)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        m = 10
        return min(xs) - m <= px <= max(xs) + m and min(ys) - m <= py <= max(ys) + m

    def render(self, frame):
        fh, fw = frame.shape[:2]
        pts = self._proj(fw, fh)
        ec = (100, 255, 100) if self.grabbed else self.color
        max_z = self.size * self.scale * 0.5

        edge_data = sorted(
            ((pts[a][2] + pts[b][2]) / 2, a, b) for a, b in self._EDGES
        )
        edge_data.reverse()

        for avg_z, a, b in edge_data:
            brightness = max(0.2, min(1.0, 0.5 - avg_z / (max_z + 1) * 0.5))
            c = tuple(int(v * brightness) for v in ec)
            gc = tuple(v // 3 for v in c)
            p1 = (int(pts[a][0]), int(pts[a][1]))
            p2 = (int(pts[b][0]), int(pts[b][1]))
            cv2.line(frame, p1, p2, gc, 3, cv2.LINE_AA)
            cv2.line(frame, p1, p2, c, 1, cv2.LINE_AA)

        for p in pts:
            brightness = max(0.2, min(1.0, 0.5 - p[2] / (max_z + 1) * 0.5))
            c = tuple(int(v * brightness) for v in ec)
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, c, -1, cv2.LINE_AA)


class Ball(Object3D):
    _LAT = 0.866  # sqrt(3/4)

    def __init__(self, x, y, scale=1.0, color=(50, 200, 100)):
        super().__init__(x, y, scale, color)

    def hit_test(self, px, py, fw, fh):
        r = self.size * self.scale * 0.5
        cx, cy = self.pos[0] * fw, self.pos[1] * fh
        return math.hypot(px - cx, py - cy) <= r + 10

    def render(self, frame):
        fh, fw = frame.shape[:2]
        r = int(self.size * self.scale * 0.5)
        cx, cy = int(self.pos[0] * fw), int(self.pos[1] * fh)
        if r < 3:
            return

        R = _rot_mat(*self.rot)
        ec = (100, 255, 100) if self.grabbed else self.color

        # 3 great circles
        self._wire(frame, R, cx, cy, r, [1, 0, 0], [0, 0, 1], ec)
        self._wire(frame, R, cx, cy, r, [1, 0, 0], [0, 1, 0], ec)
        self._wire(frame, R, cx, cy, r, [0, 1, 0], [0, 0, 1], ec)

        # 2 latitude circles
        s = self._LAT
        self._wire(frame, R, cx, cy, r, [s, 0, 0], [0, 0, s], ec, [0, 0.5, 0])
        self._wire(frame, R, cx, cy, r, [s, 0, 0], [0, 0, s], ec, [0, -0.5, 0])

    def _wire(self, frame, R, cx, cy, r, a1, a2, color, center=None):
        a1 = np.array(a1, np.float64)
        a2 = np.array(a2, np.float64)
        N = 48
        pts = []
        for i in range(N):
            a = 2 * math.pi * i / N
            local = (a1 * math.cos(a) + a2 * math.sin(a)) * r
            if center is not None:
                local += np.array(center, np.float64) * r
            p = R @ local
            pts.append((cx + p[0], cy + p[1], p[2]))

        for i in range(N):
            j = (i + 1) % N
            avg_z = (pts[i][2] + pts[j][2]) / 2
            b = max(0.12, min(1.0, 0.5 - avg_z / (r + 1) * 0.5))
            c = tuple(int(v * b) for v in color)
            gc = tuple(v // 3 for v in c)
            p1 = (int(pts[i][0]), int(pts[i][1]))
            p2 = (int(pts[j][0]), int(pts[j][1]))
            cv2.line(frame, p1, p2, gc, 3, cv2.LINE_AA)
            cv2.line(frame, p1, p2, c, 1, cv2.LINE_AA)
