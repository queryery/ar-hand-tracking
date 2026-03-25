import math

import numpy as np

PINCH_ON = 0.045
PINCH_OFF = 0.065
POS_SMOOTH = 0.30
ROT_SMOOTH = 0.25
SCALE_SMOOTH = 0.30

_THUMB = 4
_INDEX = 8
_WRIST = 0
_IDX_MCP = 5
_MID_MCP = 9
_RING_MCP = 13


class GestureEngine:
    def __init__(self, objects):
        self.objects = objects
        self._pinch_state = [False, False]

        self._grab_hand = None
        self._grab_obj = None
        self._grab_off = np.zeros(2, np.float64)
        self._grab_angle = 0.0
        self._grab_rot = np.zeros(3, np.float64)

        self._scaling = False
        self._scale_obj = None
        self._scale_d0 = 0.0
        self._scale_s0 = 1.0
        self._scale_mid_off = np.zeros(2, np.float64)

    def update(self, result, fw, fh):
        for o in self.objects:
            o.grabbed = False

        if self._grab_obj and self._grab_obj not in self.objects:
            self._grab_obj, self._grab_hand = None, None
        if self._scale_obj and self._scale_obj not in self.objects:
            self._scale_obj, self._scaling = None, False

        if result is None or not result.hand_landmarks:
            self._release()
            return

        hands = result.hand_landmarks
        n = len(hands)
        pinches = self._get_pinches(hands, n)

        both_pinch = n >= 2 and pinches[0][0] and pinches[1][0]

        if both_pinch:
            if self._scaling:
                self._continue_scale(pinches)
                return
            if self._start_scale(pinches, fw, fh):
                return

        if not both_pinch and self._scaling:
            self._scaling = False
            self._scale_obj = None

        self._do_grab(pinches, n, fw, fh)

    def _get_pinches(self, hands, n):
        pinches = []
        for i in range(n):
            lm = hands[i]
            t, ix = lm[_THUMB], lm[_INDEX]
            w = lm[_WRIST]
            idx_m, mid_m, ring_m = lm[_IDX_MCP], lm[_MID_MCP], lm[_RING_MCP]

            d = math.hypot(t.x - ix.x, t.y - ix.y)

            if i < len(self._pinch_state):
                pinching = d < PINCH_OFF if self._pinch_state[i] else d < PINCH_ON
                self._pinch_state[i] = pinching
            else:
                pinching = d < PINCH_ON

            pp = ((t.x + ix.x) / 2, (t.y + ix.y) / 2)

            avg_x = (idx_m.x + mid_m.x + ring_m.x) / 3 - w.x
            avg_y = (idx_m.y + mid_m.y + ring_m.y) / 3 - w.y
            ang = math.atan2(avg_y, avg_x)

            pinches.append((pinching, pp, ang))
        return pinches

    def _continue_scale(self, pinches):
        pp0, pp1 = pinches[0][1], pinches[1][1]
        dist = math.hypot(pp0[0] - pp1[0], pp0[1] - pp1[1])
        mid_x = (pp0[0] + pp1[0]) / 2
        mid_y = (pp0[1] + pp1[1]) / 2

        if self._scale_d0 > 1e-6:
            target = float(np.clip(
                self._scale_s0 * dist / self._scale_d0, 0.2, 6.0,
            ))
            self._scale_obj.scale += (target - self._scale_obj.scale) * SCALE_SMOOTH

        tx = mid_x + self._scale_mid_off[0]
        ty = mid_y + self._scale_mid_off[1]
        self._scale_obj.pos[0] += (tx - self._scale_obj.pos[0]) * POS_SMOOTH
        self._scale_obj.pos[1] += (ty - self._scale_obj.pos[1]) * POS_SMOOTH
        self._scale_obj.grabbed = True
        self._grab_hand, self._grab_obj = None, None

    def _start_scale(self, pinches, fw, fh):
        pp0, pp1 = pinches[0][1], pinches[1][1]
        px0, py0 = pp0[0] * fw, pp0[1] * fh
        px1, py1 = pp1[0] * fw, pp1[1] * fh
        dist = math.hypot(pp0[0] - pp1[0], pp0[1] - pp1[1])
        mid_x = (pp0[0] + pp1[0]) / 2
        mid_y = (pp0[1] + pp1[1]) / 2

        for obj in self.objects:
            if obj.hit_test(px0, py0, fw, fh) or obj.hit_test(px1, py1, fw, fh):
                self._scaling = True
                self._scale_obj = obj
                self._scale_d0 = max(dist, 0.08)
                self._scale_s0 = obj.scale
                self._scale_mid_off[0] = obj.pos[0] - mid_x
                self._scale_mid_off[1] = obj.pos[1] - mid_y
                obj.grabbed = True
                self._grab_hand, self._grab_obj = None, None
                return True
        return False

    def _do_grab(self, pinches, n, fw, fh):
        if self._grab_obj is not None:
            best, best_d = None, float("inf")
            for i in range(n):
                if pinches[i][0]:
                    pp = pinches[i][1]
                    d = math.hypot(
                        pp[0] - self._grab_obj.pos[0],
                        pp[1] - self._grab_obj.pos[1],
                    )
                    if d < best_d:
                        best_d, best = d, i

            if best is not None:
                pp, ang = pinches[best][1], pinches[best][2]

                tx = pp[0] + self._grab_off[0]
                ty = pp[1] + self._grab_off[1]
                self._grab_obj.pos[0] += (tx - self._grab_obj.pos[0]) * POS_SMOOTH
                self._grab_obj.pos[1] += (ty - self._grab_obj.pos[1]) * POS_SMOOTH

                da = ang - self._grab_angle
                target_ry = self._grab_rot[1] - da * 3.0
                target_rx = self._grab_rot[0] + da * 1.0
                self._grab_obj.rot[1] += (target_ry - self._grab_obj.rot[1]) * ROT_SMOOTH
                self._grab_obj.rot[0] += (target_rx - self._grab_obj.rot[0]) * ROT_SMOOTH

                self._grab_obj.grabbed = True
                self._grab_hand = best
            else:
                self._grab_hand, self._grab_obj = None, None
            return

        for i in range(min(n, 2)):
            if not pinches[i][0]:
                continue
            pp = pinches[i][1]
            px, py = pp[0] * fw, pp[1] * fh
            closest, cd = None, float("inf")
            for obj in self.objects:
                if obj.hit_test(px, py, fw, fh):
                    d = math.hypot(px - obj.pos[0] * fw, py - obj.pos[1] * fh)
                    if d < cd:
                        cd, closest = d, obj
            if closest is not None:
                self._grab_hand = i
                self._grab_obj = closest
                self._grab_off[0] = closest.pos[0] - pp[0]
                self._grab_off[1] = closest.pos[1] - pp[1]
                self._grab_angle = pinches[i][2]
                self._grab_rot = closest.rot.copy()
                closest.grabbed = True
                return

    def _release(self):
        self._grab_hand, self._grab_obj = None, None
        self._scaling = False
        self._scale_obj = None
