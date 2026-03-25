# AR Hand Tracking

Real-time AR hand tracking with 3D object manipulation using MediaPipe and OpenCV.

Grab, move, rotate and scale 3D wireframe objects with your hands — no special hardware needed, just a webcam.

## Features

- **Hand tracking** — clean wireframe skeleton overlay via MediaPipe
- **3D objects** — spawn cubes and balls with colored wireframe rendering
- **Grab & move** — pinch on an object to pick it up and drag it around
- **Rotate** — rotate your arm while grabbing to spin the object
- **Scale** — pinch with both hands on an object and spread/squeeze to resize
- **Live PiP** — toggleable raw camera feed picture-in-picture

## Getting Started

```bash
git clone https://github.com/queryery/ar-hand-tracking.git
cd ar-hand-tracking
python main.py
```

Dependencies install automatically on first run. The hand tracking model downloads automatically too.

**Requires:** Python 3.9+ and a webcam.

## Controls

| Key | Action |
|-----|--------|
| **C** | Add a cube |
| **B** | Add a ball |
| **X** | Remove last object |
| **V** | Toggle live camera PiP |
| **Q** / **Esc** | Quit |

## Gestures

| Gesture | Action |
|---------|--------|
| One-hand pinch on object | Grab and move |
| Rotate arm while grabbing | Rotate object |
| Two-hand pinch on object | Scale (spread = bigger, squeeze = smaller) |

## Config

Edit `config.py` to tweak camera, skeleton appearance, detection confidence, and PiP settings.

## License

MIT
