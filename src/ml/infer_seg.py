import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera


# ----------------------------
# Model loading
# ----------------------------
def load_model(model_path: Path, device: torch.device):
    """
    Lädt ENet + Checkpoint.
    num_classes wird aus head.bias abgeleitet (bei dir: 7).
    """
    try:
        from src.ml.models.enet import ENet
    except Exception as e:
        raise ImportError(
            "Konnte ENet nicht importieren. Erwartet: src/ml/models/enet.py (class ENet).\n"
            "Passe ggf. den Import in infer_seg.py an."
        ) from e

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # num_classes aus Checkpoint ableiten
    if isinstance(state, dict) and "head.bias" in state:
        num_classes = int(state["head.bias"].shape[0])
    elif isinstance(state, dict) and "head.weight" in state:
        # ENet head.weight typischerweise [out_channels, num_classes, k, k]
        num_classes = int(state["head.weight"].shape[1])
    elif isinstance(ckpt, dict) and "num_classes" in ckpt:
        num_classes = int(ckpt["num_classes"])
    else:
        raise RuntimeError("Konnte num_classes nicht aus dem Checkpoint ableiten (head.bias/head.weight/num_classes fehlt).")

    model = ENet(num_classes=num_classes)

    # DataParallel "module." entfernen
    new_state = {}
    if isinstance(state, dict):
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
    else:
        new_state = state

    # strict=False damit z.B. nicht geladene buffers o.ä. nicht killen
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model, num_classes


# ----------------------------
# Image utils
# ----------------------------
def preprocess(img_bgr: np.ndarray, width: int, height: int, device: torch.device):
    """
    BGR -> (1,3,H,W) float32 [0..1]
    """
    img = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    return x, img  # resized BGR für Anzeige


def make_color_lut(num_classes: int):
    rng = np.random.default_rng(12345)
    lut = rng.integers(low=0, high=255, size=(max(num_classes, 1), 3), dtype=np.uint8)
    if num_classes > 0:
        lut[0] = np.array([0, 0, 0], dtype=np.uint8)
    return lut


def colorize_mask(mask: np.ndarray, lut: np.ndarray):
    return lut[mask]  # (H,W,3) uint8


def overlay(base_bgr: np.ndarray, seg_bgr: np.ndarray, alpha: float):
    return cv2.addWeighted(base_bgr, 1.0 - alpha, seg_bgr, alpha, 0.0)


# ----------------------------
# BeamNG camera creation (BeamNGpy 1.35)
# ----------------------------
def create_camera(args, bng: BeamNGpy, vehicle: Vehicle):
    """
    BeamNGpy 1.35: Camera hat Parameter wie field_of_view_y + is_render_colours.
    Kamera sauber über vehicle.sensors.attach(...) nutzen (kein cam.attach()).
    """
    cam = Camera(
        "front_cam",                     # name (required)
        bng,                             # bng (required)
        vehicle=vehicle,                 # optional, aber praktisch
        requested_update_time=0.0,       # 0.0 => so schnell wie möglich
        pos=(0.0, 1.6, 1.2),
        dir=(0, 1, 0),
        up=(0, 0, 1),
        resolution=(args.cam_w, args.cam_h),
        field_of_view_y=float(args.fov),
        is_render_colours=True,
        is_render_depth=False,
        is_render_annotations=False,
        is_streaming=False,
    )

    # attach via sensors-Manager
    vehicle.sensors.attach("front_cam", cam)
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=64256)

    # Map/Scenario/Vehicle
    ap.add_argument("--level", type=str, default="west_coast_usa")
    ap.add_argument("--scenario", type=str, default="seg_infer")
    ap.add_argument("--vehicle", type=str, default="ego")

    # Spawn (deine Werte)
    ap.add_argument("--spawn_x", type=float, default=-717.121)
    ap.add_argument("--spawn_y", type=float, default=101.458)
    ap.add_argument("--spawn_z", type=float, default=118.675)
    ap.add_argument("--spawn_qx", type=float, default=0.0)
    ap.add_argument("--spawn_qy", type=float, default=0.0)
    ap.add_argument("--spawn_qz", type=float, default=0.0)
    ap.add_argument("--spawn_qw", type=float, default=1.0)

    # Model/Inference
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--cam_w", type=int, default=640)
    ap.add_argument("--cam_h", type=int, default=360)
    ap.add_argument("--net_w", type=int, default=512)
    ap.add_argument("--net_h", type=int, default=288)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--drivable_id", type=int, default=1)
    ap.add_argument("--show_drivable_only", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # BeamNG.tech auto-start
    ap.add_argument("--beamng_home", type=str, default=None, help=r'Pfad zu BeamNG.tech, z.B. "C:\BeamNG-tech"')
    ap.add_argument("--no_launch", action="store_true", help="Nicht starten, nur verbinden (wenn BeamNG bereits läuft)")

    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model nicht gefunden: {model_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, num_classes = load_model(model_path, device)
    lut = make_color_lut(num_classes).copy()
    if 0 <= args.drivable_id < lut.shape[0]:
        lut[args.drivable_id] = np.array([0, 255, 0], dtype=np.uint8)  # BGR grün

    # ---- BeamNG connection + scenario
    if args.beamng_home:
        print("[INFO] Starte BeamNG.tech via BeamNGpy...")
        bng = BeamNGpy(args.host, args.port, home=args.beamng_home)
        bng.open(launch=not args.no_launch)
    else:
        # Ohne home: erwartet, dass BeamNG schon läuft
        bng = BeamNGpy(args.host, args.port)
        bng.open(launch=False)

    scenario = Scenario(args.level, args.scenario)
    vehicle = Vehicle(args.vehicle, model="etk800", licence="SEG")

    spawn_pos = (args.spawn_x, args.spawn_y, args.spawn_z)
    spawn_rot = (args.spawn_qx, args.spawn_qy, args.spawn_qz, args.spawn_qw)
    scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot)

    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    # Camera
    _cam = create_camera(args, bng, vehicle)

    print(f"Device: {device} | num_classes: {num_classes}")
    print("Taste: q = quit | d = toggle drivable-only")

    show_drivable_only = args.show_drivable_only
    last_t = time.time()
    fps_smooth = 0.0

    try:
        while True:
            # Sensor polling
            vehicle.sensors.poll()
            cam_data = vehicle.sensors["front_cam"]

            img = cam_data.get("colour", None)
            if img is None:
                # manchmal kommt am Anfang noch nichts
                time.sleep(0.01)
                continue

            # BeamNGpy liefert i.d.R. RGB; wir wollen BGR für OpenCV
            img = np.asarray(img)
            if img.size == 0:
                time.sleep(0.01)
                continue

            # Erwartet HxWx3
            if img.ndim == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                # Fallback: wenn Format unerwartet ist
                time.sleep(0.01)
                continue

            x, img_resized_bgr = preprocess(img_bgr, args.net_w, args.net_h, device)

            with torch.no_grad():
                logits = model(x)  # (1,C,H,W)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            seg_bgr = colorize_mask(pred, lut)

            if show_drivable_only:
                dr = (pred == np.uint8(args.drivable_id)).astype(np.uint8) * 255
                dr_bgr = cv2.cvtColor(dr, cv2.COLOR_GRAY2BGR)
                view = overlay(img_resized_bgr, dr_bgr, alpha=args.alpha)
                title = "Infer (drivable only)"
            else:
                view = overlay(img_resized_bgr, seg_bgr, alpha=args.alpha)
                title = "Infer (overlay)"

            # FPS
            now = time.time()
            dt = now - last_t
            last_t = now
            fps = (1.0 / dt) if dt > 0 else 0.0
            fps_smooth = fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * fps)

            cv2.putText(
                view,
                f"FPS: {fps_smooth:.1f} | classes: {num_classes} | drivable_id: {args.drivable_id}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(title, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_drivable_only = not show_drivable_only

    finally:
        cv2.destroyAllWindows()
        try:
            bng.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
