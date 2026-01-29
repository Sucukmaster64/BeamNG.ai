# src/ml/infer_seg.py
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera


# ---------------------------
# Model loading
# ---------------------------

def load_model(model_path: Path, device: torch.device):
    """
    Lädt ENet und leitet num_classes automatisch aus dem Checkpoint ab.
    Erwartet: ENet-Klasse unter src/ml/models/enet.py (class ENet).
    """
    try:
        from src.ml.models.enet import ENet
    except Exception as e:
        raise ImportError(
            "Konnte ENet nicht importieren. Erwartet: src/ml/models/enet.py mit class ENet.\n"
            "Passe ggf. den Import in infer_seg.py an."
        ) from e

    ckpt = torch.load(model_path, map_location=device)

    # state_dict kann entweder direkt ckpt sein oder unter 'state_dict' liegen
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint-Format unerwartet: state_dict ist kein dict.")

    # num_classes ableiten: bei ENet ist der Head oft [out_channels, classes, k, k] und bias [classes]
    if "head.bias" in state:
        num_classes = int(state["head.bias"].shape[0])
    elif "head.weight" in state:
        # head.weight: [out, classes, k, k]
        num_classes = int(state["head.weight"].shape[1])
    else:
        raise RuntimeError(
            "Konnte num_classes nicht ableiten (head.bias/head.weight fehlt im state_dict)."
        )

    model = ENet(num_classes=num_classes)

    # Falls DataParallel: "module." entfernen
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    # strict=False, damit es nicht crasht, wenn Kleinigkeiten abweichen
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[WARN] Missing keys beim Laden:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] Unexpected keys beim Laden:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.to(device)
    model.eval()
    return model, num_classes


# ---------------------------
# Image helpers
# ---------------------------

def preprocess(img_bgr: np.ndarray, width: int, height: int, device: torch.device):
    """
    BGR (OpenCV) -> Tensor (1,3,H,W) float32 [0..1]
    """
    img = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    return x, img  # resized BGR fürs Overlay


def make_color_lut(num_classes: int):
    rng = np.random.default_rng(12345)
    lut = rng.integers(low=0, high=255, size=(max(num_classes, 1), 3), dtype=np.uint8)
    if num_classes > 0:
        lut[0] = np.array([0, 0, 0], dtype=np.uint8)
    return lut


def colorize_mask(mask: np.ndarray, lut: np.ndarray):
    return lut[mask]


def overlay(base_bgr: np.ndarray, seg_bgr: np.ndarray, alpha: float):
    return cv2.addWeighted(base_bgr, 1.0 - alpha, seg_bgr, alpha, 0.0)


def to_bgr_image(colour, w: int, h: int):
    """
    Konvertiert verschiedene BeamNGpy-Rückgabeformate zu OpenCV-BGR np.ndarray (H,W,3).
    Fix für:
      - empty frames (None)
      - falsche Typen (nicht ndarray)
      - bytes/raw buffer
      - RGBA/RGB nach BGR
    """
    if colour is None:
        return None

    # ndarray?
    if isinstance(colour, np.ndarray):
        img = colour
    elif isinstance(colour, (list, tuple)):
        img = np.array(colour)
    elif isinstance(colour, (bytes, bytearray)):
        buf = np.frombuffer(colour, dtype=np.uint8)
        if buf.size == w * h * 3:
            img = buf.reshape((h, w, 3))
        elif buf.size == w * h * 4:
            img = buf.reshape((h, w, 4))
        else:
            return None
    elif isinstance(colour, dict) and "data" in colour:
        return to_bgr_image(colour["data"], w, h)
    else:
        return None

    if img is None or getattr(img, "size", 0) == 0:
        return None

    # Kanalzahl normalisieren
    if img.ndim == 3 and img.shape[2] == 4:
        # meist RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        # oft RGB
        # Manche BeamNGpy-Versionen liefern bereits BGR. Wenn es komisch aussieht,
        # kannst du die nächste Zeile auskommentieren.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        return None

    if img.ndim != 3 or img.shape[2] != 3:
        return None
    return img


# ---------------------------
# BeamNG helpers
# ---------------------------

def open_beamng(args):
    """
    Startet/öffnet BeamNG.tech robust über BeamNGpy.
    - Kein 'launch=' im __init__
    - home optional (z.B. C:\\BeamNG-tech)
    """
    bng = BeamNGpy(args.host, args.port, home=args.beamng_home)

    print("[INFO] Starte BeamNG.tech via BeamNGpy...")
    # BeamNGpy.open hat je nach Version launch-Parameter; wir versuchen es robust.
    try:
        bng.open(launch=True)
    except TypeError:
        # ältere Signatur ohne launch
        bng.open()

    return bng


def create_camera(args, bng, vehicle: Vehicle):
    """
    BeamNGpy (neu) erwartet mindestens: Camera(name, bng, ...)
    Je nach Version attach() oder vehicle.attach_sensor(...) / vehicle.poll_sensors()
    Wir bauen es robust, aber primär für die neue API.
    """
    cam = None
    last_err = None

    # Versuch A: neue Signatur mit name, bng, vehicle, field_of_view_y
    try:
        cam = Camera(
            name="front_cam",
            bng=bng,
            vehicle=vehicle,
            pos=(0.0, 1.6, 1.2),
            dir=(0, 1, 0),
            up=(0, 0, 1),
            resolution=(args.cam_w, args.cam_h),
            field_of_view_y=args.fov,
            colour=True,
            depth=False,
            annotation=False,
        )
    except TypeError as e:
        last_err = e

    # Versuch B: manche Versionen nutzen "fov" statt "field_of_view_y"
    if cam is None:
        try:
            cam = Camera(
                name="front_cam",
                bng=bng,
                vehicle=vehicle,
                pos=(0.0, 1.6, 1.2),
                dir=(0, 1, 0),
                up=(0, 0, 1),
                resolution=(args.cam_w, args.cam_h),
                fov=args.fov,
                colour=True,
                depth=False,
                annotation=False,
            )
        except TypeError as e:
            last_err = e

    # Versuch C: ohne FOV
    if cam is None:
        try:
            cam = Camera(
                name="front_cam",
                bng=bng,
                vehicle=vehicle,
                pos=(0.0, 1.6, 1.2),
                dir=(0, 1, 0),
                up=(0, 0, 1),
                resolution=(args.cam_w, args.cam_h),
                colour=True,
                depth=False,
                annotation=False,
            )
        except TypeError as e:
            last_err = e

    if cam is None:
        raise TypeError(
            "Konnte Camera nicht initialisieren (BeamNGpy API passt nicht).\n"
            f"Letzter Fehler: {last_err}"
        )

    # Attach: neue API hat oft cam.attach()
    if hasattr(cam, "attach"):
        cam.attach()
    else:
        # Fallback für ältere API
        vehicle.attach_sensor("front_cam", cam)

    return cam



# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=64256)
    ap.add_argument("--beamng_home", type=str, default=None, help=r'Pfad zu BeamNG.tech, z.B. "C:\BeamNG-tech"')

    ap.add_argument("--level", type=str, default="west_coast_usa")
    ap.add_argument("--scenario", type=str, default="seg_infer")
    ap.add_argument("--vehicle", type=str, default="ego")

    ap.add_argument("--model", type=str, required=True, help="Pfad zu best.pt / checkpoint")
    ap.add_argument("--cam_w", type=int, default=640)
    ap.add_argument("--cam_h", type=int, default=360)
    ap.add_argument("--net_w", type=int, default=512, help="Input-Breite fürs Netz (wie Training)")
    ap.add_argument("--net_h", type=int, default=288, help="Input-Höhe fürs Netz (wie Training)")
    ap.add_argument("--fov", type=float, default=70.0)

    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay-Transparenz")
    ap.add_argument("--drivable_id", type=int, default=1, help="Klassen-ID für 'drivable' (falls vorhanden)")
    ap.add_argument("--show_drivable_only", action="store_true")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--sleep_after_start", type=float, default=1.0, help="Wartezeit nach Szenario-Start (sek)")
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

    # Optional: drivable grün
    if 0 <= args.drivable_id < lut.shape[0]:
        lut[args.drivable_id] = np.array([0, 255, 0], dtype=np.uint8)

    # BeamNG
    bng = open_beamng(args)

    scenario = Scenario(args.level, args.scenario)
    vehicle = Vehicle(args.vehicle, model="etk800", licence="SEG")
    LEVEL = "west_coast_usa"
    SPAWN_POS = (-717.121, 101.458, 118.675)
    SPAWN_ROT = (0, 0, 0, 1)

    # optional: args.level überschreiben, damit es garantiert passt
    args.level = LEVEL

    scenario = Scenario(args.level, args.scenario)
    vehicle = Vehicle(args.vehicle, model="etk800", licence="SEG")
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)


    # Szenario bauen/laden/starten
    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    # Manche Setups brauchen einen Moment bis Sensors Daten liefern
    time.sleep(max(0.0, args.sleep_after_start))

    # Kamera an Vehicle hängen
    _cam = create_camera(args, bng, vehicle)

    print(f"Device: {device} | num_classes: {num_classes}")
    print("Taste: q = quit | d = toggle drivable-only")

    show_drivable_only = args.show_drivable_only
    last_t = time.time()
    fps_smooth = 0.0

    try:
        while True:
            # bevorzugt: neue API
            colour = None
            if hasattr(_cam, "poll"):
                data = _cam.poll()
                # je nach Version "colour" oder "color"
                colour = data.get("colour", None)
                if colour is None:
                    colour = data.get("color", None)
            else:
                # fallback: alte API
                sensors = vehicle.poll_sensors()
                cam_data = sensors.get("front_cam", {})
                colour = cam_data.get("colour", None)
                if colour is None:
                    colour = cam_data.get("color", None)

            img_bgr = to_bgr_image(colour, args.cam_w, args.cam_h)
            if img_bgr is None:
                continue


            x, img_resized_bgr = preprocess(img_bgr, args.net_w, args.net_h, device)

            with torch.no_grad():
                logits = model(x)  # (1,C,H,W)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            if show_drivable_only:
                dr = (pred == np.uint8(args.drivable_id)).astype(np.uint8) * 255
                dr_bgr = cv2.cvtColor(dr, cv2.COLOR_GRAY2BGR)
                view = overlay(img_resized_bgr, dr_bgr, alpha=args.alpha)
                title = "Infer (drivable only)"
            else:
                seg_bgr = colorize_mask(pred, lut)
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
