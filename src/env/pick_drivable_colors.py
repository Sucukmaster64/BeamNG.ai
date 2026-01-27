import os
import time
import cv2
import numpy as np

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera

HOME = r"C:\BeamNG-tech"
LEVEL = "west_coast_usa"

SPAWN_POS = (-717.121, 101.458, 118.675)
SPAWN_ROT = (0, 0, 0, 1)

SAVE_EVERY_N_FRAMES = 3
OUT_RAW_DIR = "data/raw"
OUT_MASK_DIR = "data/masks"

# Drivable-Farben (Annotation-RGB!)
DRIVABLE_COLORS = []  # leer lassen -> per Klick sammeln
picked = []


def to_np(img):
    if img is None:
        return None
    try:
        return img if isinstance(img, np.ndarray) else np.array(img)
    except Exception:
        return None


def to_bgr(arr):
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3][:, :, ::-1].copy()
    return arr


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    ann = param.get("ann")
    if ann is None:
        return
    rgb = ann[y, x, :3].astype(int)
    picked.append(tuple(rgb.tolist()))
    print(f"picked RGB at ({x},{y}): {picked[-1]}")


def build_mask_from_colors(ann_rgb, colors):
    """
    ann_rgb: HxWx3 uint8 (RGB)
    colors: list of (R,G,B)
    returns: HxW uint8 mask (0 or 255) or None
    """
    if ann_rgb is None or len(colors) == 0:
        return None

    mask = np.zeros((ann_rgb.shape[0], ann_rgb.shape[1]), dtype=np.uint8)
    for (r, g, b) in colors:
        match = (ann_rgb[:, :, 0] == r) & (ann_rgb[:, :, 1] == g) & (ann_rgb[:, :, 2] == b)
        mask[match] = 255
    return mask


def postprocess_mask(mask):
    """
    Füllt dünne schwarze Lücken (Lane-Markings etc.) in einer binären Maske.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)


def apply_margin(mask, pixels=8):
    """
    Schrumpft die drivable Fläche leicht, damit der Controller Abstand zu Rändern hält.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.erode(mask, k, iterations=1)


def target_from_corridor_band(mask, y0=0.62, y1=0.90, step=4, min_width=100):
    """
    Vorausschauender Zielpunkt: Korridor-Mitte über ein Band weiter oben im Bild.
    Stabiler als Schwerpunkt.
    """
    if mask is None:
        return None

    h, w = mask.shape
    centers = []
    for y in range(int(h * y0), int(h * y1), step):
        xs = np.where(mask[y, :] == 255)[0]
        if xs.size < min_width:
            continue
        centers.append((xs[0] + xs[-1]) / 2.0)

    if not centers:
        return None

    tx = int(np.mean(centers))
    ty = int(h * ((y0 + y1) / 2.0))
    return tx, ty


def forward_drivable_ratio(mask, y0=0.45, y1=0.75, x_margin=0.22):
    """
    Anteil drivable Pixel in einem vorderen Fenster (für vorausschauendes Bremsen).
    mask: 0/255
    """
    if mask is None:
        return 0.0
    h, w = mask.shape
    xa = int(w * x_margin)
    xb = int(w * (1.0 - x_margin))
    ya = int(h * y0)
    yb = int(h * y1)
    roi = mask[ya:yb, xa:xb]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi == 255))


def draw_forward_window(vis_bgr, y0=0.45, y1=0.75, x_margin=0.22):
    """
    Zeichnet das Forward-Fenster ins Overlay.
    """
    h, w = vis_bgr.shape[:2]
    xa = int(w * x_margin)
    xb = int(w * (1.0 - x_margin))
    ya = int(h * y0)
    yb = int(h * y1)
    cv2.rectangle(vis_bgr, (xa, ya), (xb, yb), (0, 255, 255), 2)


def compute_control_from_target(tx, w, steer_prev, kp=0.9, smooth=0.25):
    """
    Ziel-x -> Steering/Throttle. Brake wird später per vorausschauender Logik überschrieben.
    """
    cx = (w - 1) / 2.0
    err = (tx - cx) / cx  # ~[-1, 1]

    steer_raw = float(np.clip(kp * err, -1.0, 1.0))
    steer = (1.0 - smooth) * steer_prev + smooth * steer_raw

    # Basis-Speed: langsamer bei stärkeren Lenkungen
    throttle = float(np.clip(0.22 - 0.12 * abs(steer), 0.08, 0.22))
    brake = 0.0
    return steer, throttle, brake

def make_soft_mask(mask_bin, blur=11):
    """
    mask_bin: 0/255
    returns: float32 0..1
    """
    if mask_bin is None:
        return None
    m = (mask_bin.astype(np.float32) / 255.0)
    m = cv2.GaussianBlur(m, (blur, blur), 0)
    m = np.clip(m, 0.0, 1.0)
    return m



def main():
    # SAVE lokal halten (kein global nötig)
    SAVE = False

    os.makedirs(OUT_RAW_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    bng = BeamNGpy("localhost", 64256, home=HOME)
    bng.open()

    scenario = Scenario(LEVEL, "drivable_annotation")
    vehicle = Vehicle("ego", model="etk800", licence="AI")
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    scenario.make(bng)

    bng.load_scenario(scenario)
    bng.start_scenario()

    cam = Camera(
        name="front_cam",
        bng=bng,
        vehicle=vehicle,
        requested_update_time=0.05,
        pos=(0.0, -2.0, 1.2),
        dir=(0.0, -1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        resolution=(640, 360),
        field_of_view_y=60,
        is_render_colours=True,
        is_render_annotations=True,
        is_render_depth=False
    )

    print("Running. Click on ROAD pixels in the 'Annotation' window to collect drivable colors.")
    print("Press 'c' to confirm colors. Press 's' to toggle saving. Press ESC to quit.")

    time.sleep(2)

    cv2.namedWindow("Annotation")
    state = {"ann": None}
    cv2.setMouseCallback("Annotation", on_mouse, state)

    frame_idx = 0
    confirmed_colors = list(DRIVABLE_COLORS)

    # Steering-Glättung braucht persistenten Zustand
    steer_prev = 0.0

    while True:
        data = cam.poll()
        colour = to_np(data.get("colour"))
        ann = to_np(data.get("annotation"))

        if ann is None or colour is None:
            time.sleep(0.01)
            continue

        ann_rgb = ann[:, :, :3].astype(np.uint8)
        state["ann"] = ann_rgb

        cv2.imshow("Colour", to_bgr(colour))
        cv2.imshow("Annotation", to_bgr(ann_rgb))

        h, w = ann_rgb.shape[:2]

        # Rohmaske aus Farben
        mask = build_mask_from_colors(ann_rgb, confirmed_colors)

        if mask is None:
            # Noch keine Farben bestätigt -> keine Steuerung, nur bremsen
            cv2.imshow("Drivable Mask", np.zeros((h, w), dtype=np.uint8))
            vehicle.control(steering=steer_prev, throttle=0.0, brake=0.3)
        else:
            # Postprocessing: dünne Linien schließen + Abstand zum Rand
            mask_pp = postprocess_mask(mask)
            mask_pp = apply_margin(mask_pp, pixels=6)

            # Vorausschauender Zielpunkt (Band weiter oben)
            target = target_from_corridor_band(mask_pp, y0=0.62, y1=0.90, step=4, min_width=100)


            # Vorwärts-Freiraum: für frühes Bremsen
            free = forward_drivable_ratio(mask_pp, y0=0.45, y1=0.75, x_margin=0.22
)

            # Visualisierung
            vis = cv2.cvtColor(mask_pp, cv2.COLOR_GRAY2BGR)
            draw_forward_window(vis, y0=0.45, y1=0.75, x_margin=0.22)
            cv2.putText(
                vis, f"free={free:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2
            )

            if target is not None:
                tx, ty = target
                cv2.circle(vis, (tx, ty), 6, (0, 0, 255), -1)

                steer_prev, throttle, brake = compute_control_from_target(tx, w, steer_prev)

                # Vorausschauende Bremse:
                # Wenn vorne wenig drivable -> früh bremsen, bevor man „ins Feld“ rollt
                if free < 0.25:
                    throttle = 0.0
                    brake = 0.7
                elif free < 0.40:
                    throttle = min(throttle, 0.10)
                    brake = 0.25
                else:
                    brake = 0.0

                vehicle.control(steering=steer_prev, throttle=throttle, brake=brake)
            else:
                # Unsicher -> abbremsen
                vehicle.control(steering=steer_prev, throttle=0.0, brake=0.4)

            cv2.imshow("Drivable Mask", vis)

            # Speichern: postprozessierte Maske verwenden
            if SAVE and (frame_idx % SAVE_EVERY_N_FRAMES == 0):
                fid = f"{int(time.time() * 1000)}"
                cv2.imwrite(os.path.join(OUT_RAW_DIR, f"{fid}.png"), colour[:, :, :3])
                cv2.imwrite(os.path.join(OUT_MASK_DIR, f"{fid}.png"), mask_pp)
                print(f"saved {fid}.png")

        frame_idx += 1

        # keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            vehicle.control(steering=0.0, throttle=0.0, brake=1.0)
            break

        if key == ord("c"):
            confirmed_colors = sorted(set(picked))
            print("\nConfirmed drivable colors:")
            print(confirmed_colors)

        if key == ord("s"):
            SAVE = not SAVE
            print(f"Saving: {SAVE}")

    cv2.destroyAllWindows()
    bng.close()


if __name__ == "__main__":
    main()
