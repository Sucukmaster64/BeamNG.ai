import os
import time
import cv2
import math
import numpy as np
import json

from scipy.spatial.transform import Rotation as R

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera

HOME = r"C:\BeamNG-tech"


LEVEL = "west_coast_usa"
SPAWN_POS = (-717.121, 101.458, 118.675)
SPAWN_ROT = (0, 0, 0, 1)

COLOR_PRESET_PATH = os.path.join("configs", f"colors_{LEVEL}.json")

# Speichern an/aus
SAVE_EVERY_N_FRAMES = 3
OUT_RAW_DIR = "data/raw"
OUT_LABEL_DIR = "data/labels"   # Multi-Class Labels (0..N)
OUT_MASK_DIR = "data/masks"     # Optional: binär (drivable) für Controller
OUT_LABEL_VIS_DIR = "data/labels_vis"

# Klassen-IDs
CLASS_OTHER    = 0
CLASS_ROAD     = 1
CLASS_SHOULDER = 2
CLASS_SIDEWALK = 3
CLASS_TERRAIN  = 4
CLASS_OBST     = 5
CLASS_MARKING  = 6

# Farben, die du per Klick sammelst (Annotation-RGB)
colors_by_class = {
    CLASS_ROAD:     set(),
    CLASS_SHOULDER: set(),
    CLASS_SIDEWALK: set(),
    CLASS_TERRAIN:  set(),
    CLASS_OBST:     set(),
    CLASS_MARKING:  set(),
}

# Aktuelle Klick-Klasse
current_class = CLASS_ROAD


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


def remove_color_from_all_classes(rgb):
    """Entfernt rgb aus allen Klassen."""
    for cls, s in colors_by_class.items():
        if rgb in s:
            s.remove(rgb)


def add_color_to_class(rgb, cls):
    """
    Fügt rgb der Klasse cls hinzu und entfernt es automatisch aus allen anderen Klassen,
    damit es keine Überschneidungen geben kann.
    """
    remove_color_from_all_classes(rgb)
    colors_by_class[cls].add(rgb)


def remove_color_from_class(rgb, cls):
    """Entfernt rgb nur aus der gewählten Klasse (falls vorhanden)."""
    if rgb in colors_by_class[cls]:
        colors_by_class[cls].remove(rgb)
        return True
    return False


def on_mouse(event, x, y, flags, param):
    global current_class

    ann = param.get("ann")
    if ann is None:
        return

    rgb = tuple(ann[y, x, :3].astype(int).tolist())

    # Linksklick: hinzufügen (mit Auto-Remove aus allen anderen Klassen)
    if event == cv2.EVENT_LBUTTONDOWN:
        add_color_to_class(rgb, current_class)
        print(f"ADD {rgb} -> class {current_class} (auto-removed from other classes)")

    # Rechtsklick: entfernen aus aktueller Klasse
    elif event == cv2.EVENT_RBUTTONDOWN:
        ok = remove_color_from_class(rgb, current_class)
        if ok:
            print(f"REMOVE {rgb} from class {current_class}")
        else:
            print(f"NOT FOUND {rgb} in class {current_class}")


def build_label_mask(ann_rgb):
    """
    ann_rgb: HxWx3 uint8 (RGB)
    returns: HxW uint8 label mask with class ids
    """
    h, w = ann_rgb.shape[:2]
    labels = np.zeros((h, w), dtype=np.uint8)  # default OTHER

    # Wichtig: Reihenfolge = Priorität (höher -> überschreibt niedriger).
    # Markings und Obstacles sollen niemals als Road/Shoulder enden.
    priority = [
        CLASS_MARKING,
        CLASS_OBST,
        CLASS_TERRAIN,
        CLASS_SIDEWALK,
        CLASS_SHOULDER,
        CLASS_ROAD,
    ]

    for cls in priority:
        for (r, g, b) in colors_by_class[cls]:
            m = (ann_rgb[:, :, 0] == r) & (ann_rgb[:, :, 1] == g) & (ann_rgb[:, :, 2] == b)
            labels[m] = cls

    return labels


def colorize_labels(lbl):
    """
    Visualisierung der Klassen als farbiges Bild (BGR für OpenCV).
    """
    h, w = lbl.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # BGR Farben (nur zur Anzeige)
    vis[lbl == CLASS_ROAD]     = (0, 255, 0)      # grün
    vis[lbl == CLASS_SHOULDER] = (0, 165, 255)    # orange
    vis[lbl == CLASS_SIDEWALK] = (255, 0, 0)      # blau
    vis[lbl == CLASS_TERRAIN]  = (0, 128, 0)      # dunkelgrün
    vis[lbl == CLASS_OBST]     = (0, 0, 255)      # rot
    vis[lbl == CLASS_MARKING]  = (255, 255, 255)  # weiß (Markierungen auffällig)
    # OTHER bleibt schwarz
    return vis


def drivable_from_labels(lbl, allow_shoulder=False):
    """
    Binäre Drivable-Maske für Debug/Controller.
    WICHTIG: MARKING bleibt IMMER undrivable, egal ob Shoulder erlaubt ist.
    """
    if allow_shoulder:
        m = (lbl == CLASS_ROAD) | (lbl == CLASS_SHOULDER)
    else:
        m = (lbl == CLASS_ROAD)

    # Markierungen niemals befahrbar
    m = m & (lbl != CLASS_MARKING)

    return (m.astype(np.uint8) * 255)


def load_colors_from_json(path):
    """
    Lädt Farben aus JSON und überschreibt colors_by_class.
    JSON Format: {"1": [[r,g,b], ...], "2": [...], ...}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Alles leeren
    for cls in colors_by_class.keys():
        colors_by_class[cls].clear()

    # Laden
    for k, cols in data.items():
        cls = int(k)
        if cls not in colors_by_class:
            continue
        for c in cols:
            colors_by_class[cls].add(tuple(int(x) for x in c))


def save_colors_to_json(path):
    """
    Speichert colors_by_class in JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for cls, s in colors_by_class.items():
        out[str(cls)] = [list(map(int, rgb)) for rgb in sorted(s)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    SAVE = False
    ALLOW_SHOULDER_AS_DRIVABLE = False  # Taste 'h' toggelt

    os.makedirs(OUT_RAW_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_VIS_DIR, exist_ok=True)

    # (3) Preset beim Start laden (falls vorhanden)
    try:
        load_colors_from_json(COLOR_PRESET_PATH)
        print(f"Loaded color preset from: {COLOR_PRESET_PATH}")
    except FileNotFoundError:
        print(f"No preset found at {COLOR_PRESET_PATH}. Click colours to create one, then press 'p' to save.")

    bng = BeamNGpy("localhost", 64256, home=HOME)
    bng.open()

    scenario = Scenario(LEVEL, "multiclass_pick")
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

    print("Multi-class picking:")
    print("Linksklick in 'Annotation' = Farbe zur aktuellen Klasse hinzufügen (auto-remove aus anderen Klassen).")
    print("Rechtsklick in 'Annotation' = Farbe aus aktueller Klasse entfernen.")
    print("Keys: 1=ROAD 2=SHOULDER 3=SIDEWALK 4=TERRAIN 5=OBSTACLE 6=MARKING")
    print("c=print colors, p=save preset, r=reload preset, s=toggle saving, h=toggle shoulder-as-drivable, ESC=quit")

    time.sleep(2)

    cv2.namedWindow("Annotation")
    state = {"ann": None}
    cv2.setMouseCallback("Annotation", on_mouse, state)

    frame_idx = 0

    global current_class
    current_class = CLASS_ROAD

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

        # Labels bauen + anzeigen
        lbl = build_label_mask(ann_rgb)
        lbl_vis = colorize_labels(lbl)
        cv2.imshow("Labels (visual)", lbl_vis)

        # Optional: binäre Drivable-Maske
        drv = drivable_from_labels(lbl, allow_shoulder=ALLOW_SHOULDER_AS_DRIVABLE)
        cv2.imshow("Drivable (derived)", drv)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        # Klasse wechseln
        if key == ord("1"):
            current_class = CLASS_ROAD
            print("current class = ROAD (1)")
        elif key == ord("2"):
            current_class = CLASS_SHOULDER
            print("current class = SHOULDER (2)")
        elif key == ord("3"):
            current_class = CLASS_SIDEWALK
            print("current class = SIDEWALK (3)")
        elif key == ord("4"):
            current_class = CLASS_TERRAIN
            print("current class = TERRAIN (4)")
        elif key == ord("5"):
            current_class = CLASS_OBST
            print("current class = OBSTACLE (5)")
        elif key == ord("6"):
            current_class = CLASS_MARKING
            print("current class = MARKING (6)")

        elif key == ord("h"):
            ALLOW_SHOULDER_AS_DRIVABLE = not ALLOW_SHOULDER_AS_DRIVABLE
            print(f"ALLOW_SHOULDER_AS_DRIVABLE = {ALLOW_SHOULDER_AS_DRIVABLE}")

        elif key == ord("c"):
            print("\nCollected colors:")
            for cls in [CLASS_ROAD, CLASS_SHOULDER, CLASS_SIDEWALK, CLASS_TERRAIN, CLASS_OBST, CLASS_MARKING]:
                print(f"class {cls}: {sorted(colors_by_class[cls])}")

        # (4) Preset speichern / neu laden
        elif key == ord("p"):
            save_colors_to_json(COLOR_PRESET_PATH)
            print(f"Saved preset to: {COLOR_PRESET_PATH}")

        elif key == ord("r"):
            load_colors_from_json(COLOR_PRESET_PATH)
            print(f"Reloaded preset: {COLOR_PRESET_PATH}")

        elif key == ord("s"):
            SAVE = not SAVE
            print(f"Saving: {SAVE}")

        # Speichern
        if SAVE and (frame_idx % SAVE_EVERY_N_FRAMES == 0):
            fid = f"{int(time.time() * 1000)}"
            cv2.imwrite(os.path.join(OUT_RAW_DIR, f"{fid}.png"), colour[:, :, :3])
            cv2.imwrite(os.path.join(OUT_LABEL_DIR, f"{fid}.png"), lbl)   # 0..6 als PNG
            cv2.imwrite(os.path.join(OUT_MASK_DIR, f"{fid}.png"), drv)    # optional binär
            cv2.imwrite(os.path.join(OUT_LABEL_VIS_DIR, f"{fid}.png"), lbl_vis)
            print(f"saved {fid}.png (raw/label/mask/labels_vis)")

        frame_idx += 1

    cv2.destroyAllWindows()
    bng.close()


if __name__ == "__main__":
    main()

