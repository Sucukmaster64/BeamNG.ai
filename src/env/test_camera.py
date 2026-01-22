import time
import cv2
import numpy as np

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera

HOME = r"C:\BeamNG-tech"
LEVEL = "west_coast_usa"

SPAWN_POS = (-717.121, 101.458, 118.675)
SPAWN_ROT = (0, 0, 0, 1)

def to_bgr(img):
    if img is None:
        return None
    # Accept both PIL Image and numpy array inputs.
    try:
        arr = img if isinstance(img, np.ndarray) else np.array(img)
    except Exception:
        return None

    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3][:, :, ::-1].copy()
    return arr

def main():
    bng = BeamNGpy("localhost", 64256, home=HOME)
    bng.open()

    scenario = Scenario(LEVEL, "camera_test")
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
        # Place camera at the vehicle's front (approx. front bumper)
        pos=(0.0, -2.0, 1.2),
        # Face forward along vehicle's heading
        dir=(0.0, -1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        resolution=(640, 360),
        field_of_view_y=60,
        is_render_colours=True,
        is_render_annotations=True,
        is_render_depth=False
    )

    # Newer beamngpy camera API: Camera is created with a vehicle
    # passed to the constructor and does not implement `attach()`.
    # Do not call `vehicle.attach_sensor` here to avoid AttributeError.

    print("Camera attached via vehicle. Press ESC to exit.")
    time.sleep(2)

    while True:
        data = cam.poll()

        colour = data.get("colour")
        annotation = data.get("annotation")

        frame = to_bgr(colour)
        if frame is not None:
            cv2.imshow("BeamNG Colour", frame)

        if annotation is not None:
            ann_bgr = to_bgr(annotation)
            cv2.imshow("BeamNG Annotation", ann_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    bng.close()

if __name__ == "__main__":
    main()
