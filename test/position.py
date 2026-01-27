from beamngpy import BeamNGpy
import time

beamng = BeamNGpy('localhost', 64256)
beamng.open()

vehicle = next(iter(beamng.get_current_vehicles().values()))
print("Fahrzeug-ID:", vehicle.vid)

try:
    vehicle.update_vehicle()
    x, y, z = vehicle.state['pos']
    print(x, y, z)

except KeyboardInterrupt:
    pass

beamng.close()
