import time
from beamngpy import BeamNGpy, Scenario, Vehicle

HOME = r"C:\BeamNG-tech"
LEVEL = "west_coast_usa"

beamng = BeamNGpy("localhost", 64256, home=HOME)
beamng.open()

scenario = Scenario(LEVEL, "connection_test")

vehicle = Vehicle(
    vid="ego",
    model="etk800",
    licence="AI"
)

# Bewährter, sicherer Spawnpunkt auf west_coast_usa
# (asphaltierter Bereich, garantiert Boden)
spawn_pos = (-717.121, 101.458, 118.675)
spawn_rot = (0, 0, 0, 1)

scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot)
scenario.make(beamng)

beamng.load_scenario(scenario)
beamng.start_scenario()

print("BeamNG.tech connection successful")

# Warten, damit BeamNG vollständig laden kann
time.sleep(15)

input("Press ENTER to close BeamNG.tech...")

beamng.close()
