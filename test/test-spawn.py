from beamngpy import BeamNGpy

HOME = r"C:\BeamNG-tech"
LEVEL = "italy"

bng = BeamNGpy("localhost", 64256, home=HOME)
bng.open()

# WICHTIG: Level muss geladen sein
bng.load_level(LEVEL)

spawnpoints = bng.get_level_spawnpoints()
print(f"Found {len(spawnpoints)} spawnpoints\n")

for i, sp in enumerate(spawnpoints):
    print(f"[{i}] {sp}")

bng.close()
