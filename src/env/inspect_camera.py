from beamngpy.sensors import Camera
import inspect

print("Camera __init__ signature:\n")
print(inspect.signature(Camera.__init__))

print("\nCamera __init__ help:\n")
help(Camera)
