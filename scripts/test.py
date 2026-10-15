import inspect
import robosuite
from robosuite.robots.single_arm import SingleArm

print("\n--- Metodi controller disponibili ---")
print([name for name in dir(SingleArm) if "controller" in name.lower()])

print("Robosuite:", robosuite.__version__)

print("\n--- load_controller ---")
if hasattr(SingleArm, "_load_controller"):
    print(inspect.getsource(SingleArm._load_controller))
else:
    print("Metodo non presente")
    
print("\n--- reset ---")
print(inspect.getsource(SingleArm.reset))