from .manipulator import Gripper, RobotManipulator

__all__ = ["RobotManipulator", "Gripper", "PyBulletSimulation"]


def __getattr__(name: str):
    # PyBullet is heavy and prints a build banner on import, so the physics
    # simulation module is imported lazily (only when actually needed).
    if name == "PyBulletSimulation":
        from .pybullet_sim import PyBulletSimulation

        return PyBulletSimulation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
