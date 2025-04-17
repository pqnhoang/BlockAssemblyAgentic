import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation
    

def create_cuboid(structure, name, dimension, position, yaw, color=[1, 0, 0, 1]):
    cuboid = gs.morphs.Box(
        size = dimension,
        pos = position,
        rot = yaw
    )
    structure.scene.add_entity(cuboid, material=gs.materials.Rigid(friction=1), surface=gs.surfaces.Default(color=color))
    return cuboid

def create_cylinder(structure, name, dimension, position, yaw, color=[1, 0, 0, 1]):
    cylinder = gs.morphs.Cylinder(
        radius = dimension[0],
        height = dimension[1],
        pos = position,
        rot = yaw
    )
    structure.scene.add_entity(cylinder, material=gs.materials.Rigid(friction=1), surface=gs.surfaces.Default(color=color))
    return cylinder

def create_soft_cuboid(structure, name, dimension, position, yaw, color=[1, 0, 0, 1]):
    pass

def create_soft_cylinder(structure, name, dimension, position, yaw, color=[1, 0, 0, 1]):
    pass

def check_collision(structure, entity1, entity2):
    pass
def set_euler_rotation(entity, euler_angles, sequence='xyz', degrees=True, zero_velocity=True, envs_idx=None):
    """
    Set rotation of a rigid entity using Euler angles.
    
    Parameters:
    -----------
    entity : RigidEntity
        The rigid entity to rotate
    euler_angles : list or numpy.ndarray
        The [roll, pitch, yaw] angles for rotation
    sequence : str, optional
        The sequence of rotation axes, e.g., 'xyz', 'zyx', 'xzy', etc.
        Default is 'xyz' (roll around x, then pitch around y, then yaw around z)
    degrees : bool, optional
        Whether the angles are specified in degrees (True) or radians (False)
        Default is True
    zero_velocity : bool, optional
        Whether to zero velocities after setting rotation
        Default is True
    envs_idx : None or array_like, optional
        The indices of environments for parallel simulations
        If None, applies to all environments
        Default is None
        
    Returns:
    --------
    None
    
    Example:
    --------
    # Rotate 90 degrees around z-axis
    set_euler_rotation(robot, [0, 0, 90])
    
    # Rotate 45 degrees around x, then 30 around y using radians
    set_euler_rotation(object, [np.pi/4, np.pi/6, 0], degrees=False)
    """
    # Convert degrees to radians if needed
    if degrees:
        euler_angles = [np.radians(angle) for angle in euler_angles]
    
    # Create rotation object and convert to quaternion
    r = Rotation.from_euler(sequence, euler_angles)
    quat = r.as_quat()  # Returns in [x, y, z, w] format
    
    # Set the quaternion on the entity
    entity.set_quat(quat, zero_velocity=zero_velocity, envs_idx=envs_idx)