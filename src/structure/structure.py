import numpy as np
import pybullet as p
import pybullet_data
import os
import sys
from PIL import Image
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from src.utils.transform_utils import quat2mat, angular_error
from src.structure.block import Block

from src.pybullet.place_blocks_in_json import (
    move_until_contact,
    move_out_of_contact,
)
from src.pybullet.pybullet_axes import get_imgs
import time
from typing import List
from data.soft_block_box import manipulator_definition
from data.soft_block_capsule import manipulator_definition2
from somo.sm_continuum_manipulator import SMContinuumManipulator

class Structure:
    def __init__(self, available_blocks={}):
        self.structure: List[Block] = []
        self.available_blocks = available_blocks
        self.drop = True
        self.stability_physics = True
        self.sort_by_height = False

    def get_json(self):
        return [block.get_json() for block in self.structure]

    def get_gpt_json(self):
        return {
            block.gpt_name: {
                "block_type": block.block_name,
                "position": block.get_round_position(),
                "orientation": block.euler_orientation,
            }
            for block in self.structure
        }

    def _get_block_index_by_id(self, id):
        for i, block in enumerate(self.structure):
            if block.id == id:
                return i
        return None

    def get_block_by_id(self, id):
        idx = self._get_block_index_by_id(id)
        if idx is None:
            return -1
        return self.structure[idx]

    def set_block_by_id(self, id, block):
        idx = self._get_block_index_by_id(id)
        if idx is None:
            return -1
        self.structure[idx] = block

        self.place_blocks(drop=self.drop)
        return 0

    def add_block(self, block):
        self.structure.append(block)

    def add_blocks(self, blocks):
        self.structure.extend(blocks)

    def delete_by_id(self, id):
        idx = self._get_block_index_by_id(id)
        self.structure.pop(idx)

    def _place_block(self, block, drop=True, debug=False):
        id = create_block(block)

        if drop:
            # Move the block until it makes contact with another object
            move_until_contact(id, direction=[0, 0, -1], step_size=0.001, debug=debug)
            move_out_of_contact(id)

            position, orientation = p.getBasePositionAndOrientation(id)

            block.position = [pos * 1000 for pos in position]
            block.orientation = orientation
            block.id = id

    def place_blocks(self, drop=True, debug=False):
        p.resetSimulation()
        plane = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        for block in self.structure:
            self._place_block(block)

    def check_stability(
        self,
        id,
        place_all_blocks=False,
        drop=False,
        position_threshold=0.01,
        rotation_threshold=0.1,
        debug=False,
    ):
        # We only want to place blocks up to the specified point
        p.resetSimulation()
        plane = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)
        for block in self.structure:
            self._place_block(block, drop=drop)
            if (not place_all_blocks) and block.id == id:
                break

        pos1, quat1 = p.getBasePositionAndOrientation(id)

        time_step = 1.0 / 240.0  # 240 Hz simulation step
        p.setTimeStep(time_step)
        for _ in range(500):
            p.stepSimulation()
            if debug:
                time.sleep(time_step)

        pos2, quat2 = p.getBasePositionAndOrientation(id)

        pos_delta = np.array(pos2) - np.array(pos1)
        rot_delta = angular_error(quat2mat(np.array(quat2)), quat2mat(np.array(quat1)))

        pos_error = np.linalg.norm(pos_delta)
        rot_error = np.linalg.norm(rot_delta)

        stable = pos_error < position_threshold and rot_error < rotation_threshold

        return stable, pos_delta, rot_delta

    # def check_full_structure_stability(self) -> bool:
    #     """
    #     WARN: It would be more optimal to check the stability of the assembly process rather than of each block... but for our situation this is likely close enough to equivalent.
    #     stability_check in whole_structure is more appropriate
    #     """
    #     return all([
    #         self.check_stability(block.id) for block in self.structure
    #             ])


def create_block(block):
    if block.shape == "cuboid":
        return _create_cuboid(block)
    elif block.shape == "cylinder":
        return _create_cylinder(block)
    elif block.shape == "joint":
        return _create_joint(block)
    else:
        raise ValueError(f"Shape {block.shape} not supported")


def _create_cuboid(block, lateral_friction=0.5, spinning_friction=0.2):
    dimensions, position, orientation, color = (
        block.dimensions,
        block.position,
        block.orientation,
        block.color,
    )
    dimensions_m = [x / 1000 for x in dimensions]
    position_m = [x / 1000 for x in position]
    half_extents = [dim / 2 for dim in dimensions_m]

    density = 1000  # Density of the block in kg/m^3
    mass = density * dimensions_m[0] * dimensions_m[1] * dimensions_m[2]

    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )
    id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position_m,
        baseOrientation=orientation,
    )
    p.changeDynamics(
        id, -1, lateralFriction=lateral_friction, spinningFriction=spinning_friction
    )
    return id


def _create_cylinder(block, lateral_friction=0.5, spinning_friction=0.2):
    radius, height, position, orientation, color = (
        block.dimensions[0],
        block.dimensions[1],
        block.position,
        block.orientation,
        block.color,
    )
    position_m = [x / 1000 for x in position]
    radius_m = radius / 1000
    height_m = height / 1000

    density = 1000  # Density of the block in kg/m^3
    mass = density * np.pi * radius_m**2 * height_m

    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius_m, height=height_m
    )
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius_m, length=height_m, rgbaColor=color
    )
    id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position_m,
        baseOrientation=orientation,
    )
    p.changeDynamics(
        id, -1, lateralFriction=lateral_friction, spinningFriction=spinning_friction
    )
    return id

def _create_joint(block, lateral_friction=0.5, spinning_friction=0.2):
    dimensions, position, orientation, color = (
        block.dimensions,
        block.position,
        block.orientation,
        block.color,
    )
    baseStartPos = [x / 1000 for x in position]
    baseStartOrn = orientation

    
    # Khởi tạo SMContinuumManipulator với manipulator_definition
    joint_block = SMContinuumManipulator(manipulator_definition)
    physics_client = 0
    joint_block.load_to_pybullet(
        baseStartPos=baseStartPos,
        baseStartOrn=baseStartOrn,
        baseConstraint="static",
        physicsClient=physics_client,
    )
    print("joint_block.bodyUniqueId", joint_block.bodyUniqueId)
    p.changeDynamics(joint_block.bodyUniqueId, -1, lateralFriction=lateral_friction, spinningFriction=spinning_friction)
    p.changeDynamics(joint_block.bodyUniqueId, -1, restitution=1)

    return joint_block.bodyUniqueId
    
    
def test_with_gui():
    if not p.isConnected():
        p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    structure = Structure()
    block1 = Block(
        id=1,
        block_name="block1",
        gpt_name="box",
        shape="cuboid",
        dimensions=[180, 100, 240],
        position=[0, 0, 120],
        orientation=[0, 0, 0, 1],
        color=[1.0, 0, 0, 1.0],
    )

    block2 = Block(
        id=2,
        block_name="block2",
        gpt_name="box2",
        shape="cuboid",
        dimensions=[180, 100, 180],
        position=[0, 0, 150],
        orientation=[0, 0, 0, 1],
        color=[1.0, 0, 0, 1.0],
    )
    block3 = Block(
        id=3,
        block_name="block3",
        gpt_name="box3",
        shape="joint",
        dimensions=[100, 100, 100],
        position=[0, 0, 300],
        orientation=[0, 0, 0, 1],
        color=[1.0, 0, 0, 1.0],
    )
    
    structure.add_block(block1)
    structure.place_blocks()
    structure.add_block(block2)
    structure.place_blocks()
    structure.add_block(block3)
    structure.place_blocks()
    print(structure.check_stability(3))

    print("JSON", structure.get_json())
    print("gpt_json", structure.get_gpt_json())
    print("get by id", structure.get_block_by_id(3))
    # time_step = 0.001
    # p.setTimeStep(time_step)
    # n_steps = 20000
    # sim_time = 0.0
    # real_time = time.time()
    # torque_fns = [
    #     lambda t: 0
    # ]  # Only one torque function for the revolute joint
    # for i in range(n_steps):
    #     # Only apply torque to the revolute joint (axis 1)

    #     p.stepSimulation()
    #     sim_time += time_step

    #     # time it took to simulate
    #     delta = time.time() - real_time
    #     real_time = time.time()
    #     # print(delta)

    # structure.place_blocks()

    # structure.get_block_by_id(1).move(
    #     [0, 20, 0], True
    # )
    # structure.place_blocks()
    # structure.get_block_by_id(1).move(
    #     [0, 50, 0], True
    # )
    # structure.place_blocks()

    # structure.delete_by_id(1)
    # structure.place_blocks()
    isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
    img = Image.fromarray(isometric_img)
    img.save(f"{BASE_PATH}/imgs/isometric_img.png")
    p.disconnect()


if __name__ == "__main__":
    test_with_gui()