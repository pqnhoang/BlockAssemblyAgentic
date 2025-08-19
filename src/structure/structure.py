import numpy as np
import pybullet as p
import pybullet_data
import os
import sys
from PIL import Image

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(BASE_PATH)

from src.utils.transform_utils import quat2mat, angular_error
from src.structure.block import Block

from src.pybullet_utils.place_blocks_in_json import (
    move_until_contact,
    move_out_of_contact,
)
from src.pybullet_utils.pybullet_axes import get_imgs
import time
from typing import List
from configs import BlockMASSettings
from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.create_cmassembly_urdf import create_cmassembly_urdf
from somo.sm_link_definition import SMLinkDefinition
from somo.sm_joint_definition import SMJointDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator

settings = BlockMASSettings()
class Structure:
    def __init__(self, object_name, available_blocks={}):
        self.structure: List[Block] = []
        self.available_blocks = available_blocks
        self.drop = True
        self.stability_physics = True
        self.sort_by_height = False
        self.urdf_name = object_name
        
        self.base_block_groups = {}
        self.manipulator_groups = {}

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

    def find_block_by_name(self, name, search_by="gpt_name"):
        """
        Find block by name
        Args:
            name: Name of the block to find
            search_by: Attribute to search by ("gpt_name" or "block_name")
        Returns:
            Block object if found, None if not found
        """
        for block in self.structure:
            if search_by == "gpt_name" and block.gpt_name == name:
                return block
            elif search_by == "block_name" and block.block_name == name:
                return block
        return None

    def add_block(self, block):
        self.structure.append(block)

    def add_blocks(self, blocks):
        self.structure.extend(blocks)

    def delete_by_id(self, id):
        idx = self._get_block_index_by_id(id)
        self.structure.pop(idx)

    def _place_block(self, block, drop=True, debug=False):
        if block.shape == "joint":
            base_block = self.find_block_by_name(block.base_block, search_by="gpt_name")
            if base_block:
                # Thêm joint block vào nhóm của base block
                base_name = block.base_block
                if base_name not in self.base_block_groups:
                    self.base_block_groups[base_name] = []
                    self.manipulator_groups[base_name] = []
                
                self.base_block_groups[base_name].append(block)
                
                # Tạo manipulator definition và thêm vào nhóm
                manipulator_definition = SMManipulatorDefinition.from_file(settings.path.joint_def_path)
                offset = self._calculate_joint_offset(block)
                self.manipulator_groups[base_name].append((manipulator_definition, offset))
                
                # Tạm thời không tạo URDF, chỉ return None
                block.id = None
                return None
            else:
                # Không tìm thấy base block, tạo joint đơn lẻ
                id = _create_joint(block)
                block.id = id
                return id
        else:
            # Block thường - tạo bình thường
            id = create_block(block)
            if id is not None and id >= 0:
                pos, ori = p.getBasePositionAndOrientation(id)
                print(f"📍 Block {block.gpt_name} (ID: {id}) placed at: {pos}")
                
                # Kiểm tra xem block có visible không
                visual_data = p.getVisualShapeData(id)
                print(f"   - Visual shape data: {visual_data}")
                
                if drop:
                    move_until_contact(id, direction=[0, 0, -1], step_size=0.001, debug=debug)
                    move_out_of_contact(id)
                    
                    # Kiểm tra vị trí sau khi drop
                    pos_after, ori_after = p.getBasePositionAndOrientation(id)
                    print(f"   - Position after drop: {pos_after}")
            
            return id            
        
    def place_blocks(self, drop=True, debug=False):
        p.resetSimulation()
        plane = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)
        
        self.base_block_groups = {}
        self.manipulator_groups = {}

        count = 0
        for block in self.structure:
            self._place_block(block)
            print(f"Placed block {count}: {block.gpt_name} (ID: {block.id})")
            count += 1
        
        self._create_joint_assemblies()
    
    def _calculate_joint_offset(self, block):
        """
        Calculate the offset for the joint block
        """
        position, orientation = block.position, block.orientation
        position_m = [x / 1000 for x in position]
        x, y, z = position_m
        roll, pitch, yaw = np.radians(orientation[0]), np.radians(orientation[1]), np.radians(orientation[2])
        return [x, y, z, roll, pitch, yaw]

    def check_stability(
        self,
        id,
        place_all_blocks=False,
        drop=False,
        position_threshold=0.01,
        rotation_threshold=0.1,
        debug=False,
    ):
        # Kiểm tra ID có hợp lệ không
        if id is None or id < 0:
            print(f"Invalid block ID: {id} - skipping stability check")
            return False, [0, 0, 0], [0, 0, 0]
        
        # Tìm block với ID này
        target_block = None
        for block in self.structure:
            if block.id == id:
                target_block = block
                break
        
        ###TODO: Check stability of joint blocks
        if target_block and target_block.shape == "joint":
            print(f"Block {target_block.gpt_name} is a joint block - skipping stability check")
            return True, [0, 0, 0], [0, 0, 0]  # Giả sử joint blocks ổn định
        
        # We only want to place blocks up to the specified point
        p.resetSimulation()
        plane = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)
        
        for block in self.structure:
            self._place_block(block, drop=drop)
            if (not place_all_blocks) and block.id == id:
                break

        # Check if block exists in simulation
        try:
            pos1, quat1 = p.getBasePositionAndOrientation(id)
        except p.error as e:
            print(f"Error getting position for block ID {id}: {e}")
            print(f"Available body IDs: {[block.id for block in self.structure if block.id is not None]}")
            return False, [0, 0, 0], [0, 0, 0]

        time_step = 1.0 / 240.0  # 240 Hz simulation step
        p.setTimeStep(time_step)
        for _ in range(500):
            p.stepSimulation()
            if debug:
                time.sleep(time_step)

        # Check again before getting final position
        try:
            pos2, quat2 = p.getBasePositionAndOrientation(id)
        except p.error as e:
            print(f"Error getting final position for block ID {id}: {e}")
            return False, [0, 0, 0], [0, 0, 0]

        pos_delta = np.array(pos2) - np.array(pos1)
        rot_delta = angular_error(quat2mat(np.array(quat2)), quat2mat(np.array(quat1)))

        pos_error = np.linalg.norm(pos_delta)
        rot_error = np.linalg.norm(rot_delta)

        stable = pos_error < position_threshold and rot_error < rotation_threshold

        return stable, pos_delta, rot_delta

    def _create_joint_assemblies(self):    
        for base_name, joint_blocks in self.base_block_groups.items():
            if not joint_blocks:
                continue
                
            # Lấy base block object
            base_block = self.find_block_by_name(base_name, search_by="gpt_name")
            if not base_block:
                print(f"Base block '{base_name}' not found for joint assembly!")
                continue
            
            # Lấy manipulator pairs cho base block này
            manipulator_pairs = self.manipulator_groups[base_name]
            
            try:
                # Tạo assembly URDF cho base block này
                test_urdf = create_cmassembly_urdf(
                    base_links=[base_block.smlink_definition],
                    manipulator_definition_pairs=manipulator_pairs,
                    assembly_name=f"{self.urdf_name}_{base_name}",
                )
                
                # Load URDF vào PyBullet
                assembly_body_id = p.loadURDF(test_urdf)
                
                # Cập nhật body_id cho tất cả joint blocks thuộc base block này
                for joint_block in joint_blocks:
                    joint_block.id = assembly_body_id
                
                print(f"Created joint assembly for base '{base_name}' with {len(manipulator_pairs)} joints")
                
            except Exception as e:
                print(f"Error creating joint assembly for base '{base_name}': {e}")

def create_block(block):
    if block.shape == "cuboid":
        return _create_cuboid(block)
    elif block.shape == "cylinder":
        return _create_cylinder(block)
    # elif block.shape == "joint":
    #     return _create_joint(block)
    else:
        raise ValueError(f"Shape {block.shape} not supported")


def _create_cuboid(block):
    
    dimensions, position, orientation, color = (
        block.dimensions,
        block.position,
        block.orientation,
        block.color
    )
    dimensions_m = [x / 1000 for x in dimensions]
    position_m = [x / 1000 for x in position]
    roll, pitch, yaw = np.radians(orientation[0]), np.radians(orientation[1]), np.radians(orientation[2])
    offset = [position_m[0], position_m[1], position_m[2], roll, pitch, yaw]
    density = 1000
    mass = density * dimensions[0] * dimensions[1] * dimensions[2]
    
    cuboid_link = SMLinkDefinition(
    shape_type="box",
    dimensions=dimensions_m,
    mass=mass,
    material_color=color,
    inertial_values=[1, 0, 0, 1, 0, 1],
    material_name="base_color",
    origin_offset=offset,
)   
    block.smlink_definition = cuboid_link
    
    id = cuboid_link.load_to_pybullet(physicsClient=-1)
    return id


def _create_cylinder(block):
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
    x, y, z = position_m
    roll, pitch, yaw = np.radians(orientation[0]), np.radians(orientation[1]), np.radians(orientation[2])
    offset = [x, y, z, roll, pitch, yaw]
    density = 1000  # Density of the block in kg/m^3
    mass = density * np.pi * radius_m**2 * height_m
    cylinder_link = SMLinkDefinition(
        shape_type="cylinder",
        dimensions=[radius_m, height_m],
        mass=mass,
        material_color=color,
        inertial_values=[1, 0, 0, 1, 0, 1],
        material_name="base_color",
        origin_offset=offset,
    )
    block.smlink_definition = cylinder_link
    id = cylinder_link.load_to_pybullet(physicsClient=-1)
    
    return id


def _create_joint(block):
        position, orientation, color = (
            block.position,
            block.orientation,
            block.color
        )
        position_m = [x / 1000 for x in position]
        roll, pitch, yaw = np.radians(orientation[0]), np.radians(orientation[1]), np.radians(orientation[2])
        
        manipulator_definition = SMManipulatorDefinition.from_file(settings.path.joint_def_path)
        joint_block = SMContinuumManipulator(manipulator_definition)
        pos = [position_m[0], position_m[1], position_m[2]] 
        startOr = p.getQuaternionFromEuler([roll, pitch, yaw])
        joint_block.load_to_pybullet(
            baseStartPos=pos,
            baseStartOrn=startOr,
            baseConstraint="static",  # other options are free and constrained, but those are not recommended rn
            physicsClient=0,
        )
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