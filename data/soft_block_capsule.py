import numpy as np

# base_definition

# todo: minimize the content of this;
# todo have definitions saved in one file instead of in separate files

base_definition = None

joint_definition1 = {
    "joint_type": "revolute",
    "axis": [1, 0, 0],
    "limits": [-3.141592/10, 3.141592/10, 100, 3],
    "spring_stiffness": 500,
    "joint_neutral_position": 0,
    "neutral_axis_offset": [0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
    "joint_control_limit_force": 1.0,
}

joint_definition2 = {
    "joint_type": "revolute",
    "axis": [0, 1, 0],
    "limits": [-3.141592/10, 3.141592/10, 100, 3],
    "spring_stiffness": 500,
    "joint_neutral_position": 0,
    "joint_control_limit_force": 1.0,
}


link_definition = {
    "shape_type": "capsule",
    "dimensions": [0.025, 0.025],
    "mass": 0.350,
    "inertial_values": [1, 0, 0, 1, 0, 1],
    "material_color": [0.6, 0.0, 0.8, 1.0],
    "material_name": "green",
}
tip_definition = None

actuator_definition = {
    "actuator_length": 0.2,
    "n_segments": 8,
    "link_definition": link_definition,
    "joint_definitions": [joint_definition1, joint_definition2],
    "planar_flag": 0,
}

manipulator_definition2 = {
    "n_act": 1,
    "base_definition": base_definition,
    "actuator_definitions": [actuator_definition],
    "tip_definition": tip_definition,
    "manipulator_name": "neck",
    "urdf_filename": "neck.urdf",
}