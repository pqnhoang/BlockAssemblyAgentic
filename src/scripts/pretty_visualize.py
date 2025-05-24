"""
For making better looking visualizations in PyVista (rather than the default pybullet visualization)
"""
import sys
import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import os
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
from src.structure.assembly import Assembly


def save_visual(blocks, save_path):
    # Initialize a PyVista plotter
    plotter = pv.Plotter()

    for i, block in enumerate(blocks):
        dim = block["dimensions"]
        if block["shape"] == "cuboid":
            x, y, z = dim["x"], dim["y"], dim["z"]
            mesh = pv.Box(
                [
                    -x / 2,
                    x / 2,
                    -y / 2,
                    y / 2,
                    -z / 2,
                    z / 2,
                ]
            )
        elif block["shape"] == "cylinder":
            radius, height = dim["radius"], dim["height"]
            mesh = pv.Cylinder(
                radius=radius,
                height=height,
                direction=(0, 0, 1),
            )
        elif block["shape"] == "cone":
            radius, height = dim["radius"], dim["height"]
            mesh = pv.Cone(
                radius=radius,
                height=height,
                direction=(0, 0, 1),
                center=(0, 0, height / 2),
                resolution=100,
            )
        else:
            raise ValueError(f"Block type {block['type']} not supported")

        mesh = mesh.translate(block["position"])

        # Convert quaternion to axis-angle
        r = R.from_quat(block["orientation"])
        rotvec = r.as_rotvec()
        angle = np.linalg.norm(rotvec)
        mesh = mesh.rotate_vector(rotvec, np.degrees(angle), point=block["position"])
        mesh = mesh.triangulate()

        # Choose a color from the predefined set
        color = block["color"][:3]

        plotter.add_mesh(mesh, color=color)

    plotter.set_background("white")
    plotter.show_axes()

    # Get directory path
    save_svg_path = Path(save_path).with_suffix(".svg")
    # save_png_path = os.path.join(dir_path, f"{filename}_viz.png")

    # Save the plot as SVG
    plotter.save_graphic(save_svg_path)

    # Convert SVG to PNG
    # cairosvg.svg2png(url=save_svg_path, write_to=save_png_path)

    print(f"Visualization saved as SVG: {save_svg_path}")
    # print(f"Visualization saved as PNG: {save_png_path}")

    # delete the svg file
    # os.remove(save_svg_path)


def save_assembly_visual(assembly, save_path):
    save_visual(assembly.structure.get_json(), save_path)


paths = glob.glob("/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/gpt_caching/*/*.pkl")
for path in paths:
    assembly = Assembly.load(path)
    save_path = os.path.join(
        os.path.dirname(path), f"pretty_viz_{assembly.assembly_num}.svg"
    )
    save_assembly_visual(assembly, save_path)