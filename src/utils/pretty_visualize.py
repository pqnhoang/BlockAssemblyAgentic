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
import cairosvg  # <-- THÊM IMPORT NÀY

# Thêm đường dẫn gốc của dự án
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
from src.structure.assembly import Assembly


def save_visual(blocks, save_path_str):
    """
    Tạo và lưu cả file SVG và PNG cho một danh sách các khối.
    """
    # Chuyển đường dẫn string thành đối tượng Path để dễ dàng thao tác
    save_path = Path(save_path_str)

    # Initialize a PyVista plotter
    plotter = pv.Plotter(off_screen=True) # Dùng off_screen để không hiển thị cửa sổ đồ họa

    for i, block in enumerate(blocks):
        dim = block["dimensions"]
        if block["shape"] == "cuboid":
            x, y, z = dim["x"], dim["y"], dim["z"]
            mesh = pv.Box(bounds=[-x/2, x/2, -y/2, y/2, -z/2, z/2])
        elif block["shape"] == "cylinder":
            radius, height = dim["radius"], dim["height"]
            mesh = pv.Cylinder(radius=radius, height=height, direction=(0, 0, 1))
        elif block["shape"] == "cone":
            radius, height = dim["radius"], dim["height"]
            mesh = pv.Cone(radius=radius, height=height, direction=(0, 0, 1), center=(0, 0, height/2), resolution=100)
        else:
            print(f"Warning: Block type {block.get('type')} not supported. Skipping.")
            continue

        # Áp dụng vị trí và hướng xoay
        mesh = mesh.translate(block["position"])
        r = R.from_quat(block["orientation"])
        rotvec = r.as_rotvec()
        angle = np.linalg.norm(rotvec)
        mesh = mesh.rotate_vector(rotvec, np.degrees(angle), point=block["position"])
        mesh = mesh.triangulate()

        # Áp dụng màu sắc
        color = block["color"][:3]
        plotter.add_mesh(mesh, color=color)

    plotter.set_background("white")
    plotter.show_axes()
    plotter.camera_position = 'iso' # Đặt góc nhìn isometric

    # --- PHẦN LƯU FILE ĐÃ SỬA ĐỔI ---
    # 1. Lưu file SVG
    svg_path = save_path.with_suffix(".svg")
    plotter.save_graphic(str(svg_path))
    print(f"Successfully saved SVG: {svg_path}")

    # 2. Chuyển đổi và lưu file PNG
    png_path = save_path.with_suffix(".png")
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
        print(f"Successfully saved PNG: {png_path}")
    except Exception as e:
        print(f"Could not convert SVG to PNG. Error: {e}")
    
    # Đóng plotter để giải phóng bộ nhớ
    plotter.close()


def save_assembly_visual(assembly, save_path):
    """
    Hàm tiện ích để lấy dữ liệu JSON từ đối tượng Assembly và gọi hàm save_visual.
    """
    # Giả định assembly.structure là đối tượng Structure có phương thức get_json()
    if hasattr(assembly, 'structure') and hasattr(assembly.structure, 'get_json'):
         block_data = assembly.structure.get_json()
         save_visual(block_data, save_path)
    else:
        print(f"Error: Could not find valid structure data in the assembly from path.")


if __name__ == "__main__":
    # Tìm tất cả các file assembly đã lưu
    paths = glob.glob("/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/gpt_caching/*/*.pkl")
    
    if not paths:
        print("No .pkl assembly files found in the specified path.")

    for path in paths:
        print("-" * 20)
        print(f"Processing file: {path}")
        try:
            assembly = Assembly.load(path)
            
            # Tạo đường dẫn lưu file output, bỏ phần đuôi .pkl
            output_path_base = os.path.join(
                os.path.dirname(path), f"pretty_viz_{assembly.assembly_num}"
            )
            
            save_assembly_visual(assembly, output_path_base)
        except Exception as e:
            print(f"Failed to process {path}. Error: {e}")