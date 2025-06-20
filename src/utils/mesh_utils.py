import numpy as np
from stl import mesh
import os # 1. Import module os

# 2. Chỉ định thư mục đầu ra của bạn ở đây
output_dir = "stl_files_output" 

# 3. Đảm bảo thư mục đầu ra tồn tại
os.makedirs(output_dir, exist_ok=True)

def create_cuboid_stl(name, dimensions):
    """Tạo một tệp STL cho một khối hình hộp và lưu vào thư mục chỉ định."""
    x, y, z = dimensions['x'], dimensions['y'], dimensions['z']
    
    vertices = np.array([
        [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
        [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])

    faces = np.array([
        [0, 3, 1], [1, 3, 2], [4, 5, 7], [5, 6, 7],
        [0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5],
        [2, 3, 6], [3, 7, 6], [0, 4, 3], [3, 4, 7]])

    cuboid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cuboid.vectors[i][j] = vertices[f[j],:]

    # 4. Kết hợp đường dẫn thư mục và tên tệp
    filename = f"{name}.stl"
    full_path = os.path.join(output_dir, filename)
    cuboid.save(full_path)
    print(f"Đã tạo: {full_path}")

def create_cylinder_stl(name, dimensions, segments=32):
    """Tạo một tệp STL cho một khối hình trụ và lưu vào thư mục chỉ định."""
    radius, height = dimensions['radius'], dimensions['height']
    
    bottom_center = np.array([0, 0, 0])
    top_center = np.array([0, 0, height])
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    bottom_vertices = np.array([[radius * np.cos(a), radius * np.sin(a), 0] for a in angles])
    top_vertices = np.array([[radius * np.cos(a), radius * np.sin(a), height] for a in angles])
    vertices = np.vstack([bottom_center, top_center, bottom_vertices, top_vertices])
    
    faces = []
    for i in range(segments):
        faces.append([0, 2 + i, 2 + (i + 1) % segments])
        faces.append([1, 2 + segments + (i + 1) % segments, 2 + segments + i])
    for i in range(segments):
        p1, p2 = 2 + i, 2 + (i + 1) % segments
        p3, p4 = p1 + segments, p2 + segments
        faces.append([p1, p2, p3])
        faces.append([p2, p4, p3])
        
    faces = np.array(faces)
    
    cylinder = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cylinder.vectors[i][j] = vertices[f[j],:]
            
    # Kết hợp đường dẫn thư mục và tên tệp
    filename = f"{name}.stl"
    full_path = os.path.join(output_dir, filename)
    cylinder.save(full_path)
    print(f"Đã tạo: {full_path}")

# --- Dữ liệu và vòng lặp chính (không thay đổi) ---
if __name__ == "__main__":
    blocks_data = {
    "small-25x25x25-cuboid": {
      "dimensions": {"x": 25, "y": 25, "z": 25},
      "shape": "cuboid",
      "number_available": 4
    },
    "small-25x25x50-cuboid": {
      "dimensions": {"x": 25, "y": 25, "z": 50},
      "shape": "cuboid",
      "number_available": 5
    },
    "large-100x25x25-cuboid": {
      "dimensions": {"x": 100, "y": 25, "z": 25},
      "shape": "cuboid",
      "number_available": 4
    },
    "medium-40x40x40-cylinder": {
      "dimensions": {"radius": 20, "height": 40},
      "shape": "cylinder",
      "number_available": 4
    },
    "large-90x70x20-cuboid": {
      "dimensions": {"x": 90, "y": 70, "z": 20},
      "shape": "cuboid",
      "number_available": 2
    },
    "large-80x80x10-cylinder": {
      "dimensions": {"radius": 40, "height": 10},
      "shape": "cylinder",
      "number_available": 2
    }
  }

    for name, data in blocks_data.items():
        if data["shape"] == "cuboid":
            create_cuboid_stl(name, data["dimensions"])
        elif data["shape"] == "cylinder":
            create_cylinder_stl(name, data["dimensions"])