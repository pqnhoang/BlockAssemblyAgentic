import argparse
import os
import math
import shutil
import xml.etree.ElementTree as ET

import numpy as np

try:
    from stl import mesh as numpy_stl_mesh
except Exception as e:  # pragma: no cover
    numpy_stl_mesh = None


def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def _parse_xyz_rpy(elem):
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if elem is not None:
        if elem.get("xyz"):
            xyz = np.array([float(v) for v in elem.get("xyz").split()])
        if elem.get("rpy"):
            rpy = np.array([float(v) for v in elem.get("rpy").split()])
    return xyz, rpy


def _rpy_to_matrix(roll, pitch, yaw):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    r_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    r_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    r_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return r_z @ r_y @ r_x


def _apply_transform(vertices, xyz, rpy):
    rmat = _rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    rotated = vertices @ rmat.T
    translated = rotated + xyz
    return translated


def _compose_T(xyz, rpy):
    rmat = _rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = xyz
    return T


def _apply_T(vertices, T):
    ones = np.ones((vertices.shape[0], 1))
    hom = np.hstack([vertices, ones])
    out = hom @ T.T
    return out[:, :3]


def _save_stl_from_triangles(filename, vertices, faces):
    if numpy_stl_mesh is None:
        raise RuntimeError("numpy-stl is required to export STL files. Install 'numpy-stl'.")
    stl_mesh = numpy_stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=numpy_stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[f[j], :]
    stl_mesh.save(filename)


def _save_stl_from_many(filename, list_of_vertices_faces):
    if numpy_stl_mesh is None:
        raise RuntimeError("numpy-stl is required to export STL files. Install 'numpy-stl'.")
    total_faces = sum(f.shape[0] for _, f in list_of_vertices_faces)
    stl_mesh = numpy_stl_mesh.Mesh(np.zeros(total_faces, dtype=numpy_stl_mesh.Mesh.dtype))
    cursor = 0
    for vertices, faces in list_of_vertices_faces:
        for i, f in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[cursor + i][j] = vertices[f[j], :]
        cursor += faces.shape[0]
    stl_mesh.save(filename)


def _gen_box(size_xyz):
    x, y, z = size_xyz
    # Centered at origin
    vertices = np.array(
        [
            [-x / 2, -y / 2, -z / 2],
            [x / 2, -y / 2, -z / 2],
            [x / 2, y / 2, -z / 2],
            [-x / 2, y / 2, -z / 2],
            [-x / 2, -y / 2, z / 2],
            [x / 2, -y / 2, z / 2],
            [x / 2, y / 2, z / 2],
            [-x / 2, y / 2, z / 2],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # -y
            [1, 5, 6], [1, 6, 2],  # +x
            [2, 6, 7], [2, 7, 3],  # +y
            [3, 7, 4], [3, 4, 0],  # -x
        ]
    )
    return vertices, faces


def _gen_cylinder(radius, length, segments=48):
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    bottom = np.stack([radius * np.cos(angles), radius * np.sin(angles), -np.ones_like(angles) * length / 2], axis=1)
    top = bottom.copy()
    top[:, 2] = length / 2

    vertices = np.vstack([bottom, top])
    faces = []
    # Side faces
    for i in range(segments):
        n = (i + 1) % segments
        faces.append([i, n, segments + i])
        faces.append([n, segments + n, segments + i])
    # Caps (fan)
    center_bottom = vertices.shape[0]
    center_top = vertices.shape[0] + 1
    vertices = np.vstack([vertices, [0, 0, -length / 2], [0, 0, length / 2]])
    for i in range(segments):
        n = (i + 1) % segments
        faces.append([center_bottom, n, i])
        faces.append([center_top, segments + i, segments + n])
    return vertices, np.array(faces, dtype=int)


def _gen_sphere(radius, segments=32, rings=16):
    vertices = []
    faces = []
    for r in range(1, rings):
        phi = math.pi * r / rings
        for s in range(segments):
            theta = 2 * math.pi * s / segments
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            vertices.append([x, y, z])
    vertices.append([0, 0, radius])  # north pole
    vertices.append([0, 0, -radius])  # south pole
    north = len(vertices) - 2
    south = len(vertices) - 1
    # Faces
    for r in range(rings - 2):
        for s in range(segments):
            curr = r * segments + s
            next_s = r * segments + (s + 1) % segments
            above = (r + 1) * segments + s
            above_next = (r + 1) * segments + (s + 1) % segments
            faces.append([curr, next_s, above])
            faces.append([next_s, above_next, above])
    # top cap
    for s in range(segments):
        curr = s
        next_s = (s + 1) % segments
        faces.append([north, curr, next_s])
    # bottom cap
    base = (rings - 2) * segments
    for s in range(segments):
        curr = base + s
        next_s = base + (s + 1) % segments
        faces.append([south, next_s, curr])
    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def _export_primitive_stl(out_path, geom_tag, origin_xyz, origin_rpy):
    if geom_tag.find("box") is not None:
        size = geom_tag.find("box").get("size")
        size_xyz = np.array([float(v) for v in size.split()])
        verts, faces = _gen_box(size_xyz)
    elif geom_tag.find("cylinder") is not None:
        cyl = geom_tag.find("cylinder")
        radius = float(cyl.get("radius"))
        length = float(cyl.get("length"))
        verts, faces = _gen_cylinder(radius, length)
    elif geom_tag.find("sphere") is not None:
        sph = geom_tag.find("sphere")
        radius = float(sph.get("radius"))
        verts, faces = _gen_sphere(radius)
    else:
        return False

    verts = _apply_transform(verts, origin_xyz, origin_rpy)
    _save_stl_from_triangles(out_path, verts, faces)
    return True


def convert_urdf_to_mesh(urdf_path, out_dir, export_format="stl", merge_outfile=None):
    assert export_format.lower() in ("stl",), "Only STL export is supported for primitives."
    _ensure_outdir(out_dir)

    summary = []
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build kinematic graph from joints to compute world transforms
    parent_to_children = {}
    child_to_parent = {}
    joint_T = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        j_origin = joint.find("origin")
        j_xyz, j_rpy = _parse_xyz_rpy(j_origin)
        T_pc = _compose_T(j_xyz, j_rpy)
        parent_to_children.setdefault(parent, []).append(child)
        child_to_parent[child] = parent
        joint_T[(parent, child)] = T_pc

    # Identify base links (parents that are never children)
    all_links = [l.get("name", "") for l in root.findall("link")]
    base_links = [l for l in all_links if l not in child_to_parent]

    # Compute world transform for each link via BFS
    link_world_T = {l: np.eye(4) for l in base_links}
    queue = list(base_links)
    visited = set()
    while queue:
        parent = queue.pop(0)
        visited.add(parent)
        for child in parent_to_children.get(parent, []):
            link_world_T[child] = link_world_T[parent] @ joint_T[(parent, child)]
            if child not in visited:
                queue.append(child)

    merged_tris = []  # list of (vertices, faces) in world frame

    for link in root.findall("link"):
        link_name = link.get("name", "link")
        visual = link.find("visual")
        if visual is None:
            continue
        origin = visual.find("origin")
        xyz, rpy = _parse_xyz_rpy(origin)
        geom = visual.find("geometry")
        if geom is None:
            continue

        # Compose world transform for this link and its visual
        T_world = link_world_T.get(link_name, np.eye(4))
        T_visual = _compose_T(xyz, rpy)
        T_total = T_world @ T_visual

        mesh_tag = geom.find("mesh")
        if mesh_tag is not None and mesh_tag.get("filename"):
            src = mesh_tag.get("filename")
            # Support package:// and relative paths
            if src.startswith("package://"):
                # best-effort: strip scheme, treat remainder as relative
                src = src.replace("package://", "")
            if not os.path.isabs(src):
                src = os.path.join(os.path.dirname(urdf_path), src)
            if os.path.exists(src):
                dst = os.path.join(out_dir, f"{link_name}_visual{os.path.splitext(src)[1]}")
                try:
                    shutil.copy2(src, dst)
                    summary.append({
                        "link": link_name,
                        "type": "mesh",
                        "source": src,
                        "export": dst,
                        "origin_xyz": xyz.tolist(),
                        "origin_rpy": rpy.tolist(),
                    })
                except Exception as e:
                    summary.append({
                        "link": link_name,
                        "type": "mesh",
                        "source": src,
                        "error": str(e),
                    })
            else:
                summary.append({
                    "link": link_name,
                    "type": "mesh",
                    "source": src,
                    "error": "source not found",
                })
            continue

        # Primitive geometry -> export STL
        out_file = os.path.join(out_dir, f"{link_name}_visual.stl")
        # Generate in local, then apply T_total
        if geom.find("box") is not None:
            size = geom.find("box").get("size")
            size_xyz = np.array([float(v) for v in size.split()])
            verts, faces = _gen_box(size_xyz)
            verts = _apply_T(verts, T_total)
            if merge_outfile is None:
                _save_stl_from_triangles(out_file, verts, faces)
            merged_tris.append((verts, faces))
            ok = True
        elif geom.find("cylinder") is not None:
            cyl = geom.find("cylinder")
            radius = float(cyl.get("radius"))
            length = float(cyl.get("length"))
            verts, faces = _gen_cylinder(radius, length)
            verts = _apply_T(verts, T_total)
            if merge_outfile is None:
                _save_stl_from_triangles(out_file, verts, faces)
            merged_tris.append((verts, faces))
            ok = True
        elif geom.find("sphere") is not None:
            sph = geom.find("sphere")
            radius = float(sph.get("radius"))
            verts, faces = _gen_sphere(radius)
            verts = _apply_T(verts, T_total)
            if merge_outfile is None:
                _save_stl_from_triangles(out_file, verts, faces)
            merged_tris.append((verts, faces))
            ok = True
        else:
            ok = False
        if ok:
            summary.append({
                "link": link_name,
                "type": "primitive",
                "export": out_file,
                "origin_xyz": xyz.tolist(),
                "origin_rpy": rpy.tolist(),
                "world_from_link": T_world.tolist(),
            })
        else:
            summary.append({
                "link": link_name,
                "type": "unknown_geometry",
            })

    # Save merged file if requested
    if merge_outfile is not None and len(merged_tris) > 0:
        _ensure_outdir(os.path.dirname(merge_outfile) or ".")
        _save_stl_from_many(merge_outfile, merged_tris)
        summary.append({
            "type": "merged",
            "export": merge_outfile,
            "parts": len(merged_tris),
        })

    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert URDF visual geometries to mesh files.")
    parser.add_argument("urdf", help="Path to URDF file")
    parser.add_argument("--outdir", default="meshes_out", help="Output directory")
    parser.add_argument("--format", default="stl", choices=["stl"], help="Export format for primitives")
    parser.add_argument("--merge", action="store_true", help="Merge all primitives into a single STL file")
    parser.add_argument("--outfile", default=None, help="Filename for merged STL (default: <robot_name>.stl in outdir)")
    parser.add_argument("--summary", action="store_true", help="Print conversion summary")
    args = parser.parse_args()

    # Determine merged output path if requested
    merge_outfile = None
    if args.merge:
        # Parse robot name for default
        try:
            root = ET.parse(args.urdf).getroot()
            robot_name = root.get("name", "robot")
        except Exception:
            robot_name = "robot"
        filename = args.outfile or f"{robot_name}.stl"
        merge_outfile = os.path.join(args.outdir, filename)

    summary = convert_urdf_to_mesh(args.urdf, args.outdir, export_format=args.format, merge_outfile=merge_outfile)
    if args.summary:
        import json
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


