#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys


def compute_obj_bbox(obj_path: str):
    minx = miny = minz = float("inf")
    maxx = maxy = maxz = float("-inf")
    v_count = 0

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 只匹配顶点行：以 "v " 开头（排除 vt, vn）
            if not line.startswith("v "):
                continue

            parts = line.strip().split()
            # parts: ["v", "x", "y", "z", ...]
            if len(parts) < 4:
                continue

            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                continue

            if x < minx: minx = x
            if y < miny: miny = y
            if z < minz: minz = z
            if x > maxx: maxx = x
            if y > maxy: maxy = y
            if z > maxz: maxz = z

            v_count += 1

    if v_count == 0:
        raise ValueError("No valid vertex lines ('v x y z') found in the OBJ file.")

    sizex = maxx - minx
    sizey = maxy - miny
    sizez = maxz - minz
    center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0, (minz + maxz) / 2.0)
    diagonal = math.sqrt(sizex * sizex + sizey * sizey + sizez * sizez)

    return {
        "vertex_count": v_count,
        "min": (minx, miny, minz),
        "max": (maxx, maxy, maxz),
        "size": (sizex, sizey, sizez),
        "center": center,
        "diagonal": diagonal,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute bounding box (AABB) for a Wavefront OBJ file (based on 'v x y z' vertices)."
    )
    parser.add_argument("obj", help="Path to .obj file")
    parser.add_argument(
        "--precision", "-p", type=int, default=6, help="Number of decimal places to print (default: 6)"
    )
    args = parser.parse_args()

    try:
        bbox = compute_obj_bbox(args.obj)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    p = args.precision
    fmt = f"{{:.{p}f}}"

    mn = bbox["min"]
    mx = bbox["max"]
    sz = bbox["size"]
    ct = bbox["center"]

    print(f"OBJ: {args.obj}")
    print(f"Vertices: {bbox['vertex_count']}")
    print(f"bbox min: ({fmt.format(mn[0])}, {fmt.format(mn[1])}, {fmt.format(mn[2])})")
    print(f"bbox max: ({fmt.format(mx[0])}, {fmt.format(mx[1])}, {fmt.format(mx[2])})")
    print(f"size    : ({fmt.format(sz[0])}, {fmt.format(sz[1])}, {fmt.format(sz[2])})")
    print(f"center  : ({fmt.format(ct[0])}, {fmt.format(ct[1])}, {fmt.format(ct[2])})")
    print(f"diagonal: {fmt.format(bbox['diagonal'])}")


if __name__ == "__main__":
    main()

