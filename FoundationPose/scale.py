#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys


def scale_obj(in_path: str, out_path: str, n: float, precision: int = 6):
    """
    Scale all vertex positions (lines starting with 'v ') by factor n,
    then write to out_path. Other lines are copied verbatim.
    """
    fmt = f"{{:.{precision}f}}"

    v_count = 0
    with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            if line.startswith("v "):  # only geometry vertices, not vt/vn
                parts = line.strip().split()
                # Expected: v x y z [w]
                if len(parts) >= 4:
                    try:
                        x = float(parts[1]) * n
                        y = float(parts[2]) * n
                        z = float(parts[3]) * n
                        # Some OBJs may include optional w component
                        if len(parts) >= 5:
                            w = parts[4]
                            fout.write(f"v {fmt.format(x)} {fmt.format(y)} {fmt.format(z)} {w}\n")
                        else:
                            fout.write(f"v {fmt.format(x)} {fmt.format(y)} {fmt.format(z)}\n")
                        v_count += 1
                        continue
                    except ValueError:
                        # Fall through to writing original line if parse fails
                        pass

            fout.write(line)

    return v_count


def main():
    parser = argparse.ArgumentParser(description="Scale a Wavefront OBJ by factor n (vertex positions only).")
    parser.add_argument("input", help="Input .obj path")
    parser.add_argument("output", help="Output .obj path")
    parser.add_argument("n", type=float, help="Scale factor (e.g. 2.0 to double size)")
    parser.add_argument("--precision", "-p", type=int, default=6, help="Decimal places for scaled vertices (default 6)")
    args = parser.parse_args()

    try:
        v_count = scale_obj(args.input, args.output, args.n, args.precision)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Scaled vertices: {v_count}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

