#!/usr/bin/env python3
"""为 experiments/.../params.npz 生成 preview/orbit_360.png 与 true3d_turntable.gif。"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from viz_scripts.gaussian_viewer import GaussianScene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default="experiments/Custom/session_20260625_104051_lingbot_seed0/params.npz",
    )
    parser.add_argument("--outdir", default=None, help="默认 scene 同级 preview/")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--gif-frames", type=int, default=36)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--distance-scale", type=float, default=1.8)
    args = parser.parse_args()

    scene_path = os.path.abspath(args.scene)
    out_dir = args.outdir or os.path.join(os.path.dirname(scene_path), "preview")
    os.makedirs(out_dir, exist_ok=True)

    print(f"加载: {scene_path}")
    scene = GaussianScene(scene_path, width=args.width, height=args.height)
    print(f"高斯数: {scene.num_gaussians}, radius={scene.radius:.3f}")

    # 12 视角拼图
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("IsoGS lingbot depth - 360 Views", fontsize=16)
    for i, ax in enumerate(axes.flat):
        az = 360.0 * i / 12.0
        img = scene.render(az, args.elevation, args.distance_scale)
        ax.imshow(np.asarray(img))
        ax.set_title(f"View {i + 1}")
        ax.axis("off")
    plt.tight_layout()
    orbit_path = os.path.join(out_dir, "orbit_360.png")
    plt.savefig(orbit_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"写入 {orbit_path}")

    # 转台 GIF
    frames = []
    for i in range(args.gif_frames):
        az = 360.0 * i / args.gif_frames
        frames.append(scene.render(az, args.elevation, args.distance_scale))
    gif_path = os.path.join(out_dir, "true3d_turntable.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )
    print(f"写入 {gif_path}")


if __name__ == "__main__":
    main()
