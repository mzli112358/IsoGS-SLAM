"""
True 3D Gaussian Splatting viewer (CUDA rasterization, no WebGL / no Open3D GUI).
Usage:
  python viz_scripts/gaussian_viewer.py configs/tum/splatam_test.py
Controls:
  Mouse drag left/right : orbit
  Mouse drag up/down    : elevation
  Scroll wheel          : zoom
  Arrow keys            : orbit
  Q / Esc               : quit
"""
import argparse
import os
import sys
import math

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageTk
from importlib.machinery import SourceFileLoader

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera


def look_at_w2c(eye, target, up=np.array([0.0, -1.0, 0.0])):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-8
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-8
    u = np.cross(r, f)
    c2w = np.eye(4)
    c2w[:3, 0] = r
    c2w[:3, 1] = -u
    c2w[:3, 2] = f
    c2w[:3, 3] = eye
    return np.linalg.inv(c2w)


class GaussianScene:
    def __init__(self, scene_path, width=960, height=720):
        self.width = width
        self.height = height
        all_params = dict(np.load(scene_path, allow_pickle=True))
        params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}

        if params["log_scales"].shape[-1] == 1:
            log_scales = torch.tile(params["log_scales"], (1, 3))
        else:
            log_scales = params["log_scales"]

        self.rendervar = {
            "means3D": params["means3D"],
            "colors_precomp": params["rgb_colors"],
            "rotations": F.normalize(params["unnorm_rotations"]),
            "opacities": torch.sigmoid(params["logit_opacities"]),
            "scales": torch.exp(log_scales),
            "means2D": torch.zeros_like(params["means3D"], device="cuda"),
        }

        intrinsics = params["intrinsics"].cpu().numpy()
        self.k = intrinsics[:3, :3].copy()
        self.k[0, :] *= width / params["org_width"].item()
        self.k[1, :] *= height / params["org_height"].item()

        self.center = params["means3D"].mean(dim=0).cpu().numpy()
        # torch.quantile 对超大张量会报 "input tensor is too large"；抽样估计半径
        dists = torch.norm(params["means3D"] - params["means3D"].mean(0), dim=1)
        n = dists.numel()
        if n > 1_000_000:
            idx = torch.randperm(n, device=dists.device)[:1_000_000]
            radius_t = dists[idx].quantile(0.9)
        else:
            radius_t = dists.quantile(0.9)
        self.radius = float(radius_t.cpu())
        self.near = 0.01
        self.far = 100.0
        self.num_gaussians = params["means3D"].shape[0]

    def render(self, azimuth_deg, elevation_deg, distance_scale=1.8):
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        dist = self.radius * distance_scale
        eye = self.center + dist * np.array([
            math.cos(el) * math.sin(az),
            -math.sin(el),
            math.cos(el) * math.cos(az),
        ])
        w2c = look_at_w2c(eye, self.center)

        with torch.no_grad():
            cam = setup_camera(self.width, self.height, self.k, w2c, self.near, self.far)
            white_bg = Camera(
                image_height=cam.image_height,
                image_width=cam.image_width,
                tanfovx=cam.tanfovx,
                tanfovy=cam.tanfovy,
                bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
                scale_modifier=cam.scale_modifier,
                viewmatrix=cam.viewmatrix,
                projmatrix=cam.projmatrix,
                sh_degree=cam.sh_degree,
                campos=cam.campos,
                prefiltered=cam.prefiltered,
            )
            im, _, _ = Renderer(raster_settings=white_bg)(**self.rendervar)
            rgb = im.detach().cpu().permute(1, 2, 0).numpy()
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)


class ViewerApp:
    def __init__(self, scene, title="SplaTAM True 3D Viewer"):
        self.scene = scene
        self.azimuth = 30.0
        self.elevation = 15.0
        self.distance = 1.8
        self.drag_start = None
        self.rendering = False
        self.pending = False

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{scene.width}x{scene.height + 48}")
        self.root.configure(bg="#222")

        info = (
            f"True Gaussian Splatting | {scene.num_gaussians:,} gaussians | "
            "Drag=rotate  Scroll=zoom  Q=quit"
        )
        self.label = tk.Label(self.root, text=info, fg="#ccc", bg="#222", anchor="w")
        self.label.pack(fill="x", padx=8, pady=4)

        self.canvas = tk.Canvas(self.root, width=scene.width, height=scene.height, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.photo = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel_linux)
        self.canvas.bind("<Button-5>", self.on_wheel_linux)
        self.root.bind("<Left>", lambda e: self.nudge(-5, 0))
        self.root.bind("<Right>", lambda e: self.nudge(5, 0))
        self.root.bind("<Up>", lambda e: self.nudge(0, 3))
        self.root.bind("<Down>", lambda e: self.nudge(0, -3))
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.refresh()

    def nudge(self, d_az, d_el):
        self.azimuth += d_az
        self.elevation = float(np.clip(self.elevation + d_el, -80, 80))
        self.schedule_refresh()

    def on_press(self, event):
        self.drag_start = (event.x, event.y)

    def on_drag(self, event):
        if self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.drag_start = (event.x, event.y)
        self.azimuth += dx * 0.4
        self.elevation = float(np.clip(self.elevation + dy * 0.3, -80, 80))
        self.schedule_refresh()

    def on_wheel(self, event):
        self.distance = float(np.clip(self.distance * (0.9 if event.delta > 0 else 1.1), 0.5, 5.0))
        self.schedule_refresh()

    def on_wheel_linux(self, event):
        self.distance = float(np.clip(self.distance * (0.9 if event.num == 4 else 1.1), 0.5, 5.0))
        self.schedule_refresh()

    def schedule_refresh(self):
        if not self.rendering:
            self.pending = True
            self.root.after(10, self.refresh)

    def refresh(self):
        if self.rendering:
            self.pending = True
            return
        self.rendering = True
        self.pending = False
        self.label.config(
            text=(
                f"Rendering... az={self.azimuth:.0f}° el={self.elevation:.0f}° "
                f"dist={self.distance:.2f} | {self.scene.num_gaussians:,} gaussians"
            )
        )
        self.root.update_idletasks()

        img = self.scene.render(self.azimuth, self.elevation, self.distance)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.rendering = False
        self.label.config(
            text=(
                f"True Gaussian Splatting | az={self.azimuth:.0f}° el={self.elevation:.0f}° | "
                f"{self.scene.num_gaussians:,} gaussians | Drag=rotate Scroll=zoom Q=quit"
            )
        )
        if self.pending:
            self.root.after(10, self.refresh)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment config")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    seed_everything(seed=experiment.config["seed"])

    if "scene_path" in experiment.config:
        scene_path = experiment.config["scene_path"]
    else:
        scene_path = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"], "params.npz"
        )

    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    title = f"SplaTAM 3D | {experiment.config.get('run_name', 'reconstruction')}"
    scene = GaussianScene(scene_path, width=args.width, height=args.height)
    ViewerApp(scene, title=title).run()


if __name__ == "__main__":
    main()
