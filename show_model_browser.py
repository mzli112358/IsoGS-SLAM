#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IsoGS Model Browser - 独立的模型浏览器
整合了 show_model_browser.py, online_recon.py 和 final_recon.py 的功能
"""

import os
import sys
import re
import time
import argparse
import subprocess
import multiprocessing as mp
from datetime import datetime
from copy import deepcopy
from importlib.machinery import SourceFileLoader

import tkinter as tk
from tkinter import ttk, messagebox

# ==================== 环境检查和依赖导入 ====================

def check_and_install_dependencies():
    """检查并提示安装缺少的依赖库"""
    missing_packages = []
    
    # 检查基础库
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing_packages.append("matplotlib")
    
    try:
        import open3d
    except ImportError:
        missing_packages.append("open3d")
    
    try:
        import torch
    except ImportError:
        missing_packages.append("torch")
    
    try:
        from diff_gaussian_rasterization import GaussianRasterizer
        from diff_gaussian_rasterization import GaussianRasterizationSettings
    except ImportError:
        missing_packages.append("diff-gaussian-rasterization")
    
    if missing_packages:
        print("=" * 60)
        print("缺少以下依赖库，请使用以下命令安装：")
        print("=" * 60)
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        print("=" * 60)
        print("注意：torch 和 diff-gaussian-rasterization 可能需要特定的安装方式")
        print("     请参考项目文档或使用 conda 环境")
        print("=" * 60)
        return False
    return True

# 检查依赖
_HAS_VISUALIZATION_DEPS = check_and_install_dependencies()

# 导入可视化相关的库
VISUALIZATION_AVAILABLE = False
if _HAS_VISUALIZATION_DEPS:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import open3d as o3d
        import torch
        import torch.nn.functional as F
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
        VISUALIZATION_AVAILABLE = True
    except ImportError as e:
        VISUALIZATION_AVAILABLE = False
        print(f"警告: 可视化库导入失败 ({e})，.npz 文件可视化功能不可用")

# ==================== 内联工具函数 ====================

def seed_everything(seed=42):
    """
    设置随机种子，确保结果可复现
    """
    if not VISUALIZATION_AVAILABLE:
        return
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")


def build_rotation(q):
    """
    从四元数构建旋转矩阵
    q: [N, 4] 四元数，格式为 [w, x, y, z]
    返回: [N, 3, 3] 旋转矩阵
    """
    if not VISUALIZATION_AVAILABLE:
        return None
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    """
    设置相机参数
    w, h: 图像宽度和高度
    k: 内参矩阵 [3, 3]
    w2c: 世界到相机的变换矩阵 [4, 4]
    near, far: 近远平面
    返回: Camera 对象
    """
    if not VISUALIZATION_AVAILABLE:
        return None
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam


def get_depth_and_silhouette(pts_3D, w2c):
    """
    计算每个高斯点的深度和轮廓
    pts_3D: [num_gaussians, 3] 3D点
    w2c: [4, 4] 世界到相机的变换矩阵
    返回: [num_gaussians, 3] [depth, silhouette, depth_sq]
    """
    if not VISUALIZATION_AVAILABLE:
        return None
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1)  # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z)  # [num_gaussians, 1]
    
    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    
    return depth_silhouette


# ==================== Model Browser 部分 ====================

def get_scan_directory():
    """
    返回当前文件所在目录的绝对路径。
    该目录及其所有子目录将被扫描以查找模型文件。
    """
    scan_dir = os.path.dirname(os.path.abspath(__file__))
    return scan_dir


def view_model(path):
    """
    独立进程中运行的模型查看函数。
    使用 Trimesh 打开指定路径的 mesh 文件。
    """
    try:
        import trimesh
    except ImportError as e:
        print("[Model Viewer] Failed to import trimesh:", e, file=sys.stderr)
        return

    if not os.path.isfile(path):
        print(f"[Model Viewer] File not found: {path}", file=sys.stderr)
        return

    # 加载网格
    try:
        mesh = trimesh.load(path, force="mesh")
    except BaseException as e:
        print(f"[Model Viewer] Failed to load mesh: {path} ({e})", file=sys.stderr)
        return

    if mesh is None or getattr(mesh, "is_empty", False):
        print(f"[Model Viewer] Mesh is empty or invalid: {path}", file=sys.stderr)
        return

    # 简单清理，避免奇怪的重复顶点/面
    try:
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicate_faces()
    except BaseException:
        pass

    # Trimesh 的 show 会打开一个交互窗口；在单独进程中调用即可实现多窗口
    window_title = f"Trimesh Viewer - {os.path.basename(path)}"
    try:
        mesh.show(window_title=window_title)
    except TypeError:
        # 旧版本 trimesh 可能不支持 window_title 参数
        mesh.show()


# ==================== Online Recon 部分 ====================

def load_camera_online(cfg, scene_path):
    """加载相机参数（用于 online_recon）"""
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    
    # Check if camera parameters exist in the file
    if 'org_width' in params and 'org_height' in params and 'w2c' in params and 'intrinsics' in params:
        # Use existing camera parameters
        org_width = params['org_width']
        org_height = params['org_height']
        w2c = params['w2c']
        intrinsics = params['intrinsics']
        k = intrinsics[:3, :3]
        
        # Scale intrinsics to match the visualization resolution
        k[0, :] *= cfg['viz_w'] / org_width
        k[1, :] *= cfg['viz_h'] / org_height
    else:
        # Camera parameters not in file, compute from cam_unnorm_rots and cam_trans
        print("Warning: Camera parameters not found in .npz file, using defaults and computing from camera poses...")
        
        # Default values for Replica dataset
        org_width = 1200
        org_height = 680
        # Default intrinsics for Replica (fx, fy typically ~600 for 1200x680 resolution)
        fx = fy = 600.0
        cx = org_width / 2.0
        cy = org_height / 2.0
        
        # Compute first frame w2c from cam_unnorm_rots and cam_trans
        if 'cam_unnorm_rots' in params and 'cam_trans' in params:
            cam_rots = torch.tensor(params['cam_unnorm_rots']).cuda().float()
            cam_trans = torch.tensor(params['cam_trans']).cuda().float()
            
            # Handle different shapes: (1, 4, T) or (4, T) or (4,)
            if len(cam_rots.shape) == 3:
                cam_rot_1d = cam_rots[0, :, 0]
                cam_tran = cam_trans[0, :, 0]
            elif len(cam_rots.shape) == 2:
                cam_rot_1d = cam_rots[:, 0]
                cam_tran = cam_trans[:, 0]
            else:
                cam_rot_1d = cam_rots
                cam_tran = cam_trans
            
            # Normalize and convert to 2D for build_rotation (expects (N, 4))
            cam_rot_1d = F.normalize(cam_rot_1d, dim=0)
            cam_rot_2d = cam_rot_1d.unsqueeze(0)
            
            # build_rotation expects (N, 4) and returns (N, 3, 3)
            rot_matrix = build_rotation(cam_rot_2d)
            rot_matrix = rot_matrix[0]
            
            w2c = torch.eye(4).cuda().float()
            w2c[:3, :3] = rot_matrix
            w2c[:3, 3] = cam_tran
            w2c = w2c.cpu().numpy()
        else:
            print("Warning: cam_unnorm_rots and cam_trans not found, using identity camera pose")
            w2c = np.eye(4)
        
        # Create intrinsics matrix
        k = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        
        # Scale intrinsics to match the visualization resolution
        k[0, :] *= cfg['viz_w'] / org_width
        k[1, :] *= cfg['viz_h'] / org_height
    
    return w2c, k


def load_scene_data_online(scene_path):
    """加载场景数据（用于 online_recon）"""
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    params = all_params

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot_raw = params['cam_unnorm_rots'][..., t_i]
        cam_tran_raw = params['cam_trans'][..., t_i]
        
        # Ensure cam_rot is 1D, then normalize
        if len(cam_rot_raw.shape) > 1:
            cam_rot_1d = cam_rot_raw.flatten()[:4]
        else:
            cam_rot_1d = cam_rot_raw
        
        # Ensure cam_tran is 1D
        if len(cam_tran_raw.shape) > 1:
            cam_tran = cam_tran_raw.flatten()[:3]
        else:
            cam_tran = cam_tran_raw
        
        cam_rot_1d = F.normalize(cam_rot_1d, dim=0)
        cam_rot_2d = cam_rot_1d.unsqueeze(0)
        
        rot_matrix = build_rotation(cam_rot_2d)
        rot_matrix = rot_matrix[0]
        
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = rot_matrix
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())
    
    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    return params, all_w2cs


def get_rendervars_online(params, w2c, curr_timestep):
    """获取渲染变量（用于 online_recon）"""
    # Check if timestep field exists for filtering Gaussians
    if 'timestep' in params:
        params_timesteps = params['timestep']
        selected_params_idx = params_timesteps <= curr_timestep
    else:
        selected_params_idx = torch.ones(params['means3D'].shape[0], dtype=torch.bool, device='cuda')
    
    keys = [k for k in params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices', 'timestep']]
    selected_params = deepcopy(params)
    for k in keys:
        selected_params[k] = selected_params[k][selected_params_idx]
    if selected_params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(selected_params['log_scales'], (1, 3))
    else:
        log_scales = selected_params['log_scales']
    w2c = torch.tensor(w2c).cuda().float()
    rendervar = {
        'means3D': selected_params['means3D'],
        'colors_precomp': selected_params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
    }
    depth_rendervar = {
        'means3D': selected_params['means3D'],
        'colors_precomp': get_depth_and_silhouette(selected_params['means3D'], w2c),
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
    }
    return rendervar, depth_rendervar


def make_lineset(all_pts, all_cols, num_lines):
    """创建线条集"""
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render_online(w2c, k, timestep_data, timestep_depth_data, cfg):
    """渲染（用于 online_recon）"""
    with torch.no_grad():
        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
        white_bg_cam = Camera(
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
            prefiltered=cam.prefiltered
        )
        im, _, depth, = Renderer(raster_settings=white_bg_cam)(**timestep_data)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**timestep_depth_data)
        differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, sil


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    """将 RGBD 转换为点云"""
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    
    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def visualize_online(scene_path, cfg):
    """在线重建可视化（用于 online_recon）"""
    first_frame_w2c, k = load_camera_online(cfg, scene_path)
    params, all_w2cs = load_scene_data_online(scene_path)
    print(params['means3D'].shape)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']), 
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    scene_data, scene_depth_data = get_rendervars_online(params, first_frame_w2c, curr_timestep=0)
    im, depth, sil = render_online(first_frame_w2c, k, scene_data, scene_depth_data, cfg)
    init_pts, init_cols = rgbd2pcd(im, depth, first_frame_w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    # Initialize Estimated Camera Frustums
    frustum_size = 0.045
    num_t = len(all_w2cs)
    cam_centers = []
    cam_colormap = plt.get_cmap('cool')
    norm_factor = 0.5
    total_num_lines = num_t - 1
    line_colormap = plt.get_cmap('cool')
    
    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    first_view_w2c = first_frame_w2c
    first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])
    cparams.extrinsic = first_view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # Rendering of Online Reconstruction
    start_time = time.time()
    num_timesteps = num_t
    viz_start = True
    curr_timestep = 0
    while curr_timestep < (num_timesteps-1) or not cfg['enter_interactive_post_online']:
        passed_time = time.time() - start_time
        passed_frames = passed_time * cfg['viz_fps']
        curr_timestep = int(passed_frames % num_timesteps)
        if not viz_start:
            if curr_timestep == prev_timestep:
                continue

        # Update Camera Frustum
        if curr_timestep == 0:
            cam_centers = []
            if not viz_start:
                vis.remove_geometry(prev_lines)
        if not viz_start:
            vis.remove_geometry(prev_frustum)
        new_frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[curr_timestep], frustum_size)
        new_frustum.paint_uniform_color(np.array(cam_colormap(curr_timestep * norm_factor / num_t)[:3]))
        vis.add_geometry(new_frustum)
        prev_frustum = new_frustum
        cam_centers.append(np.linalg.inv(all_w2cs[curr_timestep])[:3, 3])
        
        # Update Camera Trajectory
        if len(cam_centers) > 1 and curr_timestep > 0:
            num_lines = [1]
            cols = []
            for line_t in range(curr_timestep):
                cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
            cols = np.array(cols)
            all_cols = [cols]
            out_pts = [np.array(cam_centers)]
            linesets = make_lineset(out_pts, all_cols, num_lines)
            lines = o3d.geometry.LineSet()
            lines.points = linesets[0].points
            lines.colors = linesets[0].colors
            lines.lines = linesets[0].lines
            vis.add_geometry(lines)
            prev_lines = lines
        elif not viz_start:
            vis.remove_geometry(prev_lines)

        # Get Current View Camera
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        view_w2c = cam_params.extrinsic
        view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
        cam_params.extrinsic = view_w2c
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

        scene_data, scene_depth_data = get_rendervars_online(params, view_w2c, curr_timestep=curr_timestep)
        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render_online(view_w2c, k, scene_data, scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, view_w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
        prev_timestep = curr_timestep
        viz_start = False

    # Enter Interactive Mode once all frames have been visualized
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render_online(w2c, k, scene_data, scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
    
    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


def run_online_recon(scene_path):
    """运行在线重建可视化的包装函数（用于多进程）"""
    if not VISUALIZATION_AVAILABLE:
        print("错误: 可视化库不可用，无法运行在线重建可视化。请安装所需依赖库。")
        return
    
    # 使用默认可视化配置
    viz_cfg = dict(
        render_mode='color',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=True,
    )
    
    seed_everything(seed=42)
    visualize_online(scene_path, viz_cfg)


# ==================== Final Recon 部分 ====================

def load_camera_final(cfg, scene_path):
    """加载相机参数（用于 final_recon）"""
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    
    # Check if camera parameters exist in the file
    if 'org_width' in params and 'org_height' in params and 'w2c' in params and 'intrinsics' in params:
        org_width = params['org_width']
        org_height = params['org_height']
        w2c = params['w2c']
        intrinsics = params['intrinsics']
        k = intrinsics[:3, :3]
        
        k[0, :] *= cfg['viz_w'] / org_width
        k[1, :] *= cfg['viz_h'] / org_height
    else:
        print("Warning: Camera parameters not found in .npz file, using defaults and computing from camera poses...")
        
        org_width = 1200
        org_height = 680
        fx = fy = 600.0
        cx = org_width / 2.0
        cy = org_height / 2.0
        
        if 'cam_unnorm_rots' in params and 'cam_trans' in params:
            cam_rots = torch.tensor(params['cam_unnorm_rots']).cuda().float()
            cam_trans = torch.tensor(params['cam_trans']).cuda().float()
            
            if len(cam_rots.shape) == 3:
                cam_rot_1d = cam_rots[0, :, 0]
                cam_tran = cam_trans[0, :, 0]
            elif len(cam_rots.shape) == 2:
                cam_rot_1d = cam_rots[:, 0]
                cam_tran = cam_trans[:, 0]
            else:
                cam_rot_1d = cam_rots
                cam_tran = cam_trans
            
            cam_rot_1d = F.normalize(cam_rot_1d, dim=0)
            cam_rot_2d = cam_rot_1d.unsqueeze(0)
            
            rot_matrix = build_rotation(cam_rot_2d)
            rot_matrix = rot_matrix[0]
            
            w2c = torch.eye(4).cuda().float()
            w2c[:3, :3] = rot_matrix
            w2c[:3, 3] = cam_tran
            w2c = w2c.cpu().numpy()
        else:
            print("Warning: cam_unnorm_rots and cam_trans not found, using identity camera pose")
            w2c = np.eye(4)
        
        k = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        
        k[0, :] *= cfg['viz_w'] / org_width
        k[1, :] *= cfg['viz_h'] / org_height
    
    return w2c, k


def load_scene_data_final(scene_path, first_frame_w2c, intrinsics):
    """加载场景数据（用于 final_recon）"""
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    # Check if Gaussians are Isotropic or Anisotropic
    if params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    depth_rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], first_frame_w2c),
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    return rendervar, depth_rendervar, all_w2cs


def render_final(w2c, k, timestep_data, timestep_depth_data, cfg):
    """渲染（用于 final_recon）"""
    with torch.no_grad():
        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
        white_bg_cam = Camera(
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
            prefiltered=cam.prefiltered
        )
        im, _, depth, = Renderer(raster_settings=white_bg_cam)(**timestep_data)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**timestep_depth_data)
        differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, sil


def visualize_final(scene_path, cfg):
    """最终重建可视化（用于 final_recon）"""
    w2c, k = load_camera_final(cfg, scene_path)
    scene_data, scene_depth_data, all_w2cs = load_scene_data_final(scene_path, w2c, k)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']), 
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    im, depth, sil = render_final(w2c, k, scene_data, scene_depth_data, cfg)
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    if cfg['visualize_cams']:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])
        
        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg['offset_first_viz_cam']:
        view_w2c = w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = w2c
    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # Interactive Rendering
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render_final(w2c, k, scene_data, scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


def run_final_recon(scene_path):
    """运行最终重建可视化的包装函数（用于多进程）"""
    if not VISUALIZATION_AVAILABLE:
        print("错误: 可视化库不可用，无法运行最终重建可视化。请安装所需依赖库。")
        return
    
    # 使用默认可视化配置
    viz_cfg = dict(
        render_mode='color',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=False,
    )
    
    seed_everything(seed=42)
    visualize_final(scene_path, viz_cfg)


# ==================== Model Browser App 部分 ====================

class ModelBrowserApp:
    """
    基于 tkinter 的简单模型浏览器，用于扫描当前文件所在目录及其所有子目录，并通过 Trimesh 查看模型。
    """

    def __init__(self, root):
        self.root = root
        self.root.title("IsoGS Model Browser")
        self.root.geometry("800x600")

        # 路径设置：扫描当前文件所在目录及其所有子目录
        self.scan_dir = get_scan_directory()

        # 保存子进程引用，避免被 GC（可选）
        self.viewer_processes = []
        # 扩展名过滤：默认只显示 .ply
        self.show_ply_var = tk.BooleanVar(value=True)
        self.show_obj_var = tk.BooleanVar(value=False)
        self.show_npz_var = tk.BooleanVar(value=False)
        # .npz 打开方式（静态 / 动态），互斥且至少一个为 True
        self.method_npz_final_var = tk.BooleanVar(value=True)   # 静态查看（final_recon）
        self.method_npz_online_var = tk.BooleanVar(value=False)  # 动态查看（online_recon）
        # 名称前缀过滤：splat* / mesh*，默认只启用 mesh*
        self.filter_splat_var = tk.BooleanVar(value=False)
        self.filter_mesh_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.scan_models()

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        # 顶层使用垂直方向的两块：上部列表，下部按钮与状态栏
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.columnconfigure(0, weight=1)

        # 上部：Treeview + 滚动条
        frame_list = ttk.Frame(self.root)
        frame_list.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        frame_list.rowconfigure(0, weight=1)
        frame_list.columnconfigure(0, weight=1)

        # 使用树形结构：#0 作为名称列，额外列为大小、最后修改时间和完整路径
        columns = ("size", "mtime", "fullpath")
        self.tree = ttk.Treeview(
            frame_list,
            columns=columns,
            show="tree headings",
            selectmode="browse",
        )

        # #0 列显示名称（文件 / 文件夹）
        self.tree.heading("#0", text="Name")
        self.tree.column("#0", anchor="w", width=500, stretch=True)

        # 附加列：大小、最后修改时间；fullpath 作为隐藏列保存绝对路径
        self.tree.heading("size", text="Size (MB)")
        self.tree.heading("mtime", text="Modified")
        self.tree.heading("fullpath", text="Full Path")

        self.tree.column("size", anchor="e", width=80, stretch=False)
        self.tree.column("mtime", anchor="center", width=140, stretch=False)
        # fullpath 作为隐藏列
        self.tree.column("fullpath", width=0, stretch=False)
        self.tree["displaycolumns"] = ("size", "mtime")

        vsb = ttk.Scrollbar(frame_list, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame_list, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # 双击事件：打开模型
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        # 下部：按钮 + 状态栏
        frame_bottom = ttk.Frame(self.root)
        frame_bottom.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        # 列：0 为左侧控件，1 为状态栏（自适应）
        frame_bottom.columnconfigure(0, weight=0)
        frame_bottom.columnconfigure(1, weight=1)

        # 第 0 行：扩展名过滤复选框
        chk_ply = ttk.Checkbutton(
            frame_bottom,
            text=".ply",
            variable=self.show_ply_var,
            command=self.scan_models,
        )
        chk_ply.grid(row=0, column=0, sticky="w", padx=(0, 8))

        chk_obj = ttk.Checkbutton(
            frame_bottom,
            text=".obj",
            variable=self.show_obj_var,
            command=self.scan_models,
        )
        chk_obj.grid(row=0, column=0, sticky="w", padx=(60, 8))

        chk_npz = ttk.Checkbutton(
            frame_bottom,
            text=".npz",
            variable=self.show_npz_var,
            command=self.scan_models,
        )
        chk_npz.grid(row=0, column=0, sticky="w", padx=(120, 8))

        # 第 1 行：名称前缀过滤复选框
        chk_splat = ttk.Checkbutton(
            frame_bottom,
            text="splat*",
            variable=self.filter_splat_var,
            command=self.scan_models,
        )
        chk_splat.grid(row=1, column=0, sticky="w", padx=(0, 8))

        chk_mesh = ttk.Checkbutton(
            frame_bottom,
            text="mesh*",
            variable=self.filter_mesh_var,
            command=self.scan_models,
        )
        chk_mesh.grid(row=1, column=0, sticky="w", padx=(80, 8))

        # 第 2 行：.npz 打开方式（静态 / 动态），互斥
        def on_npz_method_changed(changed_var):
            """
            保证 method_npz_final_var 和 method_npz_online_var 互斥且至少选一个。
            changed_var: 被用户点击的 BooleanVar。
            """
            # 若用户把某个选项从 False 点成 True，则将另一个设为 False
            if changed_var.get():
                if changed_var is self.method_npz_final_var:
                    self.method_npz_online_var.set(False)
                else:
                    self.method_npz_final_var.set(False)
            else:
                # 用户尝试取消当前选项，如果另一个也是 False，则强制保持当前为 True，保证至少一个
                other = self.method_npz_online_var if changed_var is self.method_npz_final_var else self.method_npz_final_var
                if not other.get():
                    changed_var.set(True)

        chk_npz_final = ttk.Checkbutton(
            frame_bottom,
            text="npz: 静态 (final)",
            variable=self.method_npz_final_var,
            command=lambda: on_npz_method_changed(self.method_npz_final_var),
        )
        chk_npz_final.grid(row=2, column=0, sticky="w", padx=(0, 8))

        chk_npz_online = ttk.Checkbutton(
            frame_bottom,
            text="npz: 动态 (online)",
            variable=self.method_npz_online_var,
            command=lambda: on_npz_method_changed(self.method_npz_online_var),
        )
        chk_npz_online.grid(row=2, column=0, sticky="w", padx=(130, 8))

        # 第 3 行：刷新按钮 + 状态栏
        self.btn_refresh = ttk.Button(frame_bottom, text="刷新 (Rescan)", command=self.scan_models)
        self.btn_refresh.grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(4, 0))

        self.status_var = tk.StringVar(value="")
        self.lbl_status = ttk.Label(frame_bottom, textvariable=self.status_var, anchor="w")
        self.lbl_status.grid(row=3, column=1, sticky="ew", padx=(12, 0), pady=(4, 0))

    # ---------------- 目录扫描 ----------------
    def scan_models(self):
        """扫描当前文件所在目录及其所有子目录，查找 .ply / .obj / .npz 文件。"""
        self.tree.delete(*self.tree.get_children())

        if not os.path.isdir(self.scan_dir):
            msg = f"扫描目录不存在：{self.scan_dir}"
            self.status_var.set(msg)
            messagebox.showwarning("目录不存在", msg)
            return

        # 根据复选框决定要显示的扩展名
        exts = set()
        if self.show_ply_var.get():
            exts.add(".ply")
        if self.show_obj_var.get():
            exts.add(".obj")
        if self.show_npz_var.get():
            exts.add(".npz")

        # 名称前缀过滤：若至少勾选一个，则只保留以这些前缀开头的文件；否则不过滤前缀
        prefix_filters = []
        if self.filter_splat_var.get():
            prefix_filters.append("splat")
        if self.filter_mesh_var.get():
            prefix_filters.append("mesh")

        # 按目录收集文件，便于在每个子目录内按修改时间降序排序
        files_by_dir = {}

        for root_dir, _, files in os.walk(self.scan_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if exts and ext not in exts:
                    continue

                # 前缀过滤：只作用于 .ply / .obj，.npz 不受文件名前缀约束
                if ext in (".ply", ".obj") and prefix_filters:
                    lower_name = name.lower()
                    if not any(lower_name.startswith(p) for p in prefix_filters):
                        continue

                full_path = os.path.join(root_dir, name)
                try:
                    size_bytes = os.path.getsize(full_path)
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    size_bytes = 0
                    mtime = 0.0

                rel_path = os.path.relpath(full_path, self.scan_dir)
                rel_dir = os.path.dirname(rel_path)
                file_name = os.path.basename(rel_path)
                files_by_dir.setdefault(rel_dir, []).append(
                    (file_name, full_path, size_bytes, mtime)
                )

        # 逐目录创建节点，并在每个目录内按 mtime 降序插入文件
        for rel_dir in sorted(files_by_dir.keys()):
            # 创建目录节点链
            parent = ""
            if rel_dir not in ("", "."):
                parts = rel_dir.split(os.sep)
                current_parts = []
                for folder in parts:
                    current_parts.append(folder)
                    node_id = "/".join(current_parts)
                    if not self.tree.exists(node_id):
                        self.tree.insert(
                            parent,
                            "end",
                            iid=node_id,
                            text=folder,
                            values=("", "", ""),
                            open=False,
                        )
                    parent = node_id

            # 在该目录下按修改时间排序后插入文件（最新在前）
            files_list = files_by_dir[rel_dir]
            files_list.sort(key=lambda x: x[3], reverse=True)
            for file_name, full_path, size_bytes, mtime in files_list:
                size_mb = size_bytes / (1024 * 1024.0)
                size_str = f"{size_mb:.2f}"
                mtime_str = (
                    datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    if mtime > 0
                    else ""
                )
                self.tree.insert(
                    parent,
                    "end",
                    text=file_name,
                    values=(size_str, mtime_str, full_path),
                )

        # 递归展开所有节点，默认全展开
        def expand_all(parent=""):
            children = self.tree.get_children(parent)
            for child in children:
                self.tree.item(child, open=True)
                expand_all(child)

        expand_all("")

        # 更新状态栏
        exts_desc = []
        if ".ply" in exts:
            exts_desc.append(".ply")
        if ".obj" in exts:
            exts_desc.append(".obj")
        if ".npz" in exts:
            exts_desc.append(".npz")
        if not exts_desc:
            exts_text = "无扩展名被选中"
        else:
            exts_text = ", ".join(exts_desc)

        self.status_var.set(f"已扫描：{self.scan_dir}  |  过滤：{exts_text}")

    # ---------------- 事件处理 ----------------
    def on_tree_double_click(self, event):
        """双击某一条目时，打开对应的模型查看器进程。"""
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return

        values = self.tree.item(item_id, "values")
        if not values or len(values) < 3:
            # 可能是纯目录节点或数据异常
            # 对目录节点，切换展开/折叠状态
            item = self.tree.item(item_id)
            has_children = bool(self.tree.get_children(item_id))
            if has_children:
                self.tree.item(item_id, open=not item.get("open", False))
            return

        size_str, mtime_str, full_path = values[0], values[1], values[2]
        if not full_path or os.path.isdir(full_path):
            # 目录节点：仅展开/折叠
            item = self.tree.item(item_id)
            self.tree.item(item_id, open=not item.get("open", False))
            return

        self.open_viewer_process(full_path)

    def open_viewer_process(self, full_path):
        """启动独立进程，显示指定模型或 .npz 结果。"""
        if not os.path.isfile(full_path):
            messagebox.showerror("文件不存在", f"无法找到文件：\n{full_path}")
            return

        _, ext = os.path.splitext(full_path)
        ext = ext.lower()

        try:
            if ext in [".ply", ".obj"]:
                # 传统 mesh：用 Trimesh 查看
                p = mp.Process(target=view_model, args=(full_path,))
                p.daemon = False  # 允许独立存在，直到窗口关闭
                p.start()
                self.viewer_processes.append(p)
                self.status_var.set(f"打开模型：{os.path.relpath(full_path, self.scan_dir)}")
            elif ext == ".npz":
                # .npz：根据当前选择，调用 final_recon 或 online_recon
                self.open_npz_viewer(full_path)
            else:
                messagebox.showwarning("不支持的文件类型", f"暂不支持打开该类型文件：\n{full_path}")
        except Exception as e:
            messagebox.showerror("启动查看器失败", f"无法启动查看器进程：\n{e}")

    def open_npz_viewer(self, full_path):
        """
        根据 .npz 打开方式选择，调用 final_recon 或 online_recon 在独立进程中进行可视化。
        """
        # 保证互斥和至少一个选中（防御性检查，正常情况下 UI 已经保证）
        if not self.method_npz_final_var.get() and not self.method_npz_online_var.get():
            self.method_npz_final_var.set(True)
        if self.method_npz_final_var.get() and self.method_npz_online_var.get():
            self.method_npz_online_var.set(False)

        if not VISUALIZATION_AVAILABLE:
            messagebox.showerror(
                "可视化库不可用",
                "无法打开 .npz 文件：\n可视化所需的库未正确导入。\n请按照提示安装缺少的依赖库。"
            )
            return

        try:
            if self.method_npz_final_var.get():
                # 静态查看（final_recon）
                p = mp.Process(target=run_final_recon, args=(full_path,))
                mode_desc = "静态 (final_recon)"
            else:
                # 动态查看（online_recon）
                p = mp.Process(target=run_online_recon, args=(full_path,))
                mode_desc = "动态 (online_recon)"
            
            p.daemon = False  # 允许独立存在，直到窗口关闭
            p.start()
            self.viewer_processes.append(p)
            rel = os.path.relpath(full_path, self.scan_dir)
            self.status_var.set(f"打开 .npz（{mode_desc}）：{rel}")
        except Exception as e:
            messagebox.showerror("启动 .npz 查看器失败", f"无法启动 .npz 查看器进程：\n{e}")


# ==================== Main 入口 ====================

def main_browser():
    """启动模型浏览器 GUI"""
    # 在 Linux 上默认使用 fork，一般没问题；若需兼容性更好可选择 spawn
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        # start method 已经设置过，忽略
        pass

    root = tk.Tk()
    app = ModelBrowserApp(root)
    root.mainloop()


def main_online_recon():
    """命令行入口：在线重建可视化"""
    parser = argparse.ArgumentParser(description="Visualize online 3D Gaussian Splatting reconstruction")
    parser.add_argument("input", type=str, help="Path to experiment file (.py) or .npz file")

    args = parser.parse_args()

    if not VISUALIZATION_AVAILABLE:
        print("错误: 可视化库不可用，请安装所需依赖库。")
        sys.exit(1)

    # Check if input is a .npz file (direct file specification)
    if args.input.endswith('.npz') and os.path.isfile(args.input):
        scene_path = os.path.abspath(args.input)
        print(f"Using specified .npz file: {scene_path}")
        
        viz_cfg = dict(
            render_mode='color',
            offset_first_viz_cam=True,
            show_sil=False,
            visualize_cams=True,
            viz_w=600, viz_h=340,
            viz_near=0.01, viz_far=100.0,
            view_scale=2,
            viz_fps=5,
            enter_interactive_post_online=True,
        )
        
        seed_everything(seed=42)
    else:
        # Original logic: load experiment configuration file
        experiment = SourceFileLoader(
            os.path.basename(args.input), args.input
        ).load_module()

        seed_everything(seed=experiment.config["seed"])

        if "scene_path" not in experiment.config:
            results_dir = os.path.join(
                experiment.config["workdir"], experiment.config["run_name"]
            )
            
            params_npz_path = os.path.join(results_dir, "params.npz")
            
            if os.path.exists(params_npz_path):
                scene_path = params_npz_path
                print(f"Found final params file: {scene_path}")
            else:
                pattern = re.compile(r'^params(\d+)\.npz$')
                checkpoint_files = []
                
                if os.path.exists(results_dir):
                    for filename in os.listdir(results_dir):
                        match = pattern.match(filename)
                        if match:
                            checkpoint_num = int(match.group(1))
                            checkpoint_files.append((checkpoint_num, filename))
                
                if checkpoint_files:
                    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
                    latest_checkpoint = checkpoint_files[0]
                    scene_path = os.path.join(results_dir, latest_checkpoint[1])
                    print(f"Found latest checkpoint file: {scene_path} (frame {latest_checkpoint[0]})")
                else:
                    raise FileNotFoundError(f"No params file found in {results_dir}. Please check if the experiment has been run.")
        else:
            scene_path = experiment.config["scene_path"]
        viz_cfg = experiment.config["viz"]

    visualize_online(scene_path, viz_cfg)


def main_final_recon():
    """命令行入口：最终重建可视化"""
    parser = argparse.ArgumentParser(description="Visualize 3D Gaussian Splatting reconstruction")
    parser.add_argument("input", type=str, help="Path to experiment file (.py) or .npz file")

    args = parser.parse_args()

    if not VISUALIZATION_AVAILABLE:
        print("错误: 可视化库不可用，请安装所需依赖库。")
        sys.exit(1)

    # Check if input is a .npz file (direct file specification)
    if args.input.endswith('.npz') and os.path.isfile(args.input):
        scene_path = os.path.abspath(args.input)
        print(f"Using specified .npz file: {scene_path}")
        
        viz_cfg = dict(
            render_mode='color',
            offset_first_viz_cam=True,
            show_sil=False,
            visualize_cams=True,
            viz_w=600, viz_h=340,
            viz_near=0.01, viz_far=100.0,
            view_scale=2,
            viz_fps=5,
            enter_interactive_post_online=False,
        )
        
        seed_everything(seed=42)
    else:
        # Original logic: load experiment configuration file
        experiment = SourceFileLoader(
            os.path.basename(args.input), args.input
        ).load_module()

        seed_everything(seed=experiment.config["seed"])

        if "scene_path" not in experiment.config:
            results_dir = os.path.join(
                experiment.config["workdir"], experiment.config["run_name"]
            )
            
            params_npz_path = os.path.join(results_dir, "params.npz")
            
            if os.path.exists(params_npz_path):
                scene_path = params_npz_path
                print(f"Found final params file: {scene_path}")
            else:
                pattern = re.compile(r'^params(\d+)\.npz$')
                checkpoint_files = []
                
                if os.path.exists(results_dir):
                    for filename in os.listdir(results_dir):
                        match = pattern.match(filename)
                        if match:
                            checkpoint_num = int(match.group(1))
                            checkpoint_files.append((checkpoint_num, filename))
                
                if checkpoint_files:
                    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
                    latest_checkpoint = checkpoint_files[0]
                    scene_path = os.path.join(results_dir, latest_checkpoint[1])
                    print(f"Found latest checkpoint file: {scene_path} (frame {latest_checkpoint[0]})")
                else:
                    raise FileNotFoundError(f"No params file found in {results_dir}. Please check if the experiment has been run.")
        else:
            scene_path = experiment.config["scene_path"]
        viz_cfg = experiment.config["viz"]

    visualize_final(scene_path, viz_cfg)


if __name__ == "__main__":
    # 根据命令行参数决定运行模式
    if len(sys.argv) > 1:
        # 如果有命令行参数，检查是否是 online_recon 或 final_recon 模式
        # 通过检查脚本名称或第一个参数来判断
        script_name = os.path.basename(sys.argv[0])
        first_arg = sys.argv[1] if len(sys.argv) > 1 else ""
        
        # 如果第一个参数是 --online，运行在线重建
        if first_arg == "--online" or script_name == "online_recon.py":
            if first_arg == "--online":
                sys.argv = [sys.argv[0]] + sys.argv[2:]  # 移除 --online 参数
            main_online_recon()
        # 如果第一个参数是 --final，运行最终重建
        elif first_arg == "--final" or script_name == "final_recon.py":
            if first_arg == "--final":
                sys.argv = [sys.argv[0]] + sys.argv[2:]  # 移除 --final 参数
            main_final_recon()
        # 否则运行浏览器 GUI
        else:
            main_browser()
    else:
        # 没有参数，运行浏览器 GUI
        main_browser()
