import argparse
import os
import sys
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

# ==========================================
# 路径设置：确保能导入项目模块
# ==========================================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.slam_helpers import transform_to_frame, transformed_params2depthplussilhouette
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation
from datasets.gradslam_datasets import load_dataset_config
from scripts.splatam import get_dataset # 复用 splatam.py 中的数据集加载逻辑

def parse_args():
    parser = argparse.ArgumentParser(description="Fast Mesh Extraction via TSDF Fusion (Projecting Rendered Depth)")
    parser.add_argument("config", type=str, help="Path to experiment config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file (optional)")
    parser.add_argument("--output", type=str, default=None, help="Output mesh file path (default: mesh_tsdf.ply)")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="TSDF Voxel Size in meters (smaller = more detail)")
    parser.add_argument("--skip", type=int, default=5, help="Frame skipping interval (e.g. use every 5th frame)")
    parser.add_argument("--depth_trunc", type=float, default=3.0, help="Max depth range to integrate (meters)")
    parser.add_argument("--sdf_trunc", type=float, default=0.04, help="Truncation margin (usually 3-5x voxel size)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for rendering")
    parser.add_argument("--no_cleaning", action="store_true", help="Skip mesh cleaning (largest component)")
    parser.add_argument("--run_full", action="store_true", help="Run full TSDF fusion after point cloud debug check")
    return parser.parse_args()

def load_checkpoint_and_config(args):
    # 1. 加载 Config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    result_dir = os.path.join(config['workdir'], config['run_name'])
    
    # 2. 自动寻找 Checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # 找最新的 params*.npz
        import glob
        import re
        ckpt_files = glob.glob(os.path.join(result_dir, "params*.npz"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {result_dir}")
        # 按数字排序
        def extract_num(path):
            m = re.search(r'params(\d+).npz', path)
            return int(m.group(1)) if m else -1
        ckpt_files.sort(key=extract_num)
        ckpt_path = ckpt_files[-1] # 最新的
    
    print(f"Loading checkpoint: {ckpt_path}")
    params_np = dict(np.load(ckpt_path, allow_pickle=True))
    
    # 3. 转换为 Tensor
    params = {}
    for k, v in params_np.items():
        # 过滤掉非 Tensor 数据
        if k in ['gt_w2c_all_frames', 'keyframe_time_indices', 'intrinsics', 'w2c', 'org_width', 'org_height']: 
            continue
        if isinstance(v, np.ndarray):
            params[k] = torch.tensor(v).to(args.device).float().requires_grad_(False)
            
    return config, params, params_np, result_dir

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # --- 加载数据 ---
    config, params, params_raw, result_dir = load_checkpoint_and_config(args)
    
    # --- 获取相机参数 (优先从 checkpoint 读，如果没有则加载 Dataset) ---
    print("[Loader] Loading Camera Parameters...")
    if 'intrinsics' in params_raw:
        K = params_raw['intrinsics']
        width = int(params_raw['org_width'])
        height = int(params_raw['org_height'])
        print("[Loader] ✓ Loaded camera params from checkpoint metadata.")
    else:
        # Fallback: 加载数据集来获取第一帧内参
        print("[Loader] ! Camera metadata not in checkpoint. Loading dataset...")
        dataset_config = config["data"]
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
        dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=0, end=1, stride=1,
            desired_height=dataset_config["desired_image_height"],
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True
        )
        _, _, intrinsics, pose = dataset[0]
        K = intrinsics[:3, :3].cpu().numpy()
        width = dataset_config["desired_image_width"]
        height = dataset_config["desired_image_height"]
        print("[Loader] ✓ Loaded camera params from Dataset.")

    # --- 坐标系变换矩阵定义 ---
    # OpenGL -> OpenCV 坐标系转换 (Y-up -> Y-down, Z-back -> Z-forward)
    # 左乘翻转矩阵，作用于相机坐标轴
    flip_mat = np.diag([1, -1, -1, 1]).astype(np.float64)
    
    # Open3D 内参对象
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])

    # --- 获取轨迹长度 ---
    if 'cam_trans' in params:
        num_frames = params['cam_trans'].shape[-1]
    else:
        raise ValueError("Checkpoint does not contain trajectory (cam_trans).")

    print(f"[System] Total frames: {num_frames}")
    print(f"[System] Image size: {width}x{height}")
    print(f"[System] Voxel size: {args.voxel_size}m, SDF trunc: {args.sdf_trunc}m")

    # ============================================================
    # 阶段一：点云验证 (Point Cloud Debug)
    # ============================================================
    print("\n" + "="*70)
    print("[Stage 1] Point Cloud Alignment Check (First 20 frames)")
    print("="*70)
    
    debug_pcd = o3d.geometry.PointCloud()
    debug_frames = min(20, num_frames)
    
    print(f"[Debug] Processing first {debug_frames} frames for alignment check...")
    
    for time_idx in tqdm(range(debug_frames), desc="Debug Point Cloud"):
        # 1. 计算位姿 (W2C in OpenGL)
        cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
        
        w2c_opengl = torch.eye(4).to(device).float()
        w2c_opengl[:3, :3] = build_rotation(cam_rot)
        w2c_opengl[:3, 3] = cam_tran
        w2c_opengl_np = w2c_opengl.cpu().numpy()
        
        # 2. 转换为 OpenCV 坐标系
        # w2c_opencv = flip_mat @ w2c_opengl (左乘)
        w2c_opencv = flip_mat @ w2c_opengl_np
        
        # 3. 渲染深度图（单位：米，float32）
        cam = setup_camera(width, height, K, w2c_opengl_np)
        transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
        depth_rendervar = transformed_params2depthplussilhouette(params, w2c_opengl, transformed_gaussians)
        depth_sil, _, _ = Renderer(raster_settings=cam)(**depth_rendervar)
        # SplaTAM 渲染的深度单位为米，保持 float32
        depth_np = depth_sil[0, :, :].detach().cpu().numpy().astype(np.float32)
        
        # 4. 从深度图生成点云（相机坐标系，OpenCV）
        # 关键：depth_scale=1.0（深度单位为米，不是毫米）
        depth_o3d = o3d.geometry.Image(depth_np)
        pcd_cam = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, 
            o3d_intrinsic,
            depth_scale=1.0,  # 显式强制设置为 1.0（米）
            depth_trunc=args.depth_trunc
        )
        
        # 5. 变换到世界坐标系（使用 C2W = inv(W2C)）
        c2w_opencv = np.linalg.inv(w2c_opencv)
        pcd_cam.transform(c2w_opencv)
        
        # 6. 累加到主点云
        debug_pcd += pcd_cam
    
    # 保存点云用于检查
    debug_pcd_path = os.path.join(result_dir, "debug_alignment_check.ply")
    o3d.io.write_point_cloud(debug_pcd_path, debug_pcd)
    print(f"\n[Debug] ✓ Saved alignment check point cloud: {debug_pcd_path}")
    print(f"[Debug]   Points: {len(debug_pcd.points)}")
    print("\n" + "!"*70)
    print("  IMPORTANT: Please open debug_alignment_check.ply and verify alignment!")
    print("  - If walls/structures are aligned correctly, the coordinate transform is correct.")
    print("  - If you see duplicate/ghost structures, there may still be a coordinate issue.")
    print("!"*70)
    
    if not args.run_full:
        print("\n[Info] To run full TSDF fusion, use --run_full flag.")
        return
    
    # ============================================================
    # 阶段二：TSDF 融合 (Full Fusion)
    # ============================================================
    print("\n" + "="*70)
    print("[Stage 2] Full TSDF Fusion")
    print("="*70)
    
    print(f"[TSDF] Initializing TSDF Volume (Voxel: {args.voxel_size}m, SDF Trunc: {args.sdf_trunc}m)...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    frame_range = range(0, num_frames, args.skip)
    print(f"[TSDF] Processing {len(frame_range)} frames (every {args.skip} frames)...")
    
    for time_idx in tqdm(frame_range, desc="TSDF Fusion"):
        # 1. 计算位姿 (W2C in OpenGL)
        cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
        
        w2c_opengl = torch.eye(4).to(device).float()
        w2c_opengl[:3, :3] = build_rotation(cam_rot)
        w2c_opengl[:3, 3] = cam_tran
        w2c_opengl_np = w2c_opengl.cpu().numpy()
        
        # 2. 渲染深度图（单位：米，float32）
        # SplaTAM 使用 OpenGL 坐标系渲染
        cam = setup_camera(width, height, K, w2c_opengl_np)
        transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
        depth_rendervar = transformed_params2depthplussilhouette(params, w2c_opengl, transformed_gaussians)
        depth_sil, _, _ = Renderer(raster_settings=cam)(**depth_rendervar)
        # 深度单位为米，保持 float32
        depth_np = depth_sil[0, :, :].detach().cpu().numpy().astype(np.float32)
        
        # 3. 创建 RGBD（关键：depth_scale=1.0，强制深度单位为米）
        color_np = np.ones((height, width, 3), dtype=np.uint8) * 200  # 灰色占位
        color_o3d = o3d.geometry.Image(color_np)
        depth_o3d = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, 
            depth_o3d,
            depth_scale=1.0,  # 显式强制设置为 1.0（米），不能使用默认值
            depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # 4. 坐标系转换 (OpenGL W2C -> OpenCV W2C)
        # 左乘翻转矩阵，作用于相机坐标轴
        curr_w2c_opencv = flip_mat @ w2c_opengl_np
        
        # 5. 融合进 TSDF Volume
        # 关键：Open3D 的 integrate 第三个参数必须是 W2C 矩阵（World-to-Camera），不是 C2W
        volume.integrate(rgbd, o3d_intrinsic, curr_w2c_opencv)

    # ============================================================
    # 提取与保存 Mesh
    # ============================================================
    print("\n[Extract] Extracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    
    if not args.no_cleaning:
        print("[Extract] Cleaning mesh (removing small clusters)...")
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    
    # 保存
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(result_dir, "mesh_tsdf.ply")
        
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"[Extract] ✓ Saved Mesh: {output_path}")
    print(f"[Extract]   Vertices: {len(mesh.vertices)}")
    print(f"[Extract]   Triangles: {len(mesh.triangles)}")
    
    # 同时也存个 OBJ 方便看
    obj_path = output_path.replace(".ply", ".obj")
    o3d.io.write_triangle_mesh(obj_path, mesh)
    print(f"[Extract] ✓ Saved OBJ: {obj_path}")
    
    if len(mesh.vertices) < 1000:
        print("\n[Warning] Vertex count is suspiciously low! Check the debug point cloud first.")

if __name__ == "__main__":
    main()