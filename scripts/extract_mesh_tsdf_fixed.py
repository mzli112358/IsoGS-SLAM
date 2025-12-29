import argparse
import os
import sys
import numpy as np
import torch
import cv2
import open3d as o3d
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.slam_helpers import transform_to_frame, transformed_params2depthplussilhouette
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation
from datasets.gradslam_datasets import load_dataset_config
from scripts.splatam import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="TSDF Mesh Extraction (Fixed Coordinates)")
    parser.add_argument("config", type=str, help="Path to experiment config file")
    parser.add_argument("--output", type=str, default=None, help="Output mesh file path (default: mesh_tsdf_fixed.ply)")
    parser.add_argument("--voxel_size", type=float, default=0.015, help="Voxel size (meters)")
    parser.add_argument("--skip", type=int, default=10, help="Frame skipping interval")
    parser.add_argument("--depth_trunc", type=float, default=4.0, help="Max depth range (meters)")
    parser.add_argument("--sdf_trunc", type=float, default=0.06, help="Truncation margin")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    # 添加一个开关，用于尝试翻转坐标系（默认开启）
    parser.add_argument("--no_fix_coords", action="store_true", help="Disable coordinate system fix (OpenGL -> OpenCV)")
    return parser.parse_args()

def load_checkpoint_and_config(args):
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    result_dir = os.path.join(config['workdir'], config['run_name'])
    
    import glob
    import re
    ckpt_files = glob.glob(os.path.join(result_dir, "params*.npz"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {result_dir}")
    
    def extract_num(path):
        m = re.search(r'params(\d+).npz', path)
        return int(m.group(1)) if m else -1
    ckpt_files.sort(key=extract_num)
    ckpt_path = ckpt_files[-1]
    
    print(f"[Loader] Loading checkpoint: {ckpt_path}")
    params_np = dict(np.load(ckpt_path, allow_pickle=True))
    
    params = {}
    for k, v in params_np.items():
        if k in ['gt_w2c_all_frames', 'keyframe_time_indices', 'intrinsics', 'w2c', 'org_width', 'org_height']: 
            continue
        if isinstance(v, np.ndarray):
            params[k] = torch.tensor(v).to(args.device).float().requires_grad_(False)
            
    return config, params, params_np, result_dir

def main():
    args = parse_args()
    device = torch.device(args.device)
    config, params, params_raw, result_dir = load_checkpoint_and_config(args)
    
    # --- Debug 输出目录 ---
    debug_dir = os.path.join(result_dir, "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"[Debug] Rendered depth maps will be saved to: {debug_dir}")

    # --- 加载相机参数 ---
    if 'intrinsics' in params_raw and 'w2c' in params_raw:
        K = params_raw['intrinsics']
        first_frame_w2c = torch.tensor(params_raw['w2c']).to(device).float()
        width = int(params_raw['org_width'])
        height = int(params_raw['org_height'])
    else:
        # Fallback
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
        first_frame_w2c = torch.linalg.inv(pose)
        width = dataset_config["desired_image_width"]
        height = dataset_config["desired_image_height"]

    print(f"[System] Image Size: {width}x{height}")
    print(f"[System] Voxel Size: {args.voxel_size}m, SDF Trunc: {args.sdf_trunc}m")

    # --- 初始化 Open3D Volume ---
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])

    num_frames = params['cam_trans'].shape[-1]
    
    # --- 关键：坐标系修正矩阵 (OpenGL -> OpenCV) ---
    # 绕 X 轴旋转 180 度：Y -> -Y, Z -> -Z
    flip_mat = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64)

    if not args.no_fix_coords:
        print("[Coordinate] Applying OpenGL -> OpenCV coordinate system fix")
    else:
        print("[Coordinate] Coordinate system fix is DISABLED (--no_fix_coords)")

    print(f"[Process] Starting integration ({num_frames} frames, skip={args.skip})...")
    
    for i, time_idx in enumerate(tqdm(range(0, num_frames, args.skip))):
        # 1. 计算位姿 (W2C)
        cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
        
        rel_w2c = torch.eye(4).to(device).float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        
        curr_w2c = rel_w2c @ first_frame_w2c
        
        # 2. 渲染深度
        cam = setup_camera(width, height, K, curr_w2c.cpu().numpy())
        transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
        depth_rendervar = transformed_params2depthplussilhouette(params, curr_w2c, transformed_gaussians)
        depth_sil, _, _ = Renderer(raster_settings=cam)(**depth_rendervar)
        depth = depth_sil[0, :, :]
        
        # 3. 准备数据给 Open3D
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        
        # [Debug] 保存前 5 帧深度图，确保渲染没问题
        if i < 5:
            # 归一化方便查看 (0-5m 映射到 0-255)
            depth_viz = np.clip(depth_np / 5.0 * 255, 0, 255).astype(np.uint8)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(debug_dir, f"depth_{i:03d}.png"), depth_viz)

        # Open3D 集成
        color_o3d = o3d.geometry.Image(np.ones((height, width, 3), dtype=np.uint8) * 200)
        depth_o3d = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False
        )
        
        # 4. 位姿转换 (Camera-to-World)
        curr_c2w = torch.inverse(curr_w2c).cpu().numpy()
        
        if not args.no_fix_coords:
            # 应用翻转：Open3D 期望 OpenCV 坐标系 (Y-down)
            # 而 curr_c2w 是 OpenGL (Y-up)
            # 变换逻辑： Pose_opencv = Pose_opengl @ Flip_mat
            curr_c2w = curr_c2w @ flip_mat
            
        volume.integrate(rgbd, o3d_intrinsic, curr_c2w)

    print("[Extract] Extracting Mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # 自动保存路径
    output_path = args.output if args.output else os.path.join(result_dir, "mesh_tsdf_fixed.ply")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"✓ Saved Mesh: {output_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")
    
    # 如果顶点太少，可能是坐标系修正错了（完全没重叠），或者深度图全是黑的
    if len(mesh.vertices) < 1000:
        print("[Warning] Vertex count is suspiciously low! Check the debug_frames folder.")
        print("  - If depth maps are black: Rendering failed.")
        print("  - If depth maps are good: Coordinate system might still be wrong.")
    
    # 同时也存个 OBJ 方便看
    obj_path = output_path.replace(".ply", ".obj")
    o3d.io.write_triangle_mesh(obj_path, mesh)
    print(f"✓ Saved OBJ to: {obj_path}")

if __name__ == "__main__":
    main()

