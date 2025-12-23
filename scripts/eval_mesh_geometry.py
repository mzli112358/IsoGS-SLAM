"""
Mesh Geometry Quality Evaluation Script

This script evaluates the geometric quality of a predicted mesh against a ground truth mesh
by computing various metrics including Accuracy, Completion, Completion Ratio, Chamfer Distance,
F-score, and Hausdorff Distance.
"""

import argparse
import os
import re
import sys
import csv
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Try to import pyrender for mesh rendering
try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("Warning: pyrender not available. Mesh rendering visualization will be disabled.")
    print("Install with: pip install pyrender")

# Try to import dataset loading utilities
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

try:
    from datasets.gradslam_datasets import load_dataset_config, ReplicaDataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("Warning: Dataset loading not available. Mesh rendering visualization will be disabled.")


def find_latest_mesh(mesh_dir):
    """
    自动查找最新的mesh文件（.ply格式）。
    优先查找 mesh_thickened_*.ply（按帧号排序），如果不存在则查找其他 .ply 文件。
    
    返回: (mesh_path, frame_number)
        - mesh_path: mesh文件路径
        - frame_number: 从文件名提取的帧号（如果无法提取则返回None）
    """
    if not os.path.exists(mesh_dir):
        raise FileNotFoundError(f"Mesh目录不存在: {mesh_dir}")
    
    # 优先查找 mesh_thickened_*.ply
    pattern = re.compile(r'^mesh_thickened_(\d+)\.ply$')
    mesh_files = []
    
    for filename in os.listdir(mesh_dir):
        match = pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            mesh_files.append((frame_num, filename))
    
    if mesh_files:
        mesh_files.sort(key=lambda x: x[0], reverse=True)
        latest_frame, latest_filename = mesh_files[0]
        mesh_path = os.path.join(mesh_dir, latest_filename)
        print(f"✓ 自动选择最新mesh文件: {latest_filename} (帧 {latest_frame})")
        if len(mesh_files) > 1:
            print(f"  (共找到 {len(mesh_files)} 个mesh文件)")
        return mesh_path, latest_frame
    
    # 如果没有找到 mesh_thickened_*.ply，查找其他 .ply 文件
    ply_files = [f for f in os.listdir(mesh_dir) if f.endswith('.ply')]
    if ply_files:
        # 按修改时间排序
        ply_files_with_time = []
        for f in ply_files:
            filepath = os.path.join(mesh_dir, f)
            mtime = os.path.getmtime(filepath)
            ply_files_with_time.append((mtime, f))
        ply_files_with_time.sort(reverse=True)
        latest_filename = ply_files_with_time[0][1]
        mesh_path = os.path.join(mesh_dir, latest_filename)
        print(f"✓ 自动选择最新mesh文件: {latest_filename}")
        return mesh_path, None
    
    raise FileNotFoundError(
        f"在 {mesh_dir} 中未找到mesh文件\n"
        f"期望的文件: mesh_thickened_*.ply 或 *.ply"
    )


def align_mesh_to_reference(pred_mesh, gt_mesh, method='center'):
    """
    将预测mesh对齐到GT mesh的坐标系。
    
    Args:
        pred_mesh: 预测mesh（trimesh.Trimesh）
        gt_mesh: GT mesh（trimesh.Trimesh，作为参考）
        method: 对齐方法
            - 'center': 中心对齐（平移使中心点重合）
            - 'icp': ICP对齐（需要open3d，更准确但更慢）
    
    Returns:
        aligned_pred_mesh: 对齐后的预测mesh
        transform_matrix: 4x4变换矩阵（如果method='icp'）
    """
    if method == 'center':
        # 中心对齐：将预测mesh的中心点移动到GT mesh的中心点
        pred_center = pred_mesh.centroid
        gt_center = gt_mesh.centroid
        translation = gt_center - pred_center
        
        # 创建变换矩阵
        transform = np.eye(4)
        transform[:3, 3] = translation
        
        # 应用变换
        aligned_pred_mesh = pred_mesh.copy()
        aligned_pred_mesh.apply_transform(transform)
        
        print(f"Mesh对齐（中心对齐）:")
        print(f"  预测mesh中心: {pred_center}")
        print(f"  GT mesh中心: {gt_center}")
        print(f"  平移量: {translation}")
        
        return aligned_pred_mesh, transform
    
    elif method == 'icp':
        # ICP对齐（需要open3d）
        try:
            import open3d as o3d
            
            # 采样点云
            pred_points = pred_mesh.sample(10000)
            gt_points = gt_mesh.sample(10000)
            
            # 转换为open3d点云
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
            
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
            
            # 执行ICP
            result = o3d.pipelines.registration.registration_icp(
                pred_pcd, gt_pcd, max_correspondence_distance=0.1,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            transform_matrix = result.transformation
            
            # 应用变换
            aligned_pred_mesh = pred_mesh.copy()
            aligned_pred_mesh.apply_transform(transform_matrix)
            
            print(f"Mesh对齐（ICP）:")
            print(f"  拟合度: {result.fitness:.4f}")
            print(f"  RMSE: {result.inlier_rmse:.4f}")
            
            return aligned_pred_mesh, transform_matrix
            
        except ImportError:
            print("警告: open3d未安装，无法使用ICP对齐，回退到中心对齐")
            return align_mesh_to_reference(pred_mesh, gt_mesh, method='center')
    
    else:
        raise ValueError(f"未知的对齐方法: {method}")


def get_mesh_stats(mesh):
    """
    获取mesh的统计信息。
    
    Args:
        mesh: trimesh.Trimesh object
    
    Returns:
        dict: 包含mesh统计信息的字典
    """
    stats = {
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'volume': float(mesh.volume) if mesh.is_volume else 0.0,
        'surface_area': float(mesh.area),
        'bounds_min': mesh.bounds[0].tolist(),
        'bounds_max': mesh.bounds[1].tolist(),
        'bounds_size': (mesh.bounds[1] - mesh.bounds[0]).tolist(),
        'is_watertight': mesh.is_watertight,
        'is_winding_consistent': mesh.is_winding_consistent,
    }
    return stats


def sample_points_from_mesh(mesh, num_samples=200000):
    """
    Sample points uniformly from a mesh surface.
    
    Args:
        mesh: trimesh.Trimesh object
        num_samples: number of points to sample
    
    Returns:
        points: numpy array of shape [num_samples, 3]
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return points


def compute_accuracy(pred_points, gt_points):
    """
    Compute Accuracy (L1): average distance from pred points to nearest gt points.
    
    Args:
        pred_points: numpy array of shape [N, 3]
        gt_points: numpy array of shape [M, 3]
    
    Returns:
        accuracy: float, average L1 distance
    """
    gt_tree = cKDTree(gt_points)
    distances, _ = gt_tree.query(pred_points, k=1)
    accuracy = np.mean(distances)
    return accuracy


def compute_completion(pred_points, gt_points):
    """
    Compute Completion (L1): average distance from gt points to nearest pred points.
    
    Args:
        pred_points: numpy array of shape [N, 3]
        gt_points: numpy array of shape [M, 3]
    
    Returns:
        completion: float, average L1 distance
    """
    pred_tree = cKDTree(pred_points)
    distances, _ = pred_tree.query(gt_points, k=1)
    completion = np.mean(distances)
    return completion


def compute_chamfer_distance(pred_points, gt_points):
    """
    Compute Chamfer Distance: average of accuracy and completion.
    
    Args:
        pred_points: numpy array of shape [N, 3]
        gt_points: numpy array of shape [M, 3]
    
    Returns:
        chamfer_distance: float
    """
    accuracy = compute_accuracy(pred_points, gt_points)
    completion = compute_completion(pred_points, gt_points)
    chamfer_distance = (accuracy + completion) / 2.0
    return chamfer_distance


def compute_f_score(pred_points, gt_points, threshold=0.05):
    """
    Compute F-score: harmonic mean of precision and recall.
    
    Args:
        pred_points: numpy array of shape [N, 3]
        gt_points: numpy array of shape [M, 3]
        threshold: distance threshold in meters
    
    Returns:
        f_score: float, F-score value
        precision: float, precision value
        recall: float, recall value
    """
    # Precision: percentage of pred points within threshold
    gt_tree = cKDTree(gt_points)
    pred_distances, _ = gt_tree.query(pred_points, k=1)
    precision = np.sum(pred_distances < threshold) / len(pred_points)
    
    # Recall: percentage of gt points within threshold
    pred_tree = cKDTree(pred_points)
    gt_distances, _ = pred_tree.query(gt_points, k=1)
    recall = np.sum(gt_distances < threshold) / len(gt_points)
    
    # F-score: harmonic mean
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0.0
    
    return f_score, precision, recall


def compute_hausdorff_distance(pred_points, gt_points, percentile=100):
    """
    Compute Hausdorff Distance (or percentile version).
    
    Args:
        pred_points: numpy array of shape [N, 3]
        gt_points: numpy array of shape [M, 3]
        percentile: percentile to use (100 = full Hausdorff, 95 = 95th percentile)
    
    Returns:
        hausdorff_distance: float
    """
    # Distance from pred to gt
    gt_tree = cKDTree(gt_points)
    pred_to_gt_distances, _ = gt_tree.query(pred_points, k=1)
    
    # Distance from gt to pred
    pred_tree = cKDTree(pred_points)
    gt_to_pred_distances, _ = pred_tree.query(gt_points, k=1)
    
    # Combined distances
    all_distances = np.concatenate([pred_to_gt_distances, gt_to_pred_distances])
    
    if percentile == 100:
        hausdorff_distance = np.max(all_distances)
    else:
        hausdorff_distance = np.percentile(all_distances, percentile)
    
    return hausdorff_distance


def compute_completion_ratio(pred_points, gt_points, threshold=0.05):
    """
    Compute Completion Ratio: proportion of gt points within threshold distance from pred.
    
    Args:
        pred_points: numpy array of shape [N, 3]
        gt_points: numpy array of shape [M, 3]
        threshold: distance threshold in meters (default: 0.05 = 5cm)
    
    Returns:
        completion_ratio: float, ratio between 0 and 1
    """
    pred_tree = cKDTree(pred_points)
    distances, _ = pred_tree.query(gt_points, k=1)
    within_threshold = np.sum(distances < threshold)
    completion_ratio = within_threshold / len(gt_points)
    return completion_ratio


def render_mesh_from_pose(mesh, pose, intrinsics, image_size=(1200, 680)):
    """
    从给定位姿渲染mesh的深度图。
    
    Args:
        mesh: trimesh.Trimesh object
        pose: 4x4 camera pose matrix (c2w, camera-to-world)
        intrinsics: 3x3 camera intrinsics matrix
        image_size: (width, height) tuple
    
    Returns:
        depth_image: numpy array of shape [H, W], 单位：米
    """
    if not PYRENDER_AVAILABLE:
        return None
    
    # Convert trimesh to pyrender mesh (无颜色，只渲染几何)
    scene = pyrender.Scene()
    
    # Create pyrender mesh (使用灰色材质，因为我们只需要深度)
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pyrender_mesh)
    
    # Setup camera
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    width, height = image_size
    
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    # pose是c2w，需要转换为w2c（world-to-camera）用于渲染
    w2c = np.linalg.inv(pose)
    camera_node = scene.add(camera, pose=w2c)
    
    # Setup renderer
    renderer = pyrender.OffscreenRenderer(width, height)
    
    # Render (获取RGB和深度，但我们只用深度)
    color, depth = renderer.render(scene)
    
    # Cleanup
    renderer.delete()
    
    return depth


def create_comparison_plot(pred_depth, gt_depth, frame_idx, save_path, depth_max=6.0):
    """
    创建深度图对比图。
    
    Args:
        pred_depth: 预测mesh渲染的深度图像 [H, W]，单位：米
        gt_depth: GT mesh渲染的深度图像 [H, W]，单位：米
        frame_idx: 帧索引
        save_path: 保存路径
        depth_max: 深度最大值（用于归一化显示）
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # 计算深度差异
    valid_mask = (pred_depth > 0) & (gt_depth > 0)
    depth_diff = np.zeros_like(pred_depth)
    if np.any(valid_mask):
        depth_diff[valid_mask] = np.abs(pred_depth[valid_mask] - gt_depth[valid_mask])
    
    # Row 1: GT Depth
    axs[0, 0].imshow(gt_depth, cmap='jet', vmin=0, vmax=depth_max)
    axs[0, 0].set_title("Ground Truth Mesh Depth")
    axs[0, 0].axis('off')
    
    # Row 1: Predicted Depth
    axs[0, 1].imshow(pred_depth, cmap='jet', vmin=0, vmax=depth_max)
    axs[0, 1].set_title("Predicted Mesh Depth")
    axs[0, 1].axis('off')
    
    # Row 2: Depth Difference
    if np.any(valid_mask):
        diff_max = np.percentile(depth_diff[valid_mask], 95)  # 使用95百分位数作为最大值
        if diff_max > 0:
            axs[1, 0].imshow(depth_diff, cmap='jet', vmin=0, vmax=diff_max)
        else:
            axs[1, 0].imshow(depth_diff, cmap='jet', vmin=0, vmax=0.1)
        axs[1, 0].set_title(f"Depth Difference (L1)")
    else:
        axs[1, 0].imshow(depth_diff, cmap='jet', vmin=0, vmax=0.1)
        axs[1, 0].set_title("Depth Difference (No Overlap)")
    axs[1, 0].axis('off')
    
    # Row 2: Side-by-side comparison
    # 创建并排对比图
    comparison = np.zeros((pred_depth.shape[0], pred_depth.shape[1] * 2))
    comparison[:, :pred_depth.shape[1]] = np.clip(gt_depth / depth_max, 0, 1)
    comparison[:, pred_depth.shape[1]:] = np.clip(pred_depth / depth_max, 0, 1)
    axs[1, 1].imshow(comparison, cmap='jet', vmin=0, vmax=1)
    axs[1, 1].set_title("GT (Left) vs Pred (Right)")
    axs[1, 1].axis('off')
    # 添加分割线
    axs[1, 1].axvline(x=pred_depth.shape[1], color='white', linewidth=2)
    
    fig.suptitle(f"Frame {frame_idx:04d} - Depth Comparison", fontsize=16)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mesh geometry quality against ground truth"
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Path to predicted mesh file (.ply). If not provided, will auto-find latest mesh in pred_dir."
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to ground truth mesh file (.ply)"
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default=None,
        help="Directory containing predicted mesh files. Used for auto-finding latest mesh if --pred is not provided."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Default: mesh_compare_eval in pred_dir."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200000,
        help="Number of points to sample from each mesh (default: 200000)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Distance threshold for completion ratio and F-score in meters (default: 0.05 = 5cm)"
    )
    parser.add_argument(
        "--render-views",
        action="store_true",
        help="Render mesh views from dataset camera poses for visualization"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Path to dataset config file (e.g., configs/data/replica.yaml). Required for --render-views."
    )
    parser.add_argument(
        "--dataset-basedir",
        type=str,
        default=None,
        help="Dataset base directory. Required for --render-views."
    )
    parser.add_argument(
        "--dataset-sequence",
        type=str,
        default=None,
        help="Dataset sequence name. Required for --render-views."
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=10,
        help="Render every Nth frame (default: 10)"
    )
    parser.add_argument(
        "--align-mesh",
        type=str,
        default='center',
        choices=['none', 'center', 'icp'],
        help="Mesh对齐方法: 'none'=不对齐, 'center'=中心对齐(默认), 'icp'=ICP对齐(需要open3d)"
    )
    
    args = parser.parse_args()
    
    # Determine predicted mesh path
    if args.pred is None:
        if args.pred_dir is None:
            raise ValueError("Either --pred or --pred-dir must be provided")
        pred_mesh_path, pred_frame = find_latest_mesh(args.pred_dir)
        pred_dir = args.pred_dir
    else:
        pred_mesh_path = args.pred
        pred_dir = os.path.dirname(pred_mesh_path)
        pred_frame = None
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(pred_dir, "mesh_compare_eval")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load meshes
    print(f"Loading predicted mesh from: {pred_mesh_path}")
    pred_mesh = trimesh.load(pred_mesh_path)
    if isinstance(pred_mesh, trimesh.Scene):
        pred_mesh = list(pred_mesh.geometry.values())[0]
    
    print(f"Loading ground truth mesh from: {args.gt}")
    gt_mesh = trimesh.load(args.gt)
    if isinstance(gt_mesh, trimesh.Scene):
        gt_mesh = list(gt_mesh.geometry.values())[0]
    
    # 对齐mesh（如果需要）
    if args.align_mesh != 'none':
        print(f"\n对齐mesh（方法: {args.align_mesh}）...")
        pred_mesh, transform_matrix = align_mesh_to_reference(pred_mesh, gt_mesh, method=args.align_mesh)
        print("✓ Mesh对齐完成")
    else:
        print("\n跳过mesh对齐（--align-mesh=none）")
        transform_matrix = None
    
    # Get mesh statistics
    print("\nComputing mesh statistics...")
    pred_stats = get_mesh_stats(pred_mesh)
    gt_stats = get_mesh_stats(gt_mesh)
    
    # Sample points from meshes
    print(f"\nSampling {args.num_samples} points from predicted mesh...")
    pred_points = sample_points_from_mesh(pred_mesh, args.num_samples)
    
    print(f"Sampling {args.num_samples} points from ground truth mesh...")
    gt_points = sample_points_from_mesh(gt_mesh, args.num_samples)
    
    # Compute metrics
    print("\nComputing metrics...")
    print("Computing Accuracy (L1)...")
    accuracy = compute_accuracy(pred_points, gt_points)
    
    print("Computing Completion (L1)...")
    completion = compute_completion(pred_points, gt_points)
    
    print("Computing Chamfer Distance...")
    chamfer_distance = compute_chamfer_distance(pred_points, gt_points)
    
    print(f"Computing Completion Ratio (threshold={args.threshold*100:.1f}cm)...")
    completion_ratio = compute_completion_ratio(pred_points, gt_points, args.threshold)
    
    print(f"Computing F-score (threshold={args.threshold*100:.1f}cm)...")
    f_score, precision, recall = compute_f_score(pred_points, gt_points, args.threshold)
    
    print("Computing Hausdorff Distance...")
    hausdorff_distance = compute_hausdorff_distance(pred_points, gt_points, percentile=100)
    
    print("Computing 95th percentile Hausdorff Distance...")
    hausdorff_95 = compute_hausdorff_distance(pred_points, gt_points, percentile=95)
    
    # Prepare results dictionary
    results = {
        'accuracy_l1_cm': accuracy * 100,
        'completion_l1_cm': completion * 100,
        'chamfer_distance_cm': chamfer_distance * 100,
        'completion_ratio_percent': completion_ratio * 100,
        'f_score': f_score,
        'precision': precision,
        'recall': recall,
        'hausdorff_distance_cm': hausdorff_distance * 100,
        'hausdorff_95_cm': hausdorff_95 * 100,
    }
    
    # Add mesh statistics
    results.update({
        'pred_num_vertices': pred_stats['num_vertices'],
        'pred_num_faces': pred_stats['num_faces'],
        'pred_volume': pred_stats['volume'],
        'pred_surface_area': pred_stats['surface_area'],
        'pred_is_watertight': pred_stats['is_watertight'],
        'gt_num_vertices': gt_stats['num_vertices'],
        'gt_num_faces': gt_stats['num_faces'],
        'gt_volume': gt_stats['volume'],
        'gt_surface_area': gt_stats['surface_area'],
        'gt_is_watertight': gt_stats['is_watertight'],
    })
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "mesh_eval_results.csv")
    print(f"\nSaving results to: {csv_path}")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    # Save detailed results to text file
    txt_path = os.path.join(output_dir, "mesh_eval_results.txt")
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Mesh Geometry Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("Predicted Mesh Statistics:\n")
        f.write(f"  Vertices: {pred_stats['num_vertices']:,}\n")
        f.write(f"  Faces: {pred_stats['num_faces']:,}\n")
        f.write(f"  Volume: {pred_stats['volume']:.6f}\n")
        f.write(f"  Surface Area: {pred_stats['surface_area']:.6f}\n")
        f.write(f"  Bounds: {pred_stats['bounds_min']} to {pred_stats['bounds_max']}\n")
        f.write(f"  Is Watertight: {pred_stats['is_watertight']}\n")
        f.write(f"  Is Winding Consistent: {pred_stats['is_winding_consistent']}\n\n")
        
        f.write("Ground Truth Mesh Statistics:\n")
        f.write(f"  Vertices: {gt_stats['num_vertices']:,}\n")
        f.write(f"  Faces: {gt_stats['num_faces']:,}\n")
        f.write(f"  Volume: {gt_stats['volume']:.6f}\n")
        f.write(f"  Surface Area: {gt_stats['surface_area']:.6f}\n")
        f.write(f"  Bounds: {gt_stats['bounds_min']} to {gt_stats['bounds_max']}\n")
        f.write(f"  Is Watertight: {gt_stats['is_watertight']}\n")
        f.write(f"  Is Winding Consistent: {gt_stats['is_winding_consistent']}\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write(f"  Accuracy (L1):        {accuracy*100:.4f} cm\n")
        f.write(f"  Completion (L1):      {completion*100:.4f} cm\n")
        f.write(f"  Chamfer Distance:     {chamfer_distance*100:.4f} cm\n")
        f.write(f"  Completion Ratio:     {completion_ratio*100:.2f}%\n")
        f.write(f"  F-score:              {f_score:.4f}\n")
        f.write(f"  Precision:            {precision:.4f}\n")
        f.write(f"  Recall:               {recall:.4f}\n")
        f.write(f"  Hausdorff Distance:   {hausdorff_distance*100:.4f} cm\n")
        f.write(f"  Hausdorff (95th %):   {hausdorff_95*100:.4f} cm\n")
        f.write("="*60 + "\n")
    
    # Print results
    print("\n" + "="*60)
    print("Mesh Geometry Evaluation Results")
    print("="*60)
    print(f"Accuracy (L1):        {accuracy*100:.4f} cm")
    print(f"Completion (L1):      {completion*100:.4f} cm")
    print(f"Chamfer Distance:     {chamfer_distance*100:.4f} cm")
    print(f"Completion Ratio:     {completion_ratio*100:.2f}%")
    print(f"F-score:              {f_score:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"Hausdorff Distance:   {hausdorff_distance*100:.4f} cm")
    print(f"Hausdorff (95th %):   {hausdorff_95*100:.4f} cm")
    print("="*60)
    
    # Render views if requested
    if args.render_views:
        if not PYRENDER_AVAILABLE:
            print("\nWarning: pyrender not available. Skipping mesh rendering.")
        elif not DATASET_AVAILABLE:
            print("\nWarning: Dataset loading not available. Skipping mesh rendering.")
        else:
            if args.dataset_config is None or args.dataset_basedir is None or args.dataset_sequence is None:
                print("\nWarning: Dataset config, basedir, or sequence not provided. Skipping mesh rendering.")
            else:
                print("\nRendering mesh views from dataset camera poses...")
                
                # Load dataset
                dataset_config = load_dataset_config(args.dataset_config)
                dataset = ReplicaDataset(
                    config_dict=dataset_config,
                    basedir=args.dataset_basedir,
                    sequence=args.dataset_sequence,
                    desired_height=680,
                    desired_width=1200,
                    device="cpu",
                )
                
                # Create output directories
                view_dir = os.path.join(output_dir, "view")
                rendered_view_dir = os.path.join(output_dir, "rendered_view")
                compare_view_plots_dir = os.path.join(output_dir, "compare_view_plots")
                os.makedirs(view_dir, exist_ok=True)
                os.makedirs(rendered_view_dir, exist_ok=True)
                os.makedirs(compare_view_plots_dir, exist_ok=True)
                
                # Get camera intrinsics
                fx = dataset_config["camera_params"]["fx"]
                fy = dataset_config["camera_params"]["fy"]
                cx = dataset_config["camera_params"]["cx"]
                cy = dataset_config["camera_params"]["cy"]
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                
                # 确定要渲染的帧数：只渲染训练帧数
                dataset_total_frames = len(dataset)
                if pred_frame is not None:
                    # 从mesh文件名提取的训练帧数
                    trained_frames = pred_frame
                    print(f"数据集总帧数: {dataset_total_frames}")
                    print(f"训练帧数（从mesh文件名提取）: {trained_frames}")
                    print(f"只渲染前 {trained_frames} 帧（训练时见过的帧）")
                    max_frame = min(trained_frames, dataset_total_frames)
                else:
                    # 如果无法从文件名提取，使用数据集总帧数（但会警告）
                    print(f"警告: 无法从mesh文件名提取训练帧数")
                    print(f"将渲染所有 {dataset_total_frames} 帧（可能包含训练时未见的帧）")
                    max_frame = dataset_total_frames
                
                # 渲染帧序列
                frames_to_render = list(range(0, max_frame, args.render_every))
                print(f"将渲染 {len(frames_to_render)} 帧（每 {args.render_every} 帧渲染一次）")
                
                for frame_idx in tqdm(frames_to_render, desc="Rendering mesh views"):
                    # 从数据集获取相机位姿
                    # 位姿读取机制：
                    # 1. 位姿文件位置: data/Replica/{scene_name}/traj.txt
                    # 2. 每行是一个4x4的c2w矩阵（camera-to-world transformation）
                    # 3. dataset[frame_idx]返回的pose就是c2w矩阵
                    # 4. 预测mesh和GT mesh使用相同的位姿，确保视角一致
                    color, depth, intrinsics_tensor, pose = dataset[frame_idx]
                    pose_np = pose.numpy()  # 这是c2w矩阵（camera-to-world）
                    
                    # Render predicted mesh depth
                    pred_depth = render_mesh_from_pose(
                        pred_mesh, pose_np, intrinsics, image_size=(1200, 680)
                    )
                    
                    # Render GT mesh depth
                    gt_depth = render_mesh_from_pose(
                        gt_mesh, pose_np, intrinsics, image_size=(1200, 680)
                    )
                    
                    if pred_depth is not None and gt_depth is not None:
                        # 保存深度图（归一化到0-255用于可视化）
                        # 预测mesh深度图
                        pred_depth_viz = np.clip((pred_depth / 6.0) * 255, 0, 255).astype(np.uint8)
                        pred_depth_colormap = cv2.applyColorMap(pred_depth_viz, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(view_dir, f"frame_{frame_idx:04d}.png"),
                            pred_depth_colormap
                        )
                        
                        # GT mesh深度图
                        gt_depth_viz = np.clip((gt_depth / 6.0) * 255, 0, 255).astype(np.uint8)
                        gt_depth_colormap = cv2.applyColorMap(gt_depth_viz, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(rendered_view_dir, f"frame_{frame_idx:04d}.png"),
                            gt_depth_colormap
                        )
                        
                        # Create and save comparison plot
                        create_comparison_plot(
                            pred_depth, gt_depth,
                            frame_idx, os.path.join(compare_view_plots_dir, f"frame_{frame_idx:04d}.png")
                        )
                
                print(f"\nRendered views saved to:")
                print(f"  - {view_dir}")
                print(f"  - {rendered_view_dir}")
                print(f"  - {compare_view_plots_dir}")


if __name__ == "__main__":
    main()
