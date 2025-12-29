"""
Fast Mesh Extraction from IsoGS checkpoint using Tile-based/Block-based algorithm.

This script uses a push-based approach: instead of querying all Gaussians for each voxel,
it assigns Gaussians to spatial blocks and only computes density for relevant Gaussians.
"""

import argparse
import os
import re
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage import measure
import trimesh

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.slam_external import build_rotation


def parse_args():
    parser = argparse.ArgumentParser(description="Fast mesh extraction from IsoGS checkpoint")
    parser.add_argument("config", type=str, help="Path to experiment config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Specific checkpoint file. If None, auto-selects latest.")
    parser.add_argument("--output", type=str, default=None,
                       help="Output mesh file path (default: mesh_fast.ply)")
    parser.add_argument("--voxel-size", type=float, default=0.02,
                       help="Voxel size in meters (default: 0.02)")
    parser.add_argument("--iso-level", type=float, default=1.0,
                       help="Iso-surface threshold level (default: 1.0)")
    parser.add_argument("--padding", type=float, default=0.5,
                       help="Padding around bounding box in meters (default: 0.5)")
    parser.add_argument("--block-size", type=int, default=64,
                       help="Block size for tiling (default: 64, meaning 64x64x64 voxels per block)")
    parser.add_argument("--truncate-sigma", type=float, default=3.0,
                       help="Truncation distance in sigma (default: 3.0)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--no-cleaning", action="store_true",
                       help="Disable mesh cleaning (keep full mesh instead of only largest component)")
    parser.add_argument("--no-show", action="store_true",
                       help="Do not open 3D viewer after mesh export")
    return parser.parse_args()


def load_checkpoint(config_path, checkpoint_path=None):
    """Load experiment config and checkpoint parameters."""
    # Load config
    experiment = SourceFileLoader(
        os.path.basename(config_path), config_path
    ).load_module()
    config = experiment.config
    
    # Determine checkpoint path
    result_dir = os.path.join(config['workdir'], config['run_name'])
    
    checkpoint_frame = None
    if checkpoint_path is None:
        # Smart checkpoint selection
        params_npz_path = os.path.join(result_dir, "params.npz")
        if os.path.exists(params_npz_path):
            checkpoint_path = params_npz_path
            print(f"✓ Found final params file: {checkpoint_path}")
        else:
            # Find all checkpoint files
            pattern = re.compile(r'^params(\d+)\.npz$')
            checkpoint_files = []
            if os.path.exists(result_dir):
                for filename in os.listdir(result_dir):
                    match = pattern.match(filename)
                    if match:
                        checkpoint_num = int(match.group(1))
                        checkpoint_files.append((checkpoint_num, filename))
            
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: x[0], reverse=True)
                latest_checkpoint = checkpoint_files[0]
                checkpoint_frame = latest_checkpoint[0]
                checkpoint_path = os.path.join(result_dir, latest_checkpoint[1])
                print(f"✓ Auto-selected latest checkpoint: {latest_checkpoint[1]} (frame {latest_checkpoint[0]})")
                if len(checkpoint_files) > 1:
                    print(f"  (Found {len(checkpoint_files)} checkpoints total)")
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {result_dir}\n"
                    f"Expected files: params.npz or params*.npz"
                )
    else:
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(result_dir, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"✓ Using specified checkpoint: {checkpoint_path}")
        # Try to parse frame number from filename (e.g., params800.npz)
        basename = os.path.basename(checkpoint_path)
        m = re.match(r'^params(\d+)\.npz$', basename)
        if m:
            checkpoint_frame = int(m.group(1))
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    params = dict(np.load(checkpoint_path, allow_pickle=True))
    
    return config, params, result_dir, checkpoint_path, checkpoint_frame


def build_inverse_covariances(params, device, min_scale_limit=0.0):
    """Build inverse covariance matrices for all Gaussians.
    
    Args:
        params: Dictionary containing Gaussian parameters
        device: Device to use for computation
        min_scale_limit: Minimum scale limit to prevent pancaking artifacts (default: 0.0, disabled)
    """
    # Get parameters
    means = params['means3D']
    log_scales = params['log_scales']
    unnorm_rots = params['unnorm_rotations']
    
    # Convert to tensors on device
    if not isinstance(means, torch.Tensor):
        means = torch.tensor(means, device=device, dtype=torch.float32)
    else:
        means = means.to(device).float()
    
    if not isinstance(log_scales, torch.Tensor):
        log_scales = torch.tensor(log_scales, device=device, dtype=torch.float32)
    else:
        log_scales = log_scales.to(device).float()
    
    if not isinstance(unnorm_rots, torch.Tensor):
        unnorm_rots = torch.tensor(unnorm_rots, device=device, dtype=torch.float32)
    else:
        unnorm_rots = unnorm_rots.to(device).float()
    
    # Handle isotropic case
    if log_scales.shape[1] == 1:
        log_scales = log_scales.repeat(1, 3)
    
    # Get scales and opacities
    scales = torch.exp(log_scales).clamp(min=1e-5)  # [N, 3]
    
    # Apply minimum scale limit to prevent pancaking artifacts
    # This ensures each Gaussian covers at least one sampling point
    if min_scale_limit > 0:
        print(f"Clamping scales to minimum: {min_scale_limit}")
        scales = torch.clamp(scales, min=min_scale_limit)
    
    # Handle opacities
    logit_opacities = params['logit_opacities']
    if not isinstance(logit_opacities, torch.Tensor):
        logit_opacities = torch.tensor(logit_opacities, device=device, dtype=torch.float32)
    else:
        logit_opacities = logit_opacities.to(device).float()
    opacities = torch.sigmoid(logit_opacities).squeeze(-1)  # [N]
    
    # Build rotation matrices
    quats = F.normalize(unnorm_rots, dim=1)  # [N, 4]
    R = build_rotation(quats)  # [N, 3, 3]
    
    # Build inverse covariance matrices: Σ^{-1} = R S^{-2} R^T
    S_inv_sq = 1.0 / (scales ** 2 + 1e-8)  # [N, 3]
    S_inv_sq_diag = torch.diag_embed(S_inv_sq)  # [N, 3, 3]
    R_S_inv_sq = torch.bmm(R, S_inv_sq_diag)  # [N, 3, 3]
    inverse_covariances = torch.bmm(R_S_inv_sq, R.transpose(1, 2))  # [N, 3, 3]
    
    print(f"Loaded {len(means)} Gaussians")
    print(f"  Means shape: {means.shape}")
    print(f"  Scales shape: {scales.shape}")
    print(f"  Opacities range: [{opacities.min().item():.4f}, {opacities.max().item():.4f}]")
    
    return means, inverse_covariances, opacities, scales


def compute_bounding_box(means, padding=0.5):
    """Compute bounding box from Gaussian means."""
    means_np = means.cpu().numpy() if isinstance(means, torch.Tensor) else means
    min_bounds = means_np.min(axis=0) - padding
    max_bounds = means_np.max(axis=0) + padding
    print(f"Bounding box: min={min_bounds}, max={max_bounds}")
    return min_bounds, max_bounds


def generate_block_voxel_coords(x_start, x_end, y_start, y_end, z_start, z_end,
                                  min_bounds, dims, voxel_size, device='cuda'):
    """Dynamically generate voxel coordinates for a specific block.
    
    Args:
        x_start, x_end, y_start, y_end, z_start, z_end: Block voxel indices
        min_bounds: Minimum bounds of the voxel grid in world space (numpy array or torch tensor)
        dims: Grid dimensions [Dx, Dy, Dz]
        voxel_size: Voxel size in meters (or array of 3 if non-uniform)
        device: Device to generate coordinates on
    
    Returns:
        block_voxel_coords: [num_block_voxels, 3] tensor of world coordinates
    """
    # Convert min_bounds to torch tensor if needed
    if isinstance(min_bounds, np.ndarray):
        min_bounds = torch.tensor(min_bounds, device=device, dtype=torch.float32)
    elif not isinstance(min_bounds, torch.Tensor):
        min_bounds = torch.tensor(np.asarray(min_bounds), device=device, dtype=torch.float32)
    else:
        min_bounds = min_bounds.to(device).float()
    
    # Calculate actual voxel size (handle scalar or array)
    if np.isscalar(voxel_size):
        voxel_sizes = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_sizes = np.asarray(voxel_size)
    
    # Convert voxel_sizes to torch tensor
    voxel_sizes = torch.tensor(voxel_sizes, device=device, dtype=torch.float32)
    
    # Create coordinate ranges for this block (indices as float for multiplication)
    x_indices = torch.arange(x_start, x_end, device=device, dtype=torch.float32)
    y_indices = torch.arange(y_start, y_end, device=device, dtype=torch.float32)
    z_indices = torch.arange(z_start, z_end, device=device, dtype=torch.float32)
    
    # Convert to world coordinates
    x_coords = min_bounds[0] + x_indices * voxel_sizes[0]
    y_coords = min_bounds[1] + y_indices * voxel_sizes[1]
    z_coords = min_bounds[2] + z_indices * voxel_sizes[2]
    
    # Create meshgrid
    X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Flatten and stack
    block_voxel_coords = torch.stack([
        X.flatten(),
        Y.flatten(),
        Z.flatten()
    ], dim=1)  # [num_block_voxels, 3]
    
    return block_voxel_coords


def compute_density_tiled(dims, means, inverse_covariances, opacities, scales,
                          min_bounds, max_bounds, voxel_size, block_size=64, 
                          truncate_sigma=3.0, device='cuda'):
    """
    Compute density using CPU-GPU streaming with 3D Tight Culling + Aggressive VRAM strategy.
    
    Strategy:
    1. Pre-sort all Gaussians by X coordinate for efficient binary search culling
    2. Keep global density grid on CPU (never allocate on GPU)
    3. Process 3D blocks (bx, by, bz loops) - tight spatial culling
    4. For each X block: binary search to get X candidates
    5. For each Y block: filter X candidates by Y bounds to get XY candidates
    6. For each Z block: filter XY candidates by Z bounds to get XYZ candidates
    7. Layered filtering: All -> X -> XY -> XYZ (extremely tight culling)
    8. Aggressive VRAM usage: large batch sizes to maximize GPU utilization
    9. Clear GPU cache only at end of each X block
    
    This approach maximizes culling efficiency and GPU utilization for large scenes.
    """
    num_voxels = int(np.prod(dims))
    num_gaussians = len(means)
    
    # Calculate actual voxel size (handle scalar or array)
    if np.isscalar(voxel_size):
        voxel_sizes = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_sizes = np.asarray(voxel_size)
    
    # Initialize global density array on CPU (float32 for precision)
    densities = np.zeros(tuple(dims), dtype=np.float32)
    
    # Move Gaussian parameters to GPU (assuming they fit)
    # For very large point clouds, this might need further chunking
    means_dev = means.to(device)
    inv_covs_dev = inverse_covariances.to(device)
    opacities_dev = opacities.to(device)
    scales_dev = scales.to(device)
    
    # Pre-compute max influence radius for each Gaussian (for AABB culling)
    max_scales = scales_dev.max(dim=1).values  # [N]
    max_radii = truncate_sigma * max_scales  # [N]
    max_radius_global = max_radii.max().item()
    
    print(f"Pre-sorting {num_gaussians:,} Gaussians by X coordinate...")
    
    # ========================================================================
    # STEP 1: Pre-sort all Gaussians by X coordinate (Sort-and-Sweep)
    # ========================================================================
    # Get X coordinates
    x_coords = means_dev[:, 0]  # [N]
    
    # Sort by X coordinate
    sorted_indices = torch.argsort(x_coords)  # [N]
    
    # Reorder all parameters according to sorted indices
    means_dev = means_dev[sorted_indices]  # [N, 3]
    inv_covs_dev = inv_covs_dev[sorted_indices]  # [N, 3, 3]
    opacities_dev = opacities_dev[sorted_indices]  # [N]
    scales_dev = scales_dev[sorted_indices]  # [N, 3]
    max_radii = max_radii[sorted_indices]  # [N]
    # Ensure x_coords is contiguous for torch.searchsorted (fixes non-contiguous warning)
    x_coords = means_dev[:, 0].contiguous()  # Update x_coords to sorted version [N]
    
    print(f"✓ Pre-sorting complete. Gaussians are now X-axis sorted.")
    
    # Calculate block dimensions
    block_dims = np.ceil(np.array(dims) / block_size).astype(int)
    num_blocks = int(np.prod(block_dims))
    
    print(f"Grid dimensions: {dims}")
    print(f"Block dimensions: {block_dims} (block_size={block_size})")
    print(f"Total blocks: {num_blocks}")
    print(f"Processing using 3D Tight Culling + Aggressive VRAM strategy...")
    print(f"  - Global density grid: CPU ({densities.nbytes / 1e9:.2f} GB)")
    print(f"  - Gaussian parameters: GPU (sorted by X)")
    print(f"  - Max truncation radius: {max_radius_global:.4f}")
    print(f"  - Dynamic voxel batch size (max 1,000,000, aggressive VRAM usage)")
    print(f"  - 3D layered culling: X -> XY -> XYZ")
    
    # Aggressive VRAM usage: Maximum tensor elements (50 billion = ~20GB for float32)
    # For systems with 48GB+ VRAM, this allows processing much larger blocks efficiently
    MAX_TENSOR_ELEMENTS = 5_000_000_000  # 50 billion elements ≈ 20GB
    
    # Process blocks with progress bar
    pbar = tqdm(total=num_blocks, desc="Processing 3D blocks")
    
    block_idx = 0
    with torch.no_grad():  # Disable gradient computation to save memory
        for bx in range(block_dims[0]):
            # ====================================================================
            # STEP 2: Binary search on X-axis to find candidate Gaussians
            # ====================================================================
            # Calculate current X-axis block world space bounds
            x_start = bx * block_size
            x_end = min((bx + 1) * block_size, dims[0])
            block_x_min_world = min_bounds[0] + x_start * voxel_sizes[0]
            block_x_max_world = min_bounds[0] + x_end * voxel_sizes[0]
            
            # Expand X bounds by max truncation radius
            search_x_min = block_x_min_world - max_radius_global * 1.5
            search_x_max = block_x_max_world + max_radius_global * 1.5
            
            # Binary search to find candidate range [idx_start, idx_end)
            search_x_min_t = torch.tensor(search_x_min, device=device, dtype=torch.float32)
            search_x_max_t = torch.tensor(search_x_max, device=device, dtype=torch.float32)
            idx_start = torch.searchsorted(x_coords, search_x_min_t, right=False)
            idx_end = torch.searchsorted(x_coords, search_x_max_t, right=True)
            
            # Clamp to valid range
            idx_start = max(0, idx_start.item())
            idx_end = min(num_gaussians, idx_end.item())
            x_candidate_count = idx_end - idx_start
            
            if x_candidate_count == 0:
                # No Gaussians in X range, skip all Y/Z blocks for this X block
                skip_count = block_dims[1] * block_dims[2]
                pbar.update(skip_count)
                block_idx += skip_count
                continue
            
            # Extract X-axis candidate slice
            x_candidate_means = means_dev[idx_start:idx_end]  # [X_candidates, 3]
            x_candidate_inv_covs = inv_covs_dev[idx_start:idx_end]  # [X_candidates, 3, 3]
            x_candidate_opacities = opacities_dev[idx_start:idx_end]  # [X_candidates]
            x_candidate_max_radii = max_radii[idx_start:idx_end]  # [X_candidates]
            
            # Process Y blocks for this X block
            for by in range(block_dims[1]):
                # ====================================================================
                # STEP 3: Y-axis filtering to get XY candidates
                # ====================================================================
                # Calculate Y-axis block bounds
                y_start = by * block_size
                y_end = min((by + 1) * block_size, dims[1])
                
                # Calculate Y-axis world space bounds
                block_y_min_world = min_bounds[1] + y_start * voxel_sizes[1]
                block_y_max_world = min_bounds[1] + y_end * voxel_sizes[1]
                
                # Expand Y bounds by max truncation radius
                search_y_min = block_y_min_world - max_radius_global * 1.5
                search_y_max = block_y_max_world + max_radius_global * 1.5
                
                # Filter X candidates by Y bounds
                search_y_min_t = torch.tensor(search_y_min, device=device, dtype=torch.float32)
                search_y_max_t = torch.tensor(search_y_max, device=device, dtype=torch.float32)
                in_y_bounds = (
                    (x_candidate_means[:, 1] >= search_y_min_t) &
                    (x_candidate_means[:, 1] <= search_y_max_t)
                )
                xy_candidate_indices = torch.where(in_y_bounds)[0]
                
                if len(xy_candidate_indices) == 0:
                    # No Gaussians in XY range, skip all Z blocks for this Y block
                    skip_count = block_dims[2]
                    pbar.update(skip_count)
                    block_idx += skip_count
                    continue
                
                # Extract XY candidates
                xy_candidate_means = x_candidate_means[xy_candidate_indices]  # [XY_candidates, 3]
                xy_candidate_inv_covs = x_candidate_inv_covs[xy_candidate_indices]  # [XY_candidates, 3, 3]
                xy_candidate_opacities = x_candidate_opacities[xy_candidate_indices]  # [XY_candidates]
                xy_candidate_max_radii = x_candidate_max_radii[xy_candidate_indices]  # [XY_candidates]
                
                # Process Z blocks for this XY block
                for bz in range(block_dims[2]):
                    # ====================================================================
                    # STEP 4: Z-axis filtering to get XYZ candidates (final tight culling)
                    # ====================================================================
                    # Calculate Z-axis block bounds
                    z_start = bz * block_size
                    z_end = min((bz + 1) * block_size, dims[2])
                    
                    # Calculate Z-axis world space bounds
                    block_z_min_world = min_bounds[2] + z_start * voxel_sizes[2]
                    block_z_max_world = min_bounds[2] + z_end * voxel_sizes[2]
                    
                    # Expand Z bounds by max truncation radius
                    search_z_min = block_z_min_world - max_radius_global * 1.5
                    search_z_max = block_z_max_world + max_radius_global * 1.5
                    
                    # Filter XY candidates by Z bounds
                    search_z_min_t = torch.tensor(search_z_min, device=device, dtype=torch.float32)
                    search_z_max_t = torch.tensor(search_z_max, device=device, dtype=torch.float32)
                    in_z_bounds = (
                        (xy_candidate_means[:, 2] >= search_z_min_t) &
                        (xy_candidate_means[:, 2] <= search_z_max_t)
                    )
                    xyz_candidate_indices = torch.where(in_z_bounds)[0]
                    
                    if len(xyz_candidate_indices) == 0:
                        # No Gaussians in XYZ range, density is zero (already zero in CPU array)
                        pbar.update(1)
                        block_idx += 1
                        continue
                    
                    # Extract XYZ candidates (final candidates - extremely tight!)
                    relevant_means = xy_candidate_means[xyz_candidate_indices]  # [K, 3]
                    relevant_inv_covs = xy_candidate_inv_covs[xyz_candidate_indices]  # [K, 3, 3]
                    relevant_opacities = xy_candidate_opacities[xyz_candidate_indices]  # [K]
                    relevant_max_radii = xy_candidate_max_radii[xyz_candidate_indices]  # [K]
                    num_relevant_gaussians = len(xyz_candidate_indices)  # K
                    
                    # ====================================================================
                    # STEP 5: Generate block voxel coordinates
                    # ====================================================================
                    block_voxel_coords = generate_block_voxel_coords(
                        x_start, x_end, y_start, y_end, z_start, z_end,
                        min_bounds, dims, voxel_sizes, device=device
                    )  # [num_block_voxels, 3]
                    
                    num_block_voxels = block_voxel_coords.shape[0]
                    
                    # ====================================================================
                    # STEP 6: Dynamic Batch Size Calculation (Aggressive VRAM)
                    # ====================================================================
                    if num_relevant_gaussians > 0:
                        # Calculate safe batch size based on K
                        # Factor of 10 accounts for all intermediate tensors
                        max_safe_batch = max(1, int(MAX_TENSOR_ELEMENTS / (num_relevant_gaussians * 10)))
                        # Aggressive limit: 1 million voxels per batch (for 48GB+ VRAM)
                        max_batch_limit = 1_000_000
                        voxel_batch_size = min(max_safe_batch, max_batch_limit, num_block_voxels)
                    else:
                        voxel_batch_size = num_block_voxels
                    
                    # Initialize block densities
                    block_densities = torch.zeros(num_block_voxels, device=device, dtype=torch.float32)
                    
                    # Process block voxels in batches (may process entire block at once if K is small)
                    for i in range(0, num_block_voxels, voxel_batch_size):
                        batch_end = min(i + voxel_batch_size, num_block_voxels)
                        batch_voxels = block_voxel_coords[i:batch_end]  # [batch_size, 3]
                        batch_size = batch_voxels.shape[0]
                        
                        # Compute deltas: [batch_size, K, 3]
                        deltas = batch_voxels.unsqueeze(1) - relevant_means.unsqueeze(0)
                        
                        # Compute distances for truncation
                        dists = torch.norm(deltas, dim=2)  # [batch_size, K]
                        truncate_mask = dists < relevant_max_radii.unsqueeze(0)  # [batch_size, K]
                        
                        # Compute quadratic form: delta^T @ inv_cov @ delta
                        deltas_flat = deltas.reshape(-1, 3)  # [batch_size*K, 3]
                        
                        # Expand inverse covariances: [K, 3, 3] -> [batch_size*K, 3, 3]
                        inv_covs_expanded = relevant_inv_covs.unsqueeze(0).expand(batch_size, -1, -1, -1)
                        inv_covs_flat = inv_covs_expanded.reshape(-1, 3, 3)
                        
                        # Compute quadratic form
                        deltas_expanded = deltas_flat.unsqueeze(1)  # [batch_size*K, 1, 3]
                        inv_cov_delta = torch.bmm(deltas_expanded, inv_covs_flat)  # [batch_size*K, 1, 3]
                        quad_form_flat = torch.bmm(inv_cov_delta, deltas_flat.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                        
                        # Reshape back: [batch_size, K]
                        quad_form = quad_form_flat.reshape(batch_size, num_relevant_gaussians)
                        
                        # Compute exponential
                        exp_term = torch.exp(-0.5 * quad_form)  # [batch_size, K]
                        
                        # Multiply by opacities and apply truncation
                        opacities_expanded = relevant_opacities.unsqueeze(0).expand(batch_size, -1)
                        density_contrib = opacities_expanded * exp_term * truncate_mask.float()
                        
                        # Sum over Gaussians
                        batch_densities = density_contrib.sum(dim=1)  # [batch_size]
                        block_densities[i:batch_end] = batch_densities
                        
                        # Clear intermediate tensors after each batch
                        del batch_voxels, deltas, dists, deltas_flat, inv_covs_expanded, inv_covs_flat
                        del deltas_expanded, inv_cov_delta, quad_form_flat, quad_form, exp_term
                        del opacities_expanded, density_contrib, batch_densities
                    
                    # Move block densities to CPU and update global array
                    block_densities_cpu = block_densities.cpu().numpy().reshape(
                        (x_end - x_start, y_end - y_start, z_end - z_start)
                    )
                    densities[x_start:x_end, y_start:y_end, z_start:z_end] = block_densities_cpu
                    
                    # Clear GPU memory (no cache clearing here - only at X block end)
                    del block_voxel_coords, block_densities, block_densities_cpu
                    del relevant_means, relevant_inv_covs, relevant_opacities, relevant_max_radii
                    
                    pbar.update(1)
                    block_idx += 1
                
                # Clear XY candidate tensors after processing all Z blocks for this Y block
                del xy_candidate_means, xy_candidate_inv_covs, xy_candidate_opacities, xy_candidate_max_radii
            
            # Clear X candidate tensors after processing all Y/Z blocks for this X block
            del x_candidate_means, x_candidate_inv_covs, x_candidate_opacities, x_candidate_max_radii
            # Clear GPU cache only once per X block - critical for performance
            torch.cuda.empty_cache()
    
    pbar.close()
    
    # Return flattened array to match original interface
    return densities.flatten()


def create_voxel_grid(min_bounds, max_bounds, voxel_size):
    """Calculate voxel grid dimensions without creating coordinates.
    
    Returns only dimensions and spacing information.
    Coordinates are generated dynamically per-block to save memory.
    """
    # Calculate grid dimensions
    size = max_bounds - min_bounds
    dims = np.ceil(size / voxel_size).astype(int)
    print(f"Voxel grid dimensions: {dims} (voxel_size={voxel_size})")
    print(f"Total voxels: {np.prod(dims):,}")
    
    # Create coordinate ranges for reference
    x = np.linspace(min_bounds[0], max_bounds[0], dims[0])
    y = np.linspace(min_bounds[1], max_bounds[1], dims[1])
    z = np.linspace(min_bounds[2], max_bounds[2], dims[2])
    
    # Calculate actual voxel size
    actual_voxel_size = np.array([
        (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else voxel_size,
        (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else voxel_size,
        (z[-1] - z[0]) / (len(z) - 1) if len(z) > 1 else voxel_size
    ])
    
    # Don't create full coordinate grid - will be generated per-block
    return dims, actual_voxel_size, min_bounds


def extract_mesh(density_grid, dims, iso_level=1.0, voxel_spacing=None, origin=None):
    """Extract mesh using Marching Cubes."""
    print(f"Extracting mesh at iso-level {iso_level}...")
    
    # Reshape density grid to 3D
    density_3d = density_grid.reshape(dims)
    
    # Marching Cubes
    if voxel_spacing is not None and origin is not None:
        vertices, faces, normals, values = measure.marching_cubes(
            density_3d,
            level=iso_level,
            spacing=voxel_spacing,
            gradient_direction='descent'
        )
        vertices = vertices + origin
    else:
        vertices, faces, normals, values = measure.marching_cubes(
            density_3d,
            level=iso_level,
            spacing=(1.0, 1.0, 1.0),
            gradient_direction='descent'
        )
    
    print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    return vertices, faces, normals


def clean_mesh(vertices, faces):
    """Clean mesh by keeping only largest connected component."""
    print("Cleaning mesh...")
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    components = mesh.split(only_watertight=False)
    print(f"Found {len(components)} connected components")
    
    if len(components) > 1:
        largest_idx = np.argmax([c.vertices.shape[0] for c in components])
        mesh = components[largest_idx]
        print(f"Keeping largest component with {len(mesh.vertices)} vertices")
    
    # Clean mesh: merge duplicate vertices and remove unreferenced ones
    mesh.merge_vertices()  # Merge duplicate vertices
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    
    print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    return mesh


def main():
    args = parse_args()
    
    # Load checkpoint
    config, params, result_dir, checkpoint_path, checkpoint_frame = load_checkpoint(args.config, args.checkpoint)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build inverse covariances with minimum scale limit to prevent pancaking artifacts
    # Set minimum scale to half voxel size (Nyquist sampling theorem)
    min_scale_limit = args.voxel_size * 0.5
    means, inverse_covariances, opacities, scales = build_inverse_covariances(
        params, device, min_scale_limit=min_scale_limit
    )
    
    # Compute bounding box
    min_bounds, max_bounds = compute_bounding_box(means, padding=args.padding)
    
    # Calculate voxel grid dimensions (coordinates generated per-block)
    dims, voxel_spacing, origin = create_voxel_grid(
        min_bounds, max_bounds, args.voxel_size
    )
    
    # Compute density values using CPU-GPU streaming algorithm
    print("\nComputing density values using CPU-GPU streaming algorithm...")
    density_values = compute_density_tiled(
        dims, means, inverse_covariances, opacities, scales,
        min_bounds, max_bounds, args.voxel_size,
        block_size=args.block_size,
        truncate_sigma=args.truncate_sigma,
        device=device
    )
    
    # Print density statistics
    print(f"\nDensity statistics:")
    print(f"  Min: {density_values.min():.4f}")
    print(f"  Max: {density_values.max():.4f}")
    print(f"  Mean: {density_values.mean():.4f}")
    print(f"  Std: {density_values.std():.4f}")
    
    # Extract mesh
    vertices, faces, normals = extract_mesh(
        density_values, dims,
        iso_level=args.iso_level,
        voxel_spacing=voxel_spacing,
        origin=origin
    )

    # Optionally clean mesh
    if not args.no_cleaning:
        mesh = clean_mesh(vertices, faces)
    else:
        print("Skipping mesh cleaning, keeping full mesh with all components.")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    
    # Save mesh
    # 如果用户没有显式给输出文件名，则根据 checkpoint 帧号自动命名：
    #   mesh_thickened_{frame}.ply
    # 若无法解析出帧号，则回退到原来的 mesh_fast.ply
    if args.output is None:
        if checkpoint_frame is not None:
            base_name = f"mesh_thickened_{checkpoint_frame}"
        else:
            base_name = "mesh_fast"
        output_path = os.path.join(result_dir, f"{base_name}.ply")
    else:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(result_dir, args.output)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    mesh.export(output_path)
    print(f"\nMesh saved to: {output_path}")
    
    # Print statistics
    print("\nMesh Statistics:")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")
    print(f"  Bounds: {mesh.bounds}")
    print(f"  Volume: {mesh.volume:.6f}")
    
    # Auto-export OBJ next to PLY (same name & directory)
    file_dir = os.path.dirname(output_path)
    if file_dir:
        obj_path = os.path.join(file_dir, f"{base_name}.obj")
        stl_path = os.path.join(file_dir, f"{base_name}.stl")
    else:
        obj_path = f"{base_name}.obj"
        stl_path = f"{base_name}.stl"

    print(f"\nExporting mesh to OBJ: {obj_path}")
    mesh.export(obj_path)
    print(f"✓ Successfully exported OBJ to: {obj_path}")
    print("  You can open it with Blender, MeshLab, CloudCompare, etc.")
    
    # Auto-export STL next to PLY (same name & directory)
    print(f"\nExporting mesh to STL: {stl_path}")
    mesh.export(stl_path)
    print(f"✓ Successfully exported STL to: {stl_path}")
    print("  You can open it with Blender, MeshLab, CloudCompare, etc.")

    # Also export a TXT log with the same命名规范，记录本次导出关键信息和调用命令
    if file_dir:
        txt_path = os.path.join(file_dir, f"{base_name}.txt")
    else:
        txt_path = f"{base_name}.txt"

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            # 还原命令行（近似）：在前面加上 python 方便复制
            cmd_str = "python " + " ".join(sys.argv)
            f.write(f"{cmd_str}\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            if checkpoint_frame is not None:
                f.write(f"Checkpoint frame: {checkpoint_frame}\n")
            f.write(f"Voxel size: {args.voxel_size}\n")
            f.write(f"Iso level: {args.iso_level}\n")
            f.write(f"Block size: {args.block_size}\n")
            f.write(f"No cleaning: {args.no_cleaning}\n")
            f.write(f"Output PLY: {output_path}\n")
            f.write(f"Output OBJ: {obj_path}\n")
            f.write(f"Output STL: {stl_path}\n")
            f.write(f"Vertices: {len(mesh.vertices)}\n")
            f.write(f"Faces: {len(mesh.faces)}\n")
        print(f"✓ Exported log TXT to: {txt_path}")
    except Exception as e:
        print(f"[Warning] Failed to write TXT log file: {e}")

    # Optionally open interactive 3D viewer
    if args.no_show:
        print("\nSkipping 3D viewer (--no-show set).")
    else:
        print("\nOpening 3D viewer...")
        print("Controls:")
        print("  Left mouse drag: Rotate")
        print("  Mouse wheel: Zoom")
        print("  Middle/Right mouse drag: Pan")
        print("  'w' key: Toggle wireframe")
        print("  'a' key: Toggle axes")
        print("\nClose the window to exit.")
        mesh.show()


if __name__ == "__main__":
    main()

