import faiss
import faiss.contrib.torch_utils
import torch


def torch_3d_knn(pts, num_knn, method="l2"):
    # Initialize FAISS index
    if method == "l2":
        index = faiss.IndexFlatL2(pts.shape[1])
    elif method == "cosine":
        index = faiss.IndexFlatIP(pts.shape[1])
    else:
        raise NotImplementedError(f"Method: {method}")

    use_gpu = False
    if pts.get_device() != -1 and hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            use_gpu = True
        except Exception:
            use_gpu = False

    # CPU FAISS cannot take CUDA tensors; keep results on the original device.
    pts_in = pts if use_gpu else pts.detach().cpu()
    index.add(pts_in)
    distances, indices = index.search(pts_in, num_knn)
    if (not use_gpu) and pts.get_device() != -1:
        distances = distances.to(pts.device)
        indices = indices.to(pts.device)
    return distances, indices
    

def calculate_neighbors(params, variables, time_idx, num_knn=20):
    if time_idx is None:
        pts = params['means3D'].detach()
    else:
        pts = params['means3D'][:, :, time_idx].detach()
    neighbor_dist, neighbor_indices = torch_3d_knn(pts.contiguous(), num_knn)
    neighbor_weight = torch.exp(-2000 * torch.square(neighbor_dist))
    variables["neighbor_indices"] = neighbor_indices.long().contiguous()
    variables["neighbor_weight"] = neighbor_weight.float().contiguous()
    variables["neighbor_dist"] = neighbor_dist.float().contiguous()
    return variables