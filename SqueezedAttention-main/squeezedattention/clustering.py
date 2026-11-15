import torch
import torch.nn.functional as F
import math
import time
import os
import numpy as np

# cuML/CuPy là tàn dư của bản cài đặt cũ (xem old_clustering.py) — file này
# không còn dùng tới chúng ở bất kỳ dòng nào. Để import cứng thì cả pipeline
# phải cài RAPIDS mới chạy được, nên chuyển thành import tuỳ chọn.
try:
    from cuml.cluster import KMeans
    import cupy as cp
    from cupy import fromDlpack
    _HAS_CUML = True
except ImportError:
    KMeans = None
    cp = None
    fromDlpack = None
    _HAS_CUML = False

from torch.utils.dlpack import to_dlpack

def batched_kmeans_pytorch(x_norm, K, max_iters=100):
    # QUAN TRỌNG: toàn bộ k-means phải chạy ở float32.
    # Ở float16: (1) torch.cdist mất chính xác nặng với N vài nghìn x D=128 nên
    # nhiều khoảng cách bằng nhau, argmin khi hoà luôn chọn chỉ số nhỏ nhất ->
    # dồn hết token vào một cụm; (2) counts bão hoà ở 2048 (fp16 chỉ biểu diễn
    # chính xác số nguyên tới 2048) nên mẫu số sai khi tính lại centroid.
    x_norm = x_norm.float()
    H, N, D = x_norm.shape

    # Khởi tạo: K điểm dữ liệu PHÂN BIỆT cho mỗi head (randperm thay cho randint,
    # tránh nhiều centroid trùng nhau ngay từ đầu).
    if K <= N:
        indices = torch.stack([torch.randperm(N, device=x_norm.device)[:K] for _ in range(H)])
    else:
        indices = torch.randint(0, N, (H, K), device=x_norm.device)
    centroids = torch.gather(x_norm, 1, indices.unsqueeze(-1).expand(-1, -1, D))

    labels = None
    for _ in range(max_iters):
        # Tính khoảng cách và tìm label (batched cdist)
        dist = torch.cdist(x_norm, centroids)
        new_labels = torch.argmin(dist, dim=-1)

        # Hội tụ: nhãn không đổi thì dừng sớm
        if labels is not None and torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

        # Cập nhật centroids
        new_centroids = torch.zeros_like(centroids)
        labels_expanded = labels.unsqueeze(-1).expand(-1, -1, D)
        new_centroids.scatter_add_(1, labels_expanded, x_norm)

        counts = torch.zeros((H, K), device=x_norm.device, dtype=torch.float32)
        counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))

        empty = (counts == 0)  # phải lấy TRƯỚC khi clamp
        centroids = new_centroids / counts.clamp(min=1).unsqueeze(-1)
        # Chuẩn hóa lại centroids
        centroids = F.normalize(centroids, p=2, dim=-1)

        # Hồi sinh cụm chết: cụm rỗng có centroid = vector 0, mà F.normalize(0) vẫn
        # là 0 — centroid 0 cách mọi điểm đã chuẩn hoá đúng 1.0 nên nó hút hết các
        # token không đủ gần centroid nào, tạo ra một cụm khổng lồ. Gán lại về một
        # điểm dữ liệu ngẫu nhiên để cụm sống lại.
        if empty.any():
            ridx = torch.randint(0, N, (H, K), device=x_norm.device)
            rpts = torch.gather(x_norm, 1, ridx.unsqueeze(-1).expand(-1, -1, D))
            centroids = torch.where(empty.unsqueeze(-1), rpts, centroids)

    return labels

def run_clustering(tdict, num_clusters, observation_window=100, print_log=False, device=None):
    calc_device = "cuda:0" 
    save_device = device if device is not None else "cpu"

    centroids_tensor_dict = {}
    centroids_labels_dict = {}

    num_lyrs = len(tdict)

    t1 = time.time()
    for layer_num in range(num_lyrs):
        if print_log:
            print('layer: ', layer_num)

        # FIX: .float() — bản gốc (old_clustering.py) có, bản viết lại đánh rơi.
        # Thiếu nó thì cdist và counts chạy ở fp16 và clustering sụp thành 1 cụm.
        orig_dtype = tdict[layer_num].dtype
        keys = tdict[layer_num].squeeze(0).to(calc_device).float() # [32, N, 128] float32
        
        K = num_clusters
        assert(len(keys.shape) == 3)

        if observation_window > 0:
            keys = keys[:,:-observation_window,:]
        
        H, N, D = keys.shape

        # 1. Chuẩn hóa data
        data_normalized = F.normalize(keys, p=2, dim=-1)

        # 2. Chạy KMeans BATCHED (Tính đồng thời 32 Heads cực nhanh)
        labels_gpu = batched_kmeans_pytorch(data_normalized, K, max_iters=100)

        # 3. Tính lại centroids thực (chưa chuẩn hóa) bằng scatter_add
        labels_expanded = labels_gpu.unsqueeze(-1).expand(-1, -1, D)
        unnorm_centers = torch.zeros((H, K, D), device=calc_device, dtype=keys.dtype)
        unnorm_centers.scatter_add_(1, labels_expanded, keys)

        counts = torch.zeros((H, K), device=calc_device, dtype=keys.dtype)
        counts.scatter_add_(1, labels_gpu, torch.ones_like(labels_gpu, dtype=keys.dtype))
        counts = counts.clamp(min=1).unsqueeze(-1)
        
        cluster_centers = unnorm_centers / counts

        # Lưu lại kết quả — tính ở float32 nhưng lưu về dtype gốc để kích thước
        # file .pt không đổi (lúc load, modeling đã tự .to(query_states.dtype))
        centroids_tensor_dict[layer_num] = cluster_centers.unsqueeze(0).to(orig_dtype).to(save_device)
        centroids_labels_dict[layer_num] = labels_gpu.unsqueeze(0).to(save_device).to(torch.int64)

    return centroids_tensor_dict, centroids_labels_dict


def run_global_threshold(key_dict, query_dict, centroids_tensor_dict, centroids_labels_dict, num_clusters, observation_window=100, print_log=False, device=None):
    calc_device = "cuda:0"
    save_device = device if device is not None else "cpu"

    shared_prefix_length = query_dict[0].shape[-2]
    num_lyrs = len(query_dict)
    K = num_clusters

    attn_score_centroid_list = []
    for layer_num in range(num_lyrs):
        if print_log:
            print('layer: ', layer_num)

        # Đẩy lên GPU để nhân ma trận
        centroids_tensor = centroids_tensor_dict[layer_num].squeeze(0).to(calc_device)
        centroids_labels = centroids_labels_dict[layer_num].squeeze(0).to(calc_device)

        keys = key_dict[layer_num].squeeze(0).to(calc_device)
        queries = query_dict[layer_num].squeeze(0).to(calc_device)
        keys_shared_prefix = keys[:, :-observation_window, :]

        queries_obs_window = queries[:, -observation_window:, :].float()
        attn_scores_centroids = (torch.matmul(queries_obs_window, centroids_tensor.float().transpose(1, 2)) / math.sqrt(keys.shape[-1]))

        H = keys_shared_prefix.shape[0]
        N_obs = observation_window

        # Vectorized Loop 1: Gán score từ centroid sang các token (Thay cho vòng lặp chậm)
        # attn_scores_centroids: [H, N_obs, K] -> transpose(1, 2) -> [H, K, N_obs]
        idx = centroids_labels.unsqueeze(-1).expand(-1, -1, N_obs) # [H, N_shared, N_obs]
        scores = torch.gather(attn_scores_centroids.transpose(1, 2), 1, idx)

        # Vectorized Loop 2: Đếm số token trong mỗi cluster
        num_keys_per_cluster = torch.zeros((H, K), device=calc_device, dtype=torch.float32)
        num_keys_per_cluster.scatter_add_(1, centroids_labels, torch.ones_like(centroids_labels, dtype=torch.float32))

        # === FIX: Max subtraction để tránh exp() overflow → NaN ===
        # Trừ max trước khi exp() KHÔNG thay đổi kết quả toán học:
        #   exp(x - max) / sum(N_k * exp(x_k - max)) = exp(x) / sum(N_k * exp(x_k))
        # vì max cancels out trong tỉ lệ.
        max_centroid_scores = torch.max(attn_scores_centroids, dim=-1, keepdim=True)[0]  # [H, N_obs, 1]
        attn_scores_stable = attn_scores_centroids - max_centroid_scores

        attn_scores_centroids_est_exp = torch.exp(attn_scores_stable)
        num_keys_per_cluster = num_keys_per_cluster.unsqueeze(-2)
        denom_est_tmp = num_keys_per_cluster * attn_scores_centroids_est_exp
        denom_est = torch.sum(denom_est_tmp, dim=-1)  # [H, N_obs]

        # scores: [H, N_shared, N_obs] — cũng cần trừ cùng max để nhất quán
        # max_centroid_scores: [H, N_obs, 1] → transpose → [H, 1, N_obs] để broadcast
        scores_stable = scores - max_centroid_scores.transpose(1, 2)
        scores_scaled_sm = torch.exp(scores_stable) / denom_est.unsqueeze(-2)
        scored_scaled_sm_sum = torch.mean(scores_scaled_sm, dim=-1, dtype=torch.float32)
        
        # Kéo kết quả về CPU
        attn_score_centroid_list.append(scored_scaled_sm_sum.to(save_device))

        del centroids_tensor, centroids_labels, keys, queries, keys_shared_prefix
        del queries_obs_window, attn_scores_centroids, scores, num_keys_per_cluster
        torch.cuda.empty_cache()

    full_centroid_scores = torch.stack(attn_score_centroid_list, dim=0)

    qlist = [0.5, 0.7, 0.8, 0.9]
    q = torch.tensor(qlist, device="cpu")

    full_centroid_scores_cpu = full_centroid_scores.cpu().numpy()
    quantile_result = np.quantile(full_centroid_scores_cpu, q.numpy())
    thresholds = torch.tensor(quantile_result)

    tdict = {}
    for i, q_idx in enumerate(qlist):
        tdict[q_idx] = thresholds[i].item()

    tdict['shared_prefix_length'] = shared_prefix_length
    tdict['observation_window'] = observation_window

    return tdict