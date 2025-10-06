from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.linalg import svd as full_svd
from PIL import Image
from scipy.ndimage import gaussian_filter
from numpy.random import default_rng


def load_image(path: str, color_mode: str = "rgb"):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    if color_mode == "gray":
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return gray
    elif color_mode == "rgb":
        return arr
    else:
        raise ValueError("color_mode must be 'gray' or 'rgb'")


def to_float(arr):
    return arr.astype(np.float64, copy=False)


def dynamic_range(arr):
    return 255.0 if arr.max() <= 255.0 else arr.max()


def randomized_range_finder(A, size, n_iter, rng):
    G = rng.standard_normal((A.shape[1], size))
    Y = A @ G
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    return Q


def randomized_svd(A, k, p, n_iter, rng):
    m, n = A.shape
    size = min(k + p, n)
    Q = randomized_range_finder(A, size=size, n_iter=n_iter, rng=rng)
    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub[:, :k]
    return U[:, :k], S[:k], Vt[:k, :]


def compute_svd_channel(X, method, k_max, oversample, n_power_iter, rng):
    if method == "full":
        U, S, Vt = full_svd(X, full_matrices=False)
        return U, S, Vt
    elif method == "randomized":
        U, S, Vt = randomized_svd(X, k_max, oversample, n_power_iter, rng)
        return U, S, Vt
    else:
        raise ValueError("method must be 'full' or 'randomized'")


def reconstruct_rank_k(U, S, Vt, k):
    k = min(k, len(S))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


def mse(x, y):
    return float(np.mean((x - y) ** 2))


def psnr(x, y, data_range):
    err = mse(x, y)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10((data_range**2) / err)


def _local_stats(x, win_size, sigma):
    mu = gaussian_filter(x, sigma=sigma, truncate=((win_size - 1) / 2) / sigma)
    mu_sq = mu * mu
    sigma_sq = (
        gaussian_filter(x * x, sigma=sigma, truncate=((win_size - 1) / 2) / sigma)
        - mu_sq
    )
    return mu, sigma_sq


def ssim_single(x, y, data_range, win_size=11, sigma=1.5, K1=0.01, K2=0.03):
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu_x, sig_x2 = _local_stats(x, win_size, sigma)
    mu_y, sig_y2 = _local_stats(y, win_size, sigma)
    mu_xy = gaussian_filter(x * y, sigma=sigma, truncate=((win_size - 1) / 2) / sigma)
    sig_xy = mu_xy - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x2 + sig_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def ssim_mean(x, y, data_range, win_size=11, sigma=1.5, K1=0.01, K2=0.03):
    if x.ndim == 2:
        return ssim_single(x, y, data_range, win_size, sigma, K1, K2)
    else:
        vals = [
            ssim_single(x[..., c], y[..., c], data_range, win_size, sigma, K1, K2)
            for c in range(x.shape[2])
        ]
        return float(np.mean(vals))


@dataclass
class SVDResult:
    U: List[np.ndarray]
    S: List[np.ndarray]
    Vt: List[np.ndarray]
    total_energy: float


def per_channel_svd(X, method, k_max, oversample, n_power_iter, seed):
    rng = default_rng(seed)
    channels = [X] if X.ndim == 2 else [X[..., c] for c in range(X.shape[2])]
    U_list, S_list, Vt_list = [], [], []
    energy = 0.0
    for ch in channels:
        U, S, Vt = compute_svd_channel(ch, method, k_max, oversample, n_power_iter, rng)
        U_list.append(U)
        S_list.append(S)
        Vt_list.append(Vt)
        energy += float(np.sum(S**2))
    return SVDResult(U_list, S_list, Vt_list, energy)


def reconstruct_from_factors(res, k, shape_hw_c):
    if len(shape_hw_c) == 2:
        return reconstruct_rank_k(res.U[0], res.S[0], res.Vt[0], k)
    else:
        H, W, C = shape_hw_c
        chans = []
        for c in range(C):
            Xkc = reconstruct_rank_k(res.U[c], res.S[c], res.Vt[c], k)
            chans.append(Xkc[:, :W])
        import numpy as np

        return np.stack(chans, axis=-1)


def energy_retained(res, k):
    num = 0.0
    for S in res.S:
        kk = min(k, len(S))
        num += float(np.sum(S[:kk] ** 2))
    return num / (res.total_energy + 1e-12)
