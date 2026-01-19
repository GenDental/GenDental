import torch

# ---------------------------
# so3 exp / log (stable, vectorized)
# ---------------------------
def so3_log(R):
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    small = theta.abs() < 1e-8

    r32_r23 = R[..., 2, 1] - R[..., 1, 2]
    r13_r31 = R[..., 0, 2] - R[..., 2, 0]
    r21_r12 = R[..., 1, 0] - R[..., 0, 1]

    denom = 2.0 * torch.sin(theta)
    denom_safe = denom.clone()
    denom_safe[denom_safe.abs() < 1e-8] = 1.0

    axis_x = r32_r23 / denom_safe
    axis_y = r13_r31 / denom_safe
    axis_z = r21_r12 / denom_safe

    rv_general = torch.stack([axis_x * theta, axis_y * theta, axis_z * theta], dim=-1)
    rv_small = 0.5 * torch.stack([r32_r23, r13_r31, r21_r12], dim=-1)

    rv = torch.where(small.unsqueeze(-1), rv_small, rv_general)
    return rv


def so3_exp(rotvec):
    theta = torch.norm(rotvec, dim=-1, keepdim=True)
    small = (theta.squeeze(-1).abs() < 1e-8)
    axis = rotvec / (theta + 1e-16)
    ux, uy, uz = axis.unbind(dim=-1)

    cos_t = torch.cos(theta)[..., 0]
    sin_t = torch.sin(theta)[..., 0]
    one_minus_cos = 1.0 - cos_t

    shape = rotvec.shape[:-1]
    R = torch.zeros(*shape, 3, 3, device=rotvec.device, dtype=rotvec.dtype)

    ux2 = ux * ux
    uy2 = uy * uy
    uz2 = uz * uz

    R[..., 0, 0] = cos_t + ux2 * one_minus_cos
    R[..., 0, 1] = ux * uy * one_minus_cos - uz * sin_t
    R[..., 0, 2] = ux * uz * one_minus_cos + uy * sin_t

    R[..., 1, 0] = uy * ux * one_minus_cos + uz * sin_t
    R[..., 1, 1] = cos_t + uy2 * one_minus_cos
    R[..., 1, 2] = uy * uz * one_minus_cos - ux * sin_t

    R[..., 2, 0] = uz * ux * one_minus_cos - uy * sin_t
    R[..., 2, 1] = uz * uy * one_minus_cos + ux * sin_t
    R[..., 2, 2] = cos_t + uz2 * one_minus_cos

    if small.any():
        idx = small.view(-1)
        if idx.any():
            rv_small = rotvec.view(-1, 3)[idx, :]
            x = rv_small[:, 0]; y = rv_small[:, 1]; z = rv_small[:, 2]
            K = torch.zeros((rv_small.shape[0], 3, 3), device=rotvec.device, dtype=rotvec.dtype)
            K[:, 0, 1] = -z; K[:, 0, 2] = y
            K[:, 1, 0] = z;  K[:, 1, 2] = -x
            K[:, 2, 0] = -y; K[:, 2, 1] = x
            R_approx = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).unsqueeze(0) + K
            R.view(-1, 3, 3)[idx, ...] = R_approx

    return R


# ---------------------------
# build first-order Laplacian
# ---------------------------
def build_first_order_L(l, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device('cpu')

    diag = torch.full((l,), 2.0, device=device, dtype=dtype)
    if l > 0:
        diag[0] = 1.0
        diag[-1] = 1.0

    L = torch.zeros((l, l), device=device, dtype=dtype)
    idx = torch.arange(l, device=device)
    L[idx, idx] = diag
    if l > 1:
        off = -1.0
        L[idx[:-1], idx[1:]] = off
        L[idx[1:], idx[:-1]] = off
    return L


# ---------------------------
# 后处理平滑主函数（加 damping）
# ---------------------------
def postprocess_smooth_se3(predicted_matrices,
                           masks=None,
                           lambda_trans=20.0,
                           lambda_rot=20.0,
                           eps=1e-6):
    b, l, p, _, _ = predicted_matrices.shape
    device = predicted_matrices.device
    pred = predicted_matrices.float()

    if masks is None:
        masks_tp = torch.ones((b, l, p), device=device)
    else:
        masks = masks.to(device)
        if masks.ndim == 2:
            masks_tp = masks.unsqueeze(1).float().expand(b, l, p)
        elif masks.ndim == 3:
            masks_tp = masks.float()
        else:
            raise ValueError("masks must be shape (b,p) or (b,l,p)")

    t0 = pred[..., :3, 3].contiguous()
    R0 = pred[..., :3, :3].contiguous()
    r0 = so3_log(R0)

    bp = b * p
    t0_flat = t0.permute(0, 2, 1, 3).contiguous().view(bp, l, 3)
    r0_flat = r0.permute(0, 2, 1, 3).contiguous().view(bp, l, 3)
    mask_flat = masks_tp.permute(0, 2, 1).contiguous().view(bp, l)

    L = build_first_order_L(l, device=device)
    t_smoothed = torch.zeros_like(t0_flat)
    r_smoothed = torch.zeros_like(r0_flat)

    max_chunk = 512
    start = 0
    while start < bp:
        end = min(bp, start + max_chunk)
        m_chunk = mask_flat[start:end]
        c = m_chunk.shape[0]

        W = torch.zeros((c, l, l), device=device)
        idx = torch.arange(l, device=device)
        W[:, idx, idx] = m_chunk

        A_trans = W + lambda_trans * L.unsqueeze(0) + eps * torch.eye(l, device=device).unsqueeze(0)
        A_rot   = W + lambda_rot * L.unsqueeze(0) + eps * torch.eye(l, device=device).unsqueeze(0)

        t0_chunk = t0_flat[start:end]
        r0_chunk = r0_flat[start:end]

        RHS_t = (m_chunk.unsqueeze(-1) * t0_chunk)
        RHS_r = (m_chunk.unsqueeze(-1) * r0_chunk)

        # solve safely
        X_t = torch.linalg.solve(A_trans, RHS_t)
        X_r = torch.linalg.solve(A_rot, RHS_r)

        t_smoothed[start:end] = X_t
        r_smoothed[start:end] = X_r

        start = end

    t_sm = t_smoothed.view(b, p, l, 3).permute(0, 2, 1, 3).contiguous()
    r_sm = r_smoothed.view(b, p, l, 3).permute(0, 2, 1, 3).contiguous()
    R_sm = so3_exp(r_sm)

    smoothed = torch.eye(4, device=device).view(1,1,1,4,4).repeat(b,l,p,1,1)
    smoothed[..., :3, :3] = R_sm
    smoothed[..., :3, 3] = t_sm

    return smoothed