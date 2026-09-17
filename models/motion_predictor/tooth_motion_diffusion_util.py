import torch

def mat2vec(m):
    S = (m - m.transpose(1, 0)) / 2
    vec = torch.zeros_like(m[0, :])
    vec[0] = S[2, 1]
    vec[1] = S[0, 2]
    vec[2] = S[1, 0]
    sin_theta = torch.norm(vec)
    theta = torch.asin(torch.clamp(sin_theta, min=-1.0, max=1.0))
    if sin_theta > 1e-12:
        vec = vec / sin_theta
        vec = vec * theta
    else:
        vec = vec - vec
    return vec


def S(n):
    Sn = torch.zeros(3, 3, device=n.device)
    Sn[0, 1] = -n[2]
    Sn[0, 2] = n[1]
    Sn[1, 0] = n[2]
    Sn[1, 2] = -n[0]
    Sn[2, 0] = -n[1]
    Sn[2, 1] = n[0]
    return Sn


def vec2mat(vec):
    theta = torch.norm(vec)
    if theta > 1e-7:
        n = vec / theta
        Sn = S(n)
        R = torch.eye(3, device=vec.device) + torch.sin(theta) * Sn + (1 - torch.cos(theta)) * torch.mm(Sn, Sn)
    else:
        Sr = S(vec)
        theta2 = theta ** 2
        R = torch.eye(3, device=vec.device) + (1 - theta2 / 6.) * Sr + (.5 - theta2 / 24.) * torch.mm(Sr, Sr)
    return R
