import torch
import math
from einops import rearrange, repeat
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d
from models.motion_predictor.tooth_motion_diffusion_util import *

PARAMETER_A = 0.005 # add a little value to distance incase zero or other

def ChamferVector(p1, p2):
    '''
    Calculate Chamfer Vector between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :return: sum of Chamfer Vector of two point sets
    '''

    B, N, C = p1.size()
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff * diff, dim=3)
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, idxfrom1to2 = torch.min(dist1, dim=2)
    dist_min2, idxfrom2to1 = torch.min(dist2, dim=2)

    ChamferVector = torch.zeros([B, N, C * 2], device=p1.device)
    for b in range(B):
        ChamferVector[b, :, :3] = p1[b, :, :] - p2[b, idxfrom1to2[b, :], :]
        ChamferVector[b, :, 3:] = p2[b, :, :] - p1[b, idxfrom2to1[b, :], :]
    # print(idxfrom1to2.size())
    return ChamferVector


def square_distance(src, dst, normalised=False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points , [B, N, C]
        dst: target points , [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if (normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    
    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist

def nearest_dist(pc1, pc2):
    """
    Calculate the nearest Euclid distance between two point clouds.
    :param pc1_tensor: Tensor of point cloud pc1
    :param pc1_tensor: Tensor of point cloud pc2
    :return min_dist:  The nearest Euclid distance between two point clouds
    :return index:
    """

    dist_mat = square_distance(pc1, pc2, False)
    dist_mat = dist_mat[0]
    nums = len(dist_mat)
    min_dist = torch.min(dist_mat)
    min_index = torch.argmin(dist_mat)

    row = min_index / nums
    col = min_index % nums
    row = int(row.item())
    col = col.item()

    pc1 = pc1[0]
    pc2 = pc2[0]

    point1 = pc1[row, :]
    point2 = pc2[col, :]

    d = point2 - point1
    # dist = math.hypot(d[0], d[1], d[2])
    dist = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    # dist = math.hypot(d[0], d[1])

    return dist

def cal_collision(pc_step_b):

    B, S, T, PC, _ = pc_step_b.shape

    collision_pc_step = torch.zeros_like(pc_step_b)

    collision_pc_step[:, :, 0:7, :, :] = torch.flip(pc_step_b[:, :, 0:7 :, :], dims=[2])
    collision_pc_step[:, :, 7:14, :, :] = pc_step_b[:, :, 7:14, :, :]
    collision_pc_step[:, :, 14:21, :, :] = torch.flip(pc_step_b[:, :, 14:21, :, :], dims=[2])
    collision_pc_step[:, :, 21:28, :, :] = pc_step_b[:, :, 21:28, :, :]


    collision_loss = torch.zeros([B, S])

    for b in range(B):
        for s in range(S):
            # 上牙齿
            U_upper = []
            for t in range(0, 13):
                pc_next = collision_pc_step[b, s, t+1, :]
                pc = collision_pc_step[b, s, t, :]
                centroid = torch.mean(pc, dim=0)
                centroid2 = torch.mean(pc_next, dim=0)
                
                if torch.mean(centroid) == 0:
                    dist = 0
                    U = 0 
                else: 
                    dist = nearest_dist(pc.unsqueeze(0), pc_next.unsqueeze(0))

                    lm = math.sqrt((centroid[0] - centroid2[0])**2 + (centroid[1] - centroid2[1])**2+ (centroid[2] - centroid2[2])**2)
                    tmp_lj = dist/lm
                    tmp_lj += PARAMETER_A
                    
                    tmp_value = 1/(1 + tmp_lj)
                    U = np.power(tmp_value, 12) - 2 * np.power(tmp_value, 6) + 1

                U_upper.append(torch.tensor(U))
            U_upper = torch.stack(U_upper)
            
            U_lower = []
            for t in range(14, 27):   
                pc_next = collision_pc_step[b, s, t+1, :]
                pc = collision_pc_step[b, s, t, :]
                centroid = torch.mean(pc, dim=0) 

                if torch.mean(centroid) == 0:
                    dist = 0
                    U = 0 
                else: 
                    dist = nearest_dist(pc.unsqueeze(0), pc_next.unsqueeze(0))

                    lm = math.sqrt((centroid[0] - centroid2[0])**2 + (centroid[1] - centroid2[1])**2+ (centroid[2] - centroid2[2])**2)
                    tmp_lj = dist/lm
                    tmp_lj += PARAMETER_A
                    
                    tmp_value = 1/(1 + tmp_lj)
                    U = np.power(tmp_value, 12) - 2 * np.power(tmp_value, 6) + 1   

                    U_lower.append(torch.tensor(U))
            U_lower = torch.stack(U_lower)
        
            collision_loss[b, s] = torch.mean(U_upper) + torch.mean(U_lower)

    return torch.mean(collision_loss)


def get_step_pc(rtv_step, rtv_step_gt, before_pts):      
    n,b,l,c = rtv_step.shape
    before_points = repeat(before_pts,'b n p c -> (b l n) p c', l=l)

    pred_params = rearrange(rtv_step,'n b l c -> (b l n) c')
    gt_params = rearrange(rtv_step_gt,'n b l c -> (b l n) c')

    pred_angles = pred_params[:,:3]
    pred_trans = pred_params[:,3:]
    pred_matrices = euler_angles_to_matrix(pred_angles, convention='XYZ')

    predicted_matrices = torch.zeros((b*l*n,4,4)).to(pred_trans.device)
    predicted_matrices[:,:3,:3] = pred_matrices
    predicted_matrices[:,:3,3] = pred_trans
    predicted_matrices[:,3,3] = 1
    predicted_matrices = rearrange(predicted_matrices,'b c1 c2 -> b c2 c1')
    pred_transform = Transform3d(matrix=predicted_matrices)
    pred_points = pred_transform.transform_points(before_points)

    gt_angles = gt_params[:,:3]
    gt_trans = gt_params[:,3:]
    gt_matrices = euler_angles_to_matrix(gt_angles, convention='XYZ')
    gt_matrices_full = torch.zeros((b*l*n,4,4)).to(gt_trans.device)
    gt_matrices_full[:,:3,:3] = gt_matrices
    gt_matrices_full[:,:3,3] = gt_trans
    gt_matrices_full[:,3,3] = 1
    gt_matrices_full = rearrange(gt_matrices_full,'b c1 c2 -> b c2 c1')
    gt_transform = Transform3d(matrix=gt_matrices_full)
    gt_points = gt_transform.transform_points(before_points)

    # predicted_reshaped = rearrange(pred_points,'(b l p) n c -> b l p n c', b=b, l=l)
    # gt_reshaped = rearrange(gt_points,'(b l p) n c -> b l p n c', b=b, l=l)

    # return predicted_reshaped, gt_reshaped
    return pred_points, gt_points

def cal_ADD(rtv_step, rtv_step_gt, before_pts, masks):      
    n,b,l,c = rtv_step.shape
    before_points = repeat(before_pts,'b n p c -> (b l n) p c', l=l)

    pred_params = rearrange(rtv_step,'n b l c -> (b l n) c')
    gt_params = rearrange(rtv_step_gt,'b n l c -> (b l n) c')

    pred_angles = pred_params[:,:3]
    pred_trans = pred_params[:,3:]
    pred_matrices = euler_angles_to_matrix(pred_angles, convention='XYZ')

    predicted_matrices = torch.zeros((b*l*n,4,4)).to(pred_trans.device)
    predicted_matrices[:,:3,:3] = pred_matrices
    predicted_matrices[:,:3,3] = pred_trans
    predicted_matrices[:,3,3] = 1
    predicted_matrices = rearrange(predicted_matrices,'b c1 c2 -> b c2 c1')
    pred_transform = Transform3d(matrix=predicted_matrices)
    pred_points = pred_transform.transform_points(before_points)

    gt_angles = gt_params[:,:3]
    gt_trans = gt_params[:,3:]
    gt_matrices = euler_angles_to_matrix(gt_angles, convention='XYZ')
    gt_matrices_full = torch.zeros((b*l*n,4,4)).to(gt_trans.device)
    gt_matrices_full[:,:3,:3] = gt_matrices
    gt_matrices_full[:,:3,3] = gt_trans
    gt_matrices_full[:,3,3] = 1
    gt_matrices_full = rearrange(gt_matrices_full,'b c1 c2 -> b c2 c1')
    gt_transform = Transform3d(matrix=gt_matrices_full)
    gt_points = gt_transform.transform_points(before_points)

    predicted_reshaped = rearrange(pred_points,'(b l p) n c -> b l p n c', b=b, l=l)
    gt_reshaped = rearrange(gt_points,'(b l p) n c -> b l p n c', b=b, l=l)
    ADD_mask = repeat(masks,'b p -> (b l p n)', l=l, n=512)
    ADD = 30 * torch.norm(predicted_reshaped - gt_reshaped, dim=-1).sum() / ADD_mask.sum()

    # return predicted_reshaped, gt_reshaped
    return ADD



def get_pred_frame(RTV_Pre, start_frame):
    device = RTV_Pre.device
    T, B, S, _ = RTV_Pre.size()
    B, T, FJ, FC= start_frame.size()


    RTV_steps = RTV_Pre.transpose(0, 2) # [steps, b, tooth, 6]

    frames_steps = []
    frames_steps.append(start_frame)
    for it in range(S):
        frame_new = torch.zeros_like(start_frame)  # [b, tooth_nums, 4, 3]
        for b in range(B):
            for t in range(T):
                # calculate Rotation and Transform
                trans_param1 = RTV_steps[it, b, t, :]
                R1 = vec2mat(trans_param1[:3])
                T1 = trans_param1[3:]

                frame_new[b, t, :3, :] = torch.mm(R1, frames_steps[it][b, t, :3, :].transpose(0, 1)).transpose(0, 1)
                frame_new[b, t, 3, :] = frames_steps[it][b, t, 3, :] + T1

        frames_steps.append(frame_new)
    frames_steps = torch.stack(frames_steps)[1:]
            
    return frames_steps
