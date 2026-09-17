import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module
from models.motion_predictor.gnn import GatedGraphNeuralNetwork, AdjacencyList
from models.motion_predictor.tooth_motion_diffusion_util import *
from models.motion_predictor.tooth_motion_diffusion_loss import *
 
class Denosier(nn.Module):
    def __init__(self, num_teeth, step, input_dim, hidden_size, num_layers):
        super(Denosier, self).__init__()
        self.input_size = input_dim  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers 

        self.gru_r=nn.GRU(
            input_size=self.input_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True 
        )

        self.gru_t=nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True 
        )

        self.fc_r=nn.Linear(self.hidden_size, 3)
        self.fc_t=nn.Linear(self.hidden_size, 3)

    def forward(self, x):

        B, S, _ = x.shape
        device = x.device

        h_state = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)

        r_out, r_h_n=self.gru_r(x, h_state)
        t_out, t_h_n=self.gru_t(x, h_state)  
        
        rtv_step = torch.zeros([S, B, 6]).to(device)
        for time_step in range(r_out.size(1)):
            rtv_step[time_step, :, :3] = self.fc_r(r_out[:,time_step,:])
            rtv_step[time_step, :, 3:] = self.fc_t(t_out[:,time_step,:])

        rtv_step = rtv_step.transpose(0, 1)

        return rtv_step 


class Planning_Denosier(nn.Module):
    def __init__(self, num_teeth, step, njoints, nfeats, shape_dim, hidden_size, num_layers):
        super(Planning_Denosier, self).__init__()
        self.num_teeth = num_teeth
        self.step = step
        self.njoints = njoints
        self.nfeats = nfeats
        self.shape_dim = shape_dim
        self.input_c = 6
        self.node_dim = 6  # or other
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.condition_feature = self.shape_dim 
        self.input_dim = self.condition_feature + self.input_c


        self.super_node_encode = nn.Sequential(
            nn.Conv1d(self.node_dim * self.num_teeth, self.node_dim*2, 1, bias=False),
            # nn.Conv1d(self.node_dim * self.step, self.node_dim*2, 1, bias=False),
            nn.BatchNorm1d(self.node_dim*2),
            nn.ReLU()
        )
        

        '''# gnn for local feature propagation
        ########################################################################'''
        residual_connections = {}
        edge_list_1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                     (0, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
                     (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),
                     (14, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (28, 29), (29, 30), (30,31)]
        
        edge_list_2 = [(1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8),
                      (17, 30), (18, 29), (19, 28), (20, 27), (21, 26), (22, 25), (23, 24),]

        # edge_list_3 = edge_list_4 and edge_list_5
        edge_list_4 = [(0, 32), (1, 32), (2, 32), (3, 32), (4, 32), (5, 32), (6, 32), (7, 32), (8, 32), (9, 32), (10, 32), (11, 32), (12, 32), (13, 32),(14, 32), (15, 32), (16, 33), (17, 33), (18, 33), (19, 33), (20, 33), (21, 33), (22, 33), (23, 33), (24, 33), (25, 33), (26, 33), (27, 33), (28, 33), (29, 33), (30,33), (31,33)] #super-node
        edge_list_5 = [(32,33)] #super-super
        #MODIFY
        src_1, dst_1 = zip(*edge_list_1)
        src_2, dst_2 = zip(*edge_list_2)
        src_4, dst_4 = zip(*edge_list_4)
        src_5, dst_5 = zip(*edge_list_5)
        edge_list_1 = edge_list_1 + list(zip(dst_1, src_1))
        edge_list_2 = edge_list_2 + list(zip(dst_2, src_2))
        edge_list_4 = edge_list_4 + list(zip(dst_4, src_4))
        edge_list_5 = edge_list_5 + list(zip(dst_5, src_5))

        # num_edge_types=4#4#2
        self.gnn = GatedGraphNeuralNetwork(hidden_size=self.node_dim, num_edge_types=4,
                                      layer_timesteps=[3, 3, 3], residual_connections=residual_connections)

        adj_list_type1 = AdjacencyList(node_num=self.num_teeth, adj_list=edge_list_1, device=self.gnn.device)
        adj_list_type2 = AdjacencyList(node_num=self.num_teeth, adj_list=edge_list_2, device=self.gnn.device)
        adj_list_type4 = AdjacencyList(node_num=self.num_teeth, adj_list=edge_list_4, device=self.gnn.device)
        adj_list_type5 = AdjacencyList(node_num=self.num_teeth, adj_list=edge_list_5, device=self.gnn.device)

        self.adj_list = nn.ModuleList([
            adj_list_type1,
            adj_list_type2,
            adj_list_type4,
            adj_list_type5,
        ])


        motivation_regression = []
        for i in range(self.num_teeth):
            denosier_layer = Denosier(self.num_teeth, self.step, self.input_dim, self.hidden_size, self.num_layers)
            motivation_regression.append(denosier_layer)
        self.motivation_regression = nn.ModuleList(motivation_regression)

    def forward(self, noise_x0, shape_code):

        T, B, S, RD = noise_x0.size()
        device = noise_x0.device

        # target_frame = frame_ending.repeat(self.step,1,1,1,1) 
        # target_frame = target_frame.transpose(0, 2)

        # start_frame = frame_start.repeat(self.step,1,1,1,1) 
        # start_frame = start_frame.transpose(0, 2)

        shape_code = shape_code.transpose(0, 1)
        shape_code = shape_code.repeat(self.step,1,1,1) 
        shape_code = shape_code.transpose(0, 2)

        noise_x0 = noise_x0.permute(1, 2, 0, 3)  # B, S, T, RD
        initial_node_feat = noise_x0.reshape(B*S, T, RD)  
        super_infor = noise_x0.reshape(B*S, T*RD, 1) 
        super_node_feat = self.super_node_encode(super_infor).view(B*S, self.node_dim, 2) 
        tooth_node_feat = torch.cat((initial_node_feat, super_node_feat.transpose(1, 2)), dim=1) 
        tooth_node_feat = tooth_node_feat.reshape(B, S, T+2, self.node_dim) 

        gnn_local_feat = torch.zeros([self.step, B, T+2, self.node_dim]).to(device)
        for s_i in range(self.step):
            for b in range(B):
                gnn_feat = self.gnn(initial_node_representation=tooth_node_feat[b, s_i], 
                                    adjacency_lists=self.adj_list,
                                    return_all_states=False)
                gnn_local_feat[s_i, b] = gnn_feat

        gnn_local_feat = gnn_local_feat.transpose(0, 2)


        RTV_steps = []
        for i in range(self.num_teeth):
            state_frame = gnn_local_feat[i, :] 
            # target_frame_i = target_frame[i, :].view(B, self.step, self.njoints*self.nfeats) 
            # start_frame_i = start_frame[i, :].view(B, self.step, self.njoints*self.nfeats) 
            # offset = target_frame[i, :].view(B, self.step, self.njoints*self.nfeats) - start_frame[i, :].view(B, self.step, self.njoints*self.nfeats) \
       
            # input_frame = torch.cat([state_frame, start_frame_i, target_frame_i, offset, shape_code[i]], dim=2)
            input_frame = torch.cat([state_frame, shape_code[i]], dim=2)     # MODIFY

            frame_rtv = self.motivation_regression[i](input_frame)
     
            RTV_steps.append(frame_rtv)

        RTV_steps = torch.stack(RTV_steps)  

        return RTV_steps 


class VarianceSchedule(nn.Module):
    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear',)
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        betas = torch.cat([torch.zeros([1]), betas], dim=0) 

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas



class ToothDiffuser(Module):
    def __init__(self,
        num_teeth,
        num_step,
        njoints,
        nfeats,
        shape_dim,
        hidden_size,
        num_layers,
        noise_steps,
        beta_1,
        beta_T,
        sched_mode='linear',
                 ):
        super().__init__()

        self.map_diff = Diffusion(
            net=Planning_Denosier(num_teeth=num_teeth, 
                                  step=num_step, 
                                  njoints=njoints, 
                                  nfeats=nfeats,
                                  shape_dim=shape_dim,
                                  hidden_size = hidden_size,
                                  num_layers = num_layers
                                  ),

            var_sched=VarianceSchedule(
                num_steps=noise_steps,
                beta_1=beta_1,
                beta_T=beta_T,
                mode=sched_mode
            ),

            num_teeth=num_teeth,
            num_step=num_step,
        )


    def get_loss(self, x_0_input, start_pc, shape_code, frame_step_gt,  mask):
        
        loss = self.map_diff.get_loss(x_0_input, start_pc, shape_code, frame_step_gt,  mask)

        return loss

    def get_target(self, shape_code):


        # if not ret_traj:
        #     rtv_pred = self.map_diff.sample(start_frame, ending_frame, shape_code, ret_traj=ret_traj, C=6)

        #     frame_pred = get_pred_frame(rtv_pred, start_frame) 
            
        #     return rtv_pred, frame_pred
        # else:
        #     x_0_pre, x_0_list, x_t_list = self.map_diff.sample(start_frame, ending_frame, shape_code,
        #                                                        ret_traj=ret_traj, C=6, sample_t=sample_t)
        #     frame_pred = get_pred_frame(rtv_pred, start_frame) 

        #     return rtv_pred, frame_pred, x_0_list, x_t_list
        # MODIFY
        rtv_pred = self.map_diff.sample(shape_code, ret_traj=False, C=6)
        
        return rtv_pred



class loss_iteam(nn.Module):
    def __init__(self, num_teeth=28, num_step=21):
        super(loss_iteam, self).__init__()
        self.num_teeth = num_teeth
        self.num_step = num_step
        self.edge_list_1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                            (0, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
                            (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),
                            (14, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27)]


    def forward(self, rtv, rtv_gt, pc_pred, pc_gt, mask):

        # (1) Base loss
        Rloss = F.mse_loss(rtv[:, :, :, :3], rtv_gt[:, :, :, :3])
        Tloss = F.mse_loss(rtv[:, :, :, 3:], rtv_gt[:, :, :, 3:])

        # (2) pc loss
        criterion = nn.MSELoss(reduction='none')
        l = 20
        masks = repeat(mask,'b p -> (b l p)', l=l)   
        rec_loss = criterion(pc_pred, pc_gt)
        rec_loss = (rec_loss * masks.reshape(-1,1,1)).sum(dim=(-1,-2)).mean()
        

        # # (3) pc loss
        # # (3.1) ChamferVector loss between every connecting 2-teeth
        # CVcoloss = 0
        # # (3.2) ChamferVector loss between Upper and Lower teeth
        # CVocloss = 0
        # for i in range(self.num_step):
        #     pc_new_step = pc_step_pred[i]  
        #     pc_gt_step = pc_step_gt[i]

        #     # ChamferVector loss between Upper and Lower teeth
        #     vector_new = ChamferVector(torch.transpose(pc_new_step[:, :3, :5600], 1, 2),
        #                                 torch.transpose(pc_new_step[:, :3, 5600:], 1, 2))
        #     vector_gt = ChamferVector(torch.transpose(pc_gt_step[:, :3, :5600], 1, 2),
        #                                 torch.transpose(pc_gt_step[:, :3, 5600:], 1, 2))
        #     CVocloss_new = F.smooth_l1_loss(vector_new, vector_gt)
        #     CVocloss += CVocloss_new

        #     # moved pointcloud and target pointcloud
        #     pc_new_chunk = torch.chunk(pc_new_step, self.num_teeth, dim=2)
        #     pc_gt_chunk = torch.chunk(pc_gt_step, self.num_teeth, dim=2)
        #     for a, b in self.edge_list_1:
        #         chamfer_v_new = ChamferVector(torch.transpose(pc_new_chunk[a][:, :3], 1, 2),
        #                                     torch.transpose(pc_new_chunk[b][:, :3], 1, 2))
        #         chamfer_v_gt = ChamferVector(torch.transpose(pc_gt_chunk[a][:, :3], 1, 2),
        #                                     torch.transpose(pc_gt_chunk[b][:, :3], 1, 2))
        #         CVcoloss_new = F.smooth_l1_loss(chamfer_v_new, chamfer_v_gt) / len(self.edge_list_1)
        #         CVcoloss += CVcoloss_new

        # CVcoloss = CVcoloss/self.num_step
        # CVocloss = CVocloss/self.num_step

        # # (4) collsion_avoid
        # dist = cal_collision(pc_step_pred_s)
        # collision_loss = F.mse_loss(dist, torch.tensor(0))

        # return [Rloss, Tloss, frame_axis_loss, frame_ct_loss, CVcoloss, CVocloss, collision_loss]
        return 100*Rloss + 100*Tloss + rec_loss


class Diffusion(nn.Module):
    def __init__(self, net, var_sched: VarianceSchedule, num_teeth, num_step):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.num_step = num_step
        self.compute_loss = loss_iteam(num_teeth, num_step)


    def get_loss(self, x_0_input, start_pc, shape_code, frame_step_gt,  mask, t=None):
   
        device = x_0_input.device

        B, T, S, C = x_0_input.shape

        x_0 = x_0_input.transpose(0, 1)  
        if t == None:
            t = self.var_sched.uniform_sample_t(B)
        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  

        e_rand = torch.randn_like(x_0[0])  
        
        tooth_rtv_noise = torch.zeros_like(x_0).to(device)


        for i in range(T):
            noise_image = c0 * x_0[i] + c1 * e_rand
            tooth_rtv_noise[i] = noise_image

        x_0_pre = self.net(tooth_rtv_noise, shape_code)

     
        # frame_pred = get_pred_frame(x_0_pre, start_frame)
        # MODIFY
        pc_pred, pc_gt = get_step_pc(x_0_pre, x_0, start_pc)

        loss = self.compute_loss(x_0_pre, x_0, pc_pred, pc_gt, mask)

        return loss

    def sample(self, shape_code, ret_traj=False, sample_t=None, C=6):

        # B, Tooths, _, _ = start_frame.size()
        Tooths, B, _ = shape_code.shape
        device = shape_code.device

        steps = self.num_step

        x_T = torch.randn([B, steps, C]).to(shape_code.device)
        x_T = x_T.repeat(32, 1, 1, 1)

        x_t = x_T
        if ret_traj:
            x_t_list = {}
            x_0_list = {}
        for t in range(self.var_sched.num_steps, 0, -1):
            x_0_pre = self.net(x_t, shape_code)
            if ret_traj and sample_t == None:
                x_t_list[t] = x_t.cpu()
                x_0_list[t] = x_0_pre.cpu()
            elif ret_traj:
                if t in sample_t:
                    x_t_list[t] = x_t.cpu()
                    x_0_list[t] = x_0_pre.cpu()

            alpha = self.var_sched.alphas[t - 1]
            alpha_bar = self.var_sched.alpha_bars[t - 1]
            c0 = torch.sqrt(alpha_bar).view(-1, 1, 1) 
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  

            e_rand = torch.randn_like(x_t[0])  

            x_next = torch.zeros_like(x_T).to(device)
            for i in range(Tooths):
                x_next[i] = c0 * x_0_pre[i] + c1 * e_rand

            x_t = x_next
        if not ret_traj:
            return x_0_pre
        else:
            return x_0_pre, x_0_list, x_t_list


def get_model(args):
    model = ToothDiffuser(args)
    return model


if __name__ == '__main__':
    
    print('hello world!')

    num_teeth = 28
    batch_size = 14
    steps = 21
    njoints = 4
    nfeats = 3
    num_pc = 400
    shape_dim = 12


    class args():
        def __init__(self):
            super().__init__()
            self.noise_steps = 100
            self.beta_1 = 1e-4
            self.beta_T = 0.05
            self.sched_mode = 'linear'

            self.num_teeth = num_teeth
            self.num_pc = num_pc
            self.num_step = steps
            self.njoints = njoints
            self.nfeats = nfeats
            self.shape_dim = shape_dim
            self.hidden_size = 256
            self.num_layers = 3

    with torch.no_grad():
        model = get_model(args()).cuda()
        model.train()

        # input
        start_frame = torch.rand([batch_size, num_teeth, njoints, nfeats]).cuda()
        start_pc = torch.rand(batch_size, 3, num_pc*num_teeth).cuda()
        ending_frame = torch.rand([batch_size, num_teeth, njoints, nfeats]).cuda()
        rtv_step_gt = torch.rand(batch_size, num_teeth, steps, 6).cuda()
        shape_code = torch.ones([num_teeth, batch_size, shape_dim]).cuda()
        frame_step_gt = torch.rand(batch_size, num_teeth, steps, 4, 3).cuda()
        x_0_input = rtv_step_gt

        loss = model.get_loss(x_0_input, start_frame, ending_frame, start_pc, shape_code, frame_step_gt)

        print('loss runable!')


        
        rtv_pred, frame_pred = model.get_target(start_frame, ending_frame, start_pc, shape_code,
                                                        ret_traj=False, sample_t=None)

        print('model runable!')




    