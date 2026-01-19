import torch
import torch.nn as nn
from einops import rearrange


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=2500, latent_dim=4, grid_size=4):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        # self.final_conv = nn.Sequential(
        #     nn.Conv1d(1024 + 3 + 2, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 3, 1)
        # )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, latent_dim, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, latent_dim)

        # print(feature_global.shape)
        latent_feature = feature_global
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        # print('coarse',coarse.shape)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)
        

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
        
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return latent_feature, coarse.contiguous(), fine.transpose(1, 2).contiguous()


class get_model(nn.Module):
    def __init__(self, latent_dim, num_points, num_teeth):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.num_teeth = num_teeth

        re_pc_list = []
        for i in range(self.num_teeth):
            autoencoder_layer = PCN(num_dense=num_points, latent_dim=latent_dim, grid_size=4)#.cuda()
            # autoencoder_layer = AutoEncoder(latent_dim=latent_dim, num_points = num_points, num_teeth = num_teeth)
            re_pc_list.append(autoencoder_layer)
        self.re_pc_list = nn.ModuleList(re_pc_list)

    def forward(self, point_clouds):
        # input: [B, teeth_nums*400, 3]
        # output: [B, teeth_nums*400, 3]
        point_clouds = rearrange(point_clouds,'b p n c -> b (p n) c')
        B, N, C = point_clouds.size()
        # teeth_pc 28*(B*6*400)
        teeth_pc = torch.chunk(point_clouds, self.num_teeth, dim=1)

        code_list = torch.zeros([self.num_teeth, B, self.latent_dim])
        re_pc_teeth_coarse = []
        re_pc_teeth_dense = []
        for i in range(self.num_teeth):

            latent_code, coarse_pred, dense_pred = self.re_pc_list[i](teeth_pc[i])

            re_pc_teeth_coarse.append(coarse_pred)
            re_pc_teeth_dense.append(dense_pred)

            code_list[i,] = latent_code

        re_pc_teeth_coarse = torch.cat(re_pc_teeth_coarse, dim=1)
        re_pc_teeth_dense = torch.cat(re_pc_teeth_dense, dim=1)

        return code_list, re_pc_teeth_coarse, re_pc_teeth_dense

def ChamferDistanceloss(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :return: sum of Chamfer Distance of two point sets
    '''

    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff * diff, dim=3)
 
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist1, dim=2)
    dist_min2, _ = torch.min(dist2, dim=2)

    loss = (torch.sum(dist_min1) / (p1.shape[1]) + torch.sum(dist_min2) / (p2.shape[1])) / (p1.shape[0])
    return loss

class get_loss(nn.Module):
    def __init__(self, num_teeth=28, num_pc=400):
        super(get_loss, self).__init__()
        self.num_teeth = num_teeth
        self.num_pc = num_pc
       
    def forward(self, pc_new, pc_gt):
        # moved pointcloud and target pointcloud
        pc_new_chunk = torch.chunk(pc_new.transpose(1, 2), self.num_teeth, dim=2)
        pc_gt_chunk = torch.chunk(pc_gt.transpose(1, 2), self.num_teeth, dim=2)


        # # ChamferDistance loss between every teeth
        CDloss = 0
        for i in range(self.num_teeth):
            tnew = torch.transpose(pc_new_chunk[i][:, :3], 1, 2)
            tgt = torch.transpose(pc_gt_chunk[i][:, :3], 1, 2)
            CDloss_new = ChamferDistanceloss(tnew, tgt)
            # CDloss_new = Motivationloss(tnew, tgt)
            CDloss += CDloss_new
        CDloss = CDloss / self.num_teeth

        return CDloss

class get_loss_end(nn.Module):
    def __init__(self, num_teeth=28, num_pc=400, num_pc_coarse=25):
        super(get_loss_end, self).__init__()
        self.num_teeth = num_teeth
        self.num_pc = num_pc
        self.num_pc_coarse = num_pc_coarse
        self.loss_coarse = get_loss(num_pc=num_pc_coarse).cuda()
        self.loss_dense = get_loss(num_pc=num_pc).cuda()
       
    def forward(self, pc_new_coarse, pc_gt_coarse, pc_new, pc_gt):

        CDloss_coarse = self.loss_coarse(pc_new_coarse, pc_gt_coarse)
        CDloss_dense = self.loss_dense(pc_new, pc_gt)

        
        return [CDloss_coarse, CDloss_dense]

class get_loss_end(nn.Module):
    def __init__(self, num_teeth=28, num_pc=400, num_pc_coarse=25):
        super(get_loss_end, self).__init__()
        self.num_teeth = num_teeth
        self.num_pc = num_pc
        self.num_pc_coarse = num_pc_coarse
        self.loss_coarse = get_loss(num_pc=num_pc_coarse).cuda()
        self.loss_dense = get_loss(num_pc=num_pc).cuda()
       
    def forward(self, pc_new_coarse, pc_gt_coarse, pc_new, pc_gt):

        CDloss_coarse = self.loss_coarse(pc_new_coarse, pc_gt_coarse)
        CDloss_dense = self.loss_dense(pc_new, pc_gt)

        
        return [CDloss_coarse, CDloss_dense]


def getAccuracy(pc_new_coarse, pc_gt_coarse, pc_new, pc_gt):
       
    loss_coarse = get_loss(num_pc=25)
    loss_dense = get_loss(num_pc=400)

    CDloss_coarse = loss_coarse(pc_new_coarse, pc_gt_coarse)
    CDloss_dense = loss_dense(pc_new, pc_gt)


    
    return [CDloss_coarse, CDloss_dense]
        
