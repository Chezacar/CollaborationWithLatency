from numpy.core.fromnumeric import shape
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.model import STPN, STPN_KD, MapExtractor, forecast_lstm, lidar_encoder, lidar_decoder, lidar_decoder_kd, conv2DBatchNormRelu, Sparsemax, adafusionlayer,sigmoidfusionlayer, pairfusionlayer,pairfusionlayer_1, pairfusionlayer_2, pairfusionlayer_3 ,pairfusionlayer_4
import numpy as np
import copy
import torchgeometry as tgm
from matplotlib import pyplot as plt

class ClassificationHead(nn.Module):

    def __init__(self, config):

        super(ClassificationHead, self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, category_num*anchor_num_per_loc, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x

class MotionStateHead(nn.Module):

    def __init__(self, config):

        super(MotionStateHead, self).__init__()
        category_num = 3 #ignore: 0 static: 1 moving: 2
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, category_num*anchor_num_per_loc, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x

class FeatEncoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(FeatEncoder, self).__init__()
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.stpn(bevs)

        return x

class RegressionHead(nn.Module):
    def __init__(self,config):
        super(RegressionHead,self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size

        self.box_prediction = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, anchor_num_per_loc * box_code_size, kernel_size=1, stride=1, padding=0))

    def forward(self,x):
        box = self.box_prediction(x)
        return x

class SingleRegressionHead(nn.Module):
    def __init__(self,config):
        super(SingleRegressionHead,self).__init__()
        category_num = config.category_num
        channel = 32
        if config.use_map:
            channel += 6
        if config.use_vis:
            channel += 13

        anchor_num_per_loc = len(config.anchor_size)
        box_code_size = config.box_code_size
        if config.only_det:
            out_seq_len = 1
        else:
            out_seq_len = config.pred_len
    
        if config.binary:
            if config.only_det:
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
                    nn.Conv2d(channel, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1, padding=0))        
            else:      
                self.box_prediction = nn.Sequential(
                    nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, anchor_num_per_loc * box_code_size * out_seq_len, kernel_size=1, stride=1, padding=0))
                

    def forward(self,x):
        box = self.box_prediction(x)

        return box

class FaFMGDA(nn.Module):

    def __init__(self, config):
        super(FaFMGDA, self).__init__()
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.motion_state = config.motion_state


        self.box_code_size = config.box_code_size
        self.category_num = config.category_num

        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        #self.RegressionList = nn.ModuleList([RegressionHead for i in range(seq_len)])
        self.regression = SingleRegressionHead(config)
        if self.motion_state:
            self.motion_cls = MotionStateHead(config)


    def forward(self, x):
        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)


        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)

        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)          
            result['state'] = motion_cls_preds
            
        return result

class FaFNet(nn.Module):
    def __init__(self, config):
        super(FaFNet, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        #self.RegressionList = nn.ModuleList([RegressionHead for i in range(seq_len)])
        self.regression = SingleRegressionHead(config)
        #self.fusion_method = config.fusion_method

        # if self.use_map:
        #     if self.fusion_method == 'early_fusion':
        #         self.stpn = STPN(height_feat_size=config.map_dims[2]+config.map_channel)
        #     elif self.fusion_method == 'middle_fusion':
        #         self.stpn = STPN(height_feat_size=config.map_dims[2],use_map=True)
        #     elif self.fusion_method == 'late_fusion':
        #         self.map_encoder = MapExtractor(map_channel=config.map_channel)
        #         self.stpn = STPN(height_feat_size=config.map_dims[2])
        # else:
        self.stpn = STPN_KD(height_feat_size=config.map_dims[2])
        
        if self.motion_state:
            self.motion_cls = MotionStateHead(config)


    def forward(self, bevs, maps=None,vis=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        # vis = vis.permute(0, 3, 1, 2)
        x_8, x_7, x_6, x_5, x_3 = self.stpn(bevs)
        return x_8, x_7, x_6, x_5, x_3

        # x = x_8
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        # cls_preds = self.classification(x)
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        # cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)
        #
        #
        # # Detection head
        # loc_preds =self.regression(x)
        # loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        # loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        # #loc_pred (N * T * W * H * loc)
        # result = {'loc': loc_preds,
        #           'cls': cls_preds}
        #
        # #MotionState head
        # if self.motion_state:
        #     motion_cat = 3
        #     motion_cls_preds = self.motion_cls(x)
        #     motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
        #     motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
        #     result['state'] = motion_cls_preds
        # return result, x_8, x_7, x_6, x_5

'''''''''''''''''''''''''''''''''''''''''''''''''''
Online warp of layer 4, Yiming Li, April, 15, 2021
        
'''''''''''''''''''''''''''''''''''''''''''''''''''
class FaFMIMONet_512_16_16(nn.Module):
    def __init__(self, config, n_classes=21, in_channels=13, feat_channel=512, feat_squeezer=-1, attention='additive',
                 has_query=True, sparse=False, agent_num=5, shuffle_flag=False, image_size=512,
                 shared_img_encoder='unified', key_size=1024, query_size=128):
        super(FaFMIMONet_512_16_16, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)

        self.sparse = sparse
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)


    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, vis=None, training=True, MO_flag=True, inference='activated', batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)

        '''''''''''''''''''''''''''''''''''''''''''''''''''
        Visualization debugging of online warp, need to firstly rotate, then translate using pytorch grid

        padded_voxel_points_global = torch.squeeze(padded_voxel_points_global).cpu().numpy()
        plt.clf()
        plt.xlim(0, 768)
        plt.ylim(0, 768)

        plt.imshow(np.max(padded_voxel_points_global, axis=2), alpha=1.0, zorder=12)
        plt.pause(0.1)
        plt.show()

        tmp0 = torch.squeeze(bevs[0]).cpu().numpy()
        tmp1 = torch.squeeze(bevs[1]).cpu().numpy()
        # tmp0 = np.flip(tmp0, axis=1)

        # pass encoder
        plt.clf()
        plt.xlim(0, 256)
        plt.ylim(0, 256)

        plt.imshow(np.max(tmp0, axis=0), origin='lower', alpha=1.0, zorder=12)
        plt.pause(0.1)
        plt.show()

        plt.imshow(np.max(tmp1, axis=0), origin='lower', alpha=1.0, zorder=12)
        plt.pause(0.1)
        plt.show()

        device = bevs.device
        size = (1, 13, 256, 256)
        nb_warp = trans_matrices[0, 0, 1]
        nb_agent = torch.unsqueeze(torch.squeeze(bevs[1]), 0)

        x_trans = (4*nb_warp[0, 3])/128 #左+右-
        y_trans = -(4*nb_warp[1, 3])/128
        # z_trans = (4*nb_warp[2, 3])/128

        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
        theta_trans = torch.unsqueeze(theta_trans, 0)
        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
        theta_rot = torch.unsqueeze(theta_rot, 0)
        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

        #first rotate the feature map, then translate it
        warp_img_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
        warp_img_trans = F.grid_sample(warp_img_rot, grid_trans, mode='bilinear')
        warp_img = warp_img_trans

        nb2tg_map_vis = torch.squeeze(warp_img).cpu().numpy()

        # visualize the warped feature map
        plt.imshow(np.max(nb2tg_map_vis, axis=0), origin='lower', alpha=1.0, zorder=12)
        plt.pause(0.1)
        plt.show()

        plt.imshow(np.max(nb2tg_map_vis + tmp0, axis=0), origin='lower', alpha=1.0, zorder=12)
        plt.pause(0.1)
        plt.show()
        '''''''''''''''''''''''''''''''''''''''''''''''''''

        x,x_1,x_2,x_3,feat_maps = self.u_encoder(bevs)
        device = bevs.device
        size_16 = (1, 512, 16, 16)
        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size_16))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size_16))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)

                        tg_agent = tg_agent + warp_feat.type(dtype=torch.float32)

                local_com_mat_update[b, i] = tg_agent

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        x = self.decoder(x,x_1,x_2,x_3,feat_fuse_mat,batch_size)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result

# with knowledge distillation
class FaFMIMONet_512_16_16_KD(nn.Module):
    def __init__(self, config, n_classes=21, in_channels=13, feat_channel=512, feat_squeezer=-1, attention='additive',
                 has_query=True, sparse=False, agent_num=5, shuffle_flag=False, image_size=512,
                 shared_img_encoder='unified', key_size=1024, query_size=128):
        super(FaFMIMONet_512_16_16_KD, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)

        self.sparse = sparse
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder

        # Detection decoder
        self.decoder = lidar_decoder_kd(height_feat_size=in_channels)


    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, vis=None, training=True, MO_flag=True, inference='activated', batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)
        x,x_1,x_2,x_3,feat_maps = self.u_encoder(bevs)
        device = bevs.device
        size_16 = (1, 512, 16, 16)
        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size_16))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size_16))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)

                        tg_agent = tg_agent + warp_feat.type(dtype=torch.float32)

                local_com_mat_update[b, i] = tg_agent

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        x_8, x_7, x_6, x_5 = self.decoder(x, x_1, x_2, x_3, feat_fuse_mat, batch_size)
        x = x_8
        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result, x_8, x_7, x_6, x_5

'''''''''''''''''''''''''''''''''''''''''''''''''''
            Online warp of layer 3
        
'''''''''''''''''''''''''''''''''''''''''''''''''''
class FaFMIMONet_256_32_32(nn.Module):
    def __init__(self, config, in_channels=13, shared_img_encoder='unified'):
        super(FaFMIMONet_256_32_32, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder
        self.forecast =forecast_lstm()
        self.adafusion = pairfusionlayer_3(input_channel=256)


        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)
    

    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def trans2center_now(self, device, bevs, trans2center_now, batch_size,num_agent_tensor, center_agent = 0):
        bevs_update = bevs
        size = bevs[0,0].shape
        for b in range(batch_size):
            try: 
                center_agent_int = int(center_agent[b])
            except:
                center_agent_int = center_agent
            num_agent = int(num_agent_tensor[b, center_agent_int])
            if num_agent == 0:
                break
            i = int(center_agent_int)
            # for i in range(num_agent):
            # tg_agent = []
            # tg_agent.append(local_com_mat[b, i])
            # all_warp = trans_matrices[b, i, -1] # transformation [2 5 5 4 4]

            for j in range(num_agent):
                for k in range(len(trans2center_now[0])):
                    if j != i or k == (len(trans2center_now[0]) - 1):
                        # nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_agent = bevs[j*batch_size + b][k]
                        # nb_warp = all_warp[j] # [4 4]
                        nb_warp = trans2center_now[j][k][b]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128
                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample
                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample
                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)
                        # tg_agent.append(warp_feat.type(dtype=torch.float32))
                        bevs_update [[j*batch_size + b]] = warp_feat.type(dtype=torch.float32)
        imgs = [bevs[i * 2,3,0] for i in range(4)]
        for i in range(len(imgs)):
            imgs[i] = np.array(imgs[i].cpu())
            imgs[i] = imgs[i].transpose((1,2,0))
            imgs[i] = imgs[i].sum(axis = 2)
            print(np.max(imgs[i]))
            imgs[i] /= np.max(imgs[i])
            imgs[i] *= 255
            imgs[i] = imgs[i].astype(int)
        np.save('./test_1.npy', np.array(imgs))
        # plt.subplot(221); plt.imshow(imgs[0], cmap ='gray')
        # plt.subplot(222); plt.imshow(imgs[1], cmap ='gray')
        # plt.subplot(223); plt.imshow(imgs[2], cmap ='gray')
        # plt.subplot(224); plt.imshow(imgs[3], cmap ='gray')
        # plt.show()
        return bevs_update 
        
    def forward(self, bevs, trans_matrices, num_agent_tensor, to_new_trans_mat_list, vis=None, training=True, MO_flag=True, inference='activated', batch_size=2, center_agent = 0, delta_t = [0,10,10,10,10], rank = 0):
        device = bevs.device
        # batch_size /= torch.cuda.device_count()
        # batch_size = int(batch_size)
        bevs = bevs.permute(0, 1, 2, 5, 3, 4) # (Batch, seq, z, h, w)
        bevs = self.trans2center_now(device, bevs, to_new_trans_mat_list, batch_size, num_agent_tensor, center_agent)
        shape_a,shape_b,shape_c,shape_d,shape_e,shape_f, = bevs.shape
        # bevs_new = bevs.reshape((shape_a * shape_b , shape_c, shape_d, shape_e, shape_f))
        x,x_1,x_2,x_3,x_4 = self.u_encoder(bevs[:,-1])

        # for iter1 in range(len(delta_t)):
        #     x_feature_list_1 = []
        #     for iter2 in range(batch_size):
        #         x_feature_list_2 = []
        #         for iter3 in range(len(shape_b)):
        #             x_feature_list_2.append()

        # for i in range(bevs.shape[0]):
        #     batch = int(i / len(delta_t[0]))
        #     inbatch = int(i % len(delta_t[0]))
        #     x_temp, x_1_temp, x_2_temp, x_3_temp, x_4_temp = self.u_encoder(bevs[i])
        #     if delta_t[batch][inbatch] > 0:
        #         x_feature_list.append(self.forecast(x_3_temp, delta_t[batch][inbatch]))
        #     else:
        #         x_feature_list.append(torch.unsqueeze(x_3_temp[-1], 0))
        x_feature_list = []
        for batch in range(batch_size):
            for inbatch in range(len(delta_t[0])):
                x_temp, x_1_temp, x_2_temp, x_3_temp, x_4_temp = self.u_encoder(bevs[batch * len(delta_t[0]) + inbatch])
                if delta_t[batch][inbatch] > 0:
                    
                    x_feature_list.append(self.forecast(x_3_temp, delta_t[batch][inbatch]))
                else:
                    x_feature_list.append(torch.unsqueeze(x_3_temp[-1], 0))
        x_3 = torch.cat(tuple(x_feature_list), 0)
        device = bevs.device
        size = (1, 256, 32, 32)
        padding_feat = torch.zeros((256,32,32)).cuda()


        feat_maps = x_3
        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        # for b in range(batch_size):
        #     try: 
        #         center_agent_int = int(center_agent[b])
        #     except:
        #         center_agent_int = center_agent
        #     num_agent = int(num_agent_tensor[b, center_agent_int])
        #     if num_agent == 0:
        #         break
        #     i = int(center_agent_int)
        #     # for i in range(num_agent):
        #     tg_agent = []
        #     tg_agent.append(local_com_mat[b, i])
        #     all_warp = trans_matrices[b, i, -1] # transformation [2 5 5 4 4]
        #     for j in range(num_agent):
        #         if j != i:
        #             nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
        #             nb_warp = all_warp[j] # [4 4]
        #             # normalize the translation vector
        #             x_trans = (4*nb_warp[0, 3])/128
        #             y_trans = -(4*nb_warp[1, 3])/128
        #             theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
        #             theta_rot = torch.unsqueeze(theta_rot, 0)
        #             grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample
        #             theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
        #             theta_trans = torch.unsqueeze(theta_trans, 0)
        #             grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample
        #             #first rotate the feature map, then translate it
        #             warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
        #             warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
        #             warp_feat = torch.squeeze(warp_feat_trans)
        #             tg_agent.append(warp_feat.type(dtype=torch.float32))

        #         # for k in range(5-num_agent):
        #         #     tg_agent.append(padding_feat)

        #     tg_agent=torch.stack(tg_agent)
            # tg_agent = self.adafusion(tg_agent, rank)

            # local_com_mat_update[b, i] = tg_agent

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        x = self.decoder(x,x_1,x_2,feat_fuse_mat,x_4,batch_size)
        x = torch.stack([x[i * self.agent_num] for i in range(batch_size)])
        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result

# with knowledge distillation
class FaFMIMONet_256_32_32_KD(nn.Module):
    def __init__(self, config, in_channels=13, shared_img_encoder='unified'):
        super(FaFMIMONet_256_32_32_KD, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder

        self.adafusion = pairfusionlayer_3(input_channel=256)

        # Detection decoder
        self.decoder = lidar_decoder_kd(height_feat_size=in_channels)


    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, vis=None, training=True, MO_flag=True, inference='activated', batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)

        x,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device
        size = (1, 256, 32, 32)
        padding_feat = torch.zeros((256,32,32)).cuda()


        feat_maps = x_3
        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = []
                tg_agent.append(local_com_mat[b, i])
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)

                        tg_agent.append(warp_feat.type(dtype=torch.float32))

                # for k in range(5-num_agent):
                #     tg_agent.append(padding_feat)

                tg_agent=torch.stack(tg_agent)
                tg_agent = self.adafusion(tg_agent)

                # local_com_mat_update[b, i] = tg_agent[0]
                local_com_mat_update[b, i] = tg_agent

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        x_8, x_7, x_6, x_5 = self.decoder(x, x_1, x_2, feat_fuse_mat, x_4, batch_size)
        x = x_8

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result, x_8, x_7, x_6, x_5, feat_fuse_mat


'''''''''''''''''''''''''''''''''''''''''''''''''''
            Online warp of layer 2
        
'''''''''''''''''''''''''''''''''''''''''''''''''''
class FaFMIMONet_128_64_64(nn.Module):
    def __init__(self, config, in_channels=13, shared_img_encoder='unified'):
        super(FaFMIMONet_128_64_64, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder

        self.adafusion = adafusionlayer(input_channel=128)

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)


    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, vis=None,  batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)

        x,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device
        size = (1, 128, 64, 64)
        padding_feat = torch.zeros((128,64,64)).cuda()
        # print('padding size:{}'.format(padding_feat.size()))

        feat_maps = x_2
        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = []
                tg_agent.append(local_com_mat[b, i])
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)

                        tg_agent.append(warp_feat.type(dtype=torch.float32))

                for k in range(5-num_agent):
                    tg_agent.append(padding_feat)

                tg_agent=torch.stack(tg_agent)
                tg_agent = self.adafusion(tg_agent)
                local_com_mat_update[b, i] = tg_agent

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        x = self.decoder(x,x_1,feat_fuse_mat,x_3,x_4,batch_size)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result

# with knowledge distillation
class FaFMIMONet_128_64_64_KD(nn.Module):
    def __init__(self, config, in_channels=13, shared_img_encoder='unified'):
        super(FaFMIMONet_128_64_64_KD, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder

        self.adafusion = adafusionlayer(input_channel=128)

        # Detection decoder
        self.decoder = lidar_decoder_kd(height_feat_size=in_channels)


    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, vis=None, training=True, MO_flag=True, inference='activated', batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)

        x,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device
        size = (1, 128, 64, 64)
        padding_feat = torch.zeros((128,64,64)).cuda()


        feat_maps = x_2
        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = []
                tg_agent.append(local_com_mat[b, i])
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)

                        tg_agent.append(warp_feat.type(dtype=torch.float32))

                for k in range(5-num_agent):
                    tg_agent.append(padding_feat)

                tg_agent=torch.stack(tg_agent)
                tg_agent = self.adafusion(tg_agent)

                local_com_mat_update[b, i] = tg_agent

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        x_8, x_7, x_6, x_5 = self.decoder(x, x_1, feat_fuse_mat, x_3, x_4, batch_size)
        x = x_8

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result, x_8, x_7, x_6, x_5


'''''''''''''''''''''''''''''''''''''''''''''''''''
            Online warp of layer 1

'''''''''''''''''''''''''''''''''''''''''''''''''''
class FaFMIMONet_64_128_128(nn.Module):
    def __init__(self, config, in_channels=13, shared_img_encoder='unified'):
        super(FaFMIMONet_64_128_128, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder
        self.adafusion = adafusionlayer(input_channel=128)


        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)


    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, vis=None, training=True, MO_flag=True, inference='activated', batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)

        x,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device
        size = (1, 64, 128, 128)
        padding_feat = torch.zeros((128,64,64)).cuda()


        feat_maps = x_1
        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = []
                tg_agent.append(local_com_mat[b, i])
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)

                        tg_agent = tg_agent + warp_feat.type(dtype=torch.float32)

                local_com_mat_update[b, i] = tg_agent

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        x = self.decoder(x,feat_fuse_mat,x_2,x_3,x_4,batch_size)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result


'''''''''''''''''''''''''''''''''''''''''''''''''''
            Online warp of layer 0
        
'''''''''''''''''''''''''''''''''''''''''''''''''''
class FaFMIMONet_32_256_256(nn.Module):
    def __init__(self, config, in_channels=13, shared_img_encoder='unified'):
        super(FaFMIMONet_32_256_256, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)
        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)


    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, vis=None, training=True, MO_flag=True, inference='activated', batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)

        x,x_1,x_2,x_3,x_4 = self.u_encoder(bevs)
        device = bevs.device
        size = (1, 32, 256, 256)
        feat_maps = x
        # print(feat_maps.shape, x_3.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # 得到grid 用于grid sample

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # 得到grid 用于grid sample

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)

                        tg_agent = tg_agent + warp_feat.type(dtype=torch.float32)

                local_com_mat_update[b, i] = tg_agent

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        x = self.decoder(feat_fuse_mat,x_1,x_2,x_3,x_4,batch_size)

        # vis = vis.permute(0, 3, 1, 2)
        # if not maps is None:
        #     x = torch.cat([x,maps],axis=-1)
        # if not vis is None:
        #     x = torch.cat([x,vis],axis=1)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result


'''''''''''''''''''''''''''''''''''''''''''''''''''
            Online warp of layer 3 and 4

'''''''''''''''''''''''''''''''''''''''''''''''''''
class FaFMIMONet_layer_3_and_4(nn.Module):
    def __init__(self, config, in_channels=13, shared_img_encoder='unified'):
        super(FaFMIMONet_layer_3_and_4, self).__init__()
        self.motion_state = config.motion_state
        if config.only_det:
            self.out_seq_len = 1
        else:
            self.out_seq_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.category_num = config.category_num
        self.use_map = config.use_map
        self.anchor_num_per_loc = len(config.anchor_size)
        self.classification = ClassificationHead(config)
        self.regression = SingleRegressionHead(config)

        self.u_encoder = lidar_encoder(height_feat_size=in_channels)
        self.agent_num = 5
        self.shared_img_encoder = shared_img_encoder

        # Detection decoder
        self.decoder = lidar_decoder(height_feat_size=in_channels)


    def agents2batch(self, feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1):

        bevs = bevs.permute(0, 1, 4, 2, 3) # (Batch, seq, z, h, w)

        x,x_1,x_2,feat_maps_32,feat_maps = self.u_encoder(bevs)

        device = bevs.device
        size_16 = (1, 512, 16, 16)
        size_32 = (1, 256, 32, 32)
        # print(feat_maps.shape, feat_maps_32.shape, x_2.shape, x_1.shape)

        # get feat maps for each agent [10 512 16 16] -> [2 5 512 16 16]
        feat_map = {}
        feat_list = []
        feat_map_32 = {}
        feat_list_32 = []

        for i in range(self.agent_num):
            feat_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            feat_list.append(feat_map[i])

            feat_map_32[i] = torch.unsqueeze(feat_maps_32[batch_size * i:batch_size * (i + 1)], 1)
            feat_list_32.append(feat_map_32[i])

        local_com_mat = torch.cat(tuple(feat_list), 1) # [2 5 512 16 16] [batch, agent, channel, height, width]
        local_com_mat_update = torch.cat(tuple(feat_list), 1) # to avoid the inplace operation

        local_com_mat_32 = torch.cat(tuple(feat_list_32), 1) # [2 5 256 32 32] [batch, agent, channel, height, width]
        local_com_mat_update_32 = torch.cat(tuple(feat_list_32), 1) # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                tg_agent = local_com_mat[b, i]
                tg_agent_32 = local_com_mat_32[b, i]

                all_warp = trans_matrices[b, i] # transformation [2 5 5 4 4]
                for j in range(num_agent):
                    if j != i:
                        nb_agent = torch.unsqueeze(local_com_mat[b, j], 0) # [1 512 16 16]
                        nb_agent_32 = torch.unsqueeze(local_com_mat_32[b, j], 0)  # [1 512 16 16]

                        nb_warp = all_warp[j] # [4 4]
                        # normalize the translation vector
                        x_trans = (4*nb_warp[0, 3])/128
                        y_trans = -(4*nb_warp[1, 3])/128

                        theta_rot = torch.tensor([[nb_warp[0,0], nb_warp[0,1], 0.0], [nb_warp[1,0], nb_warp[1,1], 0.0]]).type(dtype=torch.float).to(device)
                        theta_rot = torch.unsqueeze(theta_rot, 0)
                        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size_16))  # 得到grid 用于grid sample
                        grid_rot_32 = F.affine_grid(theta_rot, size=torch.Size(size_32))

                        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(device)
                        theta_trans = torch.unsqueeze(theta_trans, 0)
                        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size_16))  # 得到grid 用于grid sample
                        grid_trans_32 = F.affine_grid(theta_trans, size=torch.Size(size_32))

                        #first rotate the feature map, then translate it
                        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='bilinear')
                        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear')
                        warp_feat = torch.squeeze(warp_feat_trans)

                        warp_feat_rot_32 = F.grid_sample(nb_agent_32, grid_rot_32)
                        warp_feat_trans_32 = F.grid_sample(warp_feat_rot_32, grid_trans_32)
                        warp_feat_32 = torch.squeeze(warp_feat_trans_32)

                        tg_agent = tg_agent + warp_feat.type(dtype=torch.float32)
                        tg_agent_32 = tg_agent_32 + warp_feat_32.type(dtype=torch.float32)

                local_com_mat_update[b, i] = tg_agent
                local_com_mat_update_32[b, i] = tg_agent_32

        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents2batch(local_com_mat_update)
        feat_fuse_mat_32 = self.agents2batch(local_com_mat_update_32)

        x = self.decoder(x,x_1,x_2, feat_fuse_mat_32, feat_fuse_mat, batch_size)

        # Cell Classification head
        cls_preds = self.classification(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(cls_preds.shape[0],-1,self.category_num)

        # Detection head
        loc_preds =self.regression(x)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(-1,loc_preds.size(1),loc_preds.size(2),self.anchor_num_per_loc,self.out_seq_len,self.box_code_size)
        #loc_pred (N * T * W * H * loc)
        result = {'loc': loc_preds,
                  'cls': cls_preds}

        #MotionState head
        if self.motion_state:
            motion_cat = 3
            motion_cls_preds = self.motion_cls(x)
            motion_cls_preds = motion_cls_preds.permute(0, 2, 3, 1).contiguous()
            motion_cls_preds = motion_cls_preds.view(cls_preds.shape[0],-1,motion_cat)
            result['state'] = motion_cls_preds
        return result