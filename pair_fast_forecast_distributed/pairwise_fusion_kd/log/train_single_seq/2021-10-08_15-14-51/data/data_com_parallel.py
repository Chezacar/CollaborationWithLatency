import concurrent
from nuscenes.utils.data_classes import LidarPointCloud
from numpy.lib.npyio import load
from data.obj_util import *
from torch.utils.data import Dataset
import numpy as np
import os
import warnings
import torch
import torch.multiprocessing
from multiprocessing import Manager
from data.config_com import Config, ConfigGlobal
from matplotlib import pyplot as plt
import cv2
from concurrent import futures
from nuscenes.utils.geometry_utils import view_points, transform_matrix

def process_one(para):
    [key, agent, dict] = para
    x = torch.stack(tuple(torch.Tensor(dict[agent][key])),0)
    return x

class NuscenesDataset(Dataset):
    def __init__(self, dataset_root=None, config=None,split=None,cache_size=10000,val=False):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        category_num: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if split is None:
            self.split = config.split
        else:
            self.split = split
        self.voxel_size = config.voxel_size
        self.area_extents = config.area_extents
        self.category_num = config.category_num
        self.future_frame_skip = config.future_frame_skip
        self.pred_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.anchor_size = config.anchor_size
        self.val = val
        self.only_det = config.only_det
        self.binary = config.binary
        self.config = config
        self.use_vis = config.use_vis
        #dataset_root = dataset_root + '/'+split
        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(self.split))
        self.dataset_root = dataset_root
        seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                    if os.path.isdir(os.path.join(self.dataset_root, d))]
        seq_dirs = sorted(seq_dirs)
        self.seq_files = [os.path.join(seq_dir, f) for seq_dir in seq_dirs for f in os.listdir(seq_dir)
                      if os.path.isfile(os.path.join(seq_dir, f))]


        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(self.split, self.num_sample_seqs))

        '''
        # For training, the size of dataset should be 17065 * 2; for validation: 1623; for testing: 4309
        if split == 'train' and self.num_sample_seqs != 17065 * 2:
            warnings.warn(">> The size of training dataset is not 17065 * 2.\n")
        elif split == 'val' and self.num_sample_seqs != 1623:
            warnings.warn(">> The size of validation dataset is not 1719.\n")
        elif split == 'test' and self.num_sample_seqs != 4309:
            warnings.warn('>> The size of test dataset is not 4309.\n')
        '''

        #object information
        self.anchors_map = init_anchors_no_check(self.area_extents,self.voxel_size,self.box_code_size,self.anchor_size)
        self.map_dims = [int((self.area_extents[0][1]-self.area_extents[0][0])/self.voxel_size[0]),\
                         int((self.area_extents[1][1]-self.area_extents[1][0])/self.voxel_size[1])]
        self.reg_target_shape = (self.map_dims[0],self.map_dims[1],len(self.anchor_size),self.pred_len,self.box_code_size)
        self.label_shape = (self.map_dims[0],self.map_dims[1],len(self.anchor_size))
        self.label_one_hot_shape = (self.map_dims[0],self.map_dims[1],len(self.anchor_size),self.category_num)
        self.dims = config.map_dims
        self.num_past_pcs = config.num_past_pcs
        # manager = Manager()
        # self.cache = manager.dict()
        self.cache = {}
        # self.cache_size = cache_size if split == 'train' else 0
        self.cache_size = cache_size

    def __len__(self):
        return self.num_sample_seqs


    def get_one_hot(self,label,category_num):
        one_hot_label = np.zeros((label.shape[0],category_num))
        for i in range(label.shape[0]):
                    one_hot_label[i][label[i]] = 1

        return one_hot_label
        
    def __getitem__(self, idx):
        if idx in self.cache:
            gt_dict = self.cache[idx]
        else:
            seq_file = self.seq_files[idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            gt_dict = gt_data_handle.item()
            if len(self.cache) < self.cache_size:
                self.cache[idx] = gt_dict

        allocation_mask = gt_dict['allocation_mask'].astype(np.bool)
        reg_loss_mask = gt_dict['reg_loss_mask'].astype(np.bool)
        gt_max_iou = gt_dict['gt_max_iou']
        motion_one_hot = np.zeros(5)
        motion_mask = np.zeros(5)

        #load regression target
        reg_target_sparse = gt_dict['reg_target_sparse']
        #need to be modified Yiqi , only use reg_target and allocation_map
        reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)

        reg_target[allocation_mask] = reg_target_sparse
        reg_target[np.bitwise_not(reg_loss_mask)] = 0
        label_sparse = gt_dict['label_sparse']

        one_hot_label_sparse = self.get_one_hot(label_sparse,self.category_num)
        label_one_hot = np.zeros(self.label_one_hot_shape)
        label_one_hot[:,:,:,0] = 1
        label_one_hot[allocation_mask] = one_hot_label_sparse

        if self.config.motion_state:
            motion_sparse = gt_dict['motion_state']
            motion_one_hot_label_sparse = self.get_one_hot(motion_sparse,3)
            motion_one_hot = np.zeros(self.label_one_hot_shape[:-1]+(3,))         
            motion_one_hot[:,:,:,0] = 1
            motion_one_hot[allocation_mask] = motion_one_hot_label_sparse
            motion_mask = (motion_one_hot[:,:,:,2] == 1)

        if self.only_det:
            reg_target = reg_target[:,:,:,:1]
            reg_loss_mask = reg_loss_mask[:,:,:,:1]

        #only center for pred

        elif self.config.pred_type in ['motion','center']:
            reg_loss_mask = np.expand_dims(reg_loss_mask,axis=-1)
            reg_loss_mask = np.repeat(reg_loss_mask,self.box_code_size,axis=-1)
            reg_loss_mask[:,:,:,1:,2:]=False

        if self.config.use_map:
            if ('map_allocation_0' in gt_dict.keys()) or ('map_allocation' in gt_dict.keys()):
                semantic_maps = []
                for m_id in range(self.config.map_channel):
                    map_alloc = gt_dict['map_allocation_'+str(m_id)]
                    map_sparse = gt_dict['map_sparse_'+str(m_id)]
                    recover = np.zeros(tuple(self.config.map_dims[:2]))
                    recover[map_alloc] = map_sparse
                    recover = np.rot90(recover,3)
                    #recover_map = cv2.resize(recover,(self.config.map_dims[0],self.config.map_dims[1]))
                    semantic_maps.append(recover)
                semantic_maps = np.asarray(semantic_maps)
        else:
            semantic_maps = np.zeros(0)
        '''
        if self.binary:
            reg_target = np.concatenate([reg_target[:,:,:2],reg_target[:,:,5:]],axis=2)
            reg_loss_mask = np.concatenate([reg_loss_mask[:,:,:2],reg_loss_mask[:,:,5:]],axis=2)
            label_one_hot = np.concatenate([label_one_hot[:,:,:2],label_one_hot[:,:,5:]],axis=2)

        '''
        padded_voxel_points = list()

        for i in range(self.num_past_pcs):
            indices = gt_dict['voxel_indices_' + str(i)]
            curr_voxels = np.zeros(self.dims, dtype=np.bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            curr_voxels = np.rot90(curr_voxels,3)
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)
        anchors_map = self.anchors_map
        '''
        if self.binary:
            anchors_map = np.concatenate([anchors_map[:,:,:2],anchors_map[:,:,5:]],axis=2)
        '''
        if self.config.use_vis:
            vis_maps = np.zeros((self.num_past_pcs,self.config.map_dims[-1],self.config.map_dims[0],self.config.map_dims[1]))          
            vis_free_indices = gt_dict['vis_free_indices']
            vis_occupy_indices = gt_dict['vis_occupy_indices']
            vis_maps[vis_occupy_indices[0,:],vis_occupy_indices[1,:],vis_occupy_indices[2,:],vis_occupy_indices[3,:]] = math.log(0.7/(1-0.7))            
            vis_maps[vis_free_indices[0,:],vis_free_indices[1,:],vis_free_indices[2,:],vis_free_indices[3,:]] = math.log(0.4/(1-0.4))
            vis_maps = np.swapaxes(vis_maps,2,3)
            vis_maps = np.transpose(vis_maps,(0,2,3,1))
            for v_id in range(vis_maps.shape[0]):
                vis_maps[v_id] = np.rot90(vis_maps[v_id],3)
            vis_maps = vis_maps[-1]

        else:
            vis_maps = np.zeros(0)
        
        padded_voxel_points = padded_voxel_points.astype(np.float32)
        label_one_hot = label_one_hot.astype(np.float32)
        reg_target = reg_target.astype(np.float32)
        anchors_map = anchors_map.astype(np.float32)
        motion_one_hot = motion_one_hot.astype(np.float32)
        semantic_maps = semantic_maps.astype(np.float32)
        vis_maps = vis_maps.astype(np.float32)

        if self.val:
            return padded_voxel_points, label_one_hot,\
            reg_target,reg_loss_mask,anchors_map,motion_one_hot,motion_mask,vis_maps,[{"gt_box":gt_max_iou}],[seq_file]
        else:
            return padded_voxel_points, label_one_hot,\
            reg_target,reg_loss_mask,anchors_map,motion_one_hot,motion_mask,vis_maps

class CarscenesDataset(Dataset):
    def __init__(self, padded_voxel_points_example, label_one_hot_example, reg_target_example, reg_loss_mask_example,
                 anchors_map_example, \
                 motion_one_hot_example, motion_mask_example, vis_maps_example, dataset_root=None, config=None, 
                 config_global=None, agent_list = ['agent0', 'agent1', 'agent2', 'agent3', 'agent4'], split=None, cache_size=10000, val=False, latency_lambda = [10,10,10,10,10], forecast_num = 5):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        category_num: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if split is None:
            self.split = config.split
        else:
            self.split = split
        self.voxel_size = config.voxel_size
        self.area_extents = config.area_extents
        self.category_num = config.category_num
        self.future_frame_skip = config.future_frame_skip
        self.pred_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.anchor_size = config.anchor_size
        self.latency_lambda = latency_lambda
        self.forecast_num = forecast_num

        self.val = val
        self.only_det = config.only_det
        self.binary = config.binary
        self.config = config
        self.use_vis = config.use_vis
        self.padded_voxel_points_example = padded_voxel_points_example
        self.label_one_hot_example = label_one_hot_example
        self.reg_target_example = reg_target_example
        self.reg_loss_mask_example = reg_loss_mask_example
        self.anchors_map_example = anchors_map_example
        self.motion_one_hot_example = motion_one_hot_example
        self.motion_mask_example = motion_mask_example
        self.vis_maps_example = vis_maps_example

        # dataset_root = dataset_root + '/'+split
        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(self.split))
        self.dataset_root = dataset_root
        # seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
        #             if os.path.isdir(os.path.join(self.dataset_root, d))]
        # seq_dirs = sorted(seq_dirs)
        self.seq_dict = {}
        self.num_agent = len(agent_list)
        # for agent_num in range(self.num_agent):
        for agent_num in range(1):
            self.dataset_root_peragent = self.dataset_root + agent_list[agent_num]
            self.seq_dict[agent_num] = self.get_data_dict(self.dataset_root_peragent)
        # self.seq_files = [os.path.join(seq_dir, f) for seq_dir in seq_dirs for f in os.listdir(seq_dir)
        # #                   if os.path.isfile(os.path.join(seq_dir, f))]

        # self.num_sample_seqs = len(self.seq_files)
        # self.num_sample_seqs = len(self.seq_dict[0][0]) * self.num_agent
        self.num_sample_seqs = len(self.seq_dict[0][0])
        print("The number of {} sequences: {}".format(self.split, self.num_sample_seqs))
        # object information
        self.anchors_map = init_anchors_no_check(self.area_extents, self.voxel_size, self.box_code_size,
                                                 self.anchor_size)
        self.map_dims = [int((self.area_extents[0][1] - self.area_extents[0][0]) / self.voxel_size[0]), \
                         int((self.area_extents[1][1] - self.area_extents[1][0]) / self.voxel_size[1])]
        self.reg_target_shape = (
        self.map_dims[0], self.map_dims[1], len(self.anchor_size), self.pred_len, self.box_code_size)
        self.label_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size))
        self.label_one_hot_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size), self.category_num)
        self.dims = config.map_dims
        self.num_past_pcs = config.num_past_pcs
        # manager = Manager()
        # self.cache = manager.dict()
        self.cache = {}
        # for i in range(self.num_agent):
        #     self.cache[i] = {}
        # self.cache_size = cache_size if split == 'train' else 0
        self.cache_size = cache_size

        if self.val:
            self.voxel_size_global = config_global.voxel_size
            self.area_extents_global = config_global.area_extents
            self.pred_len_global = config_global.pred_len
            self.box_code_size_global = config_global.box_code_size
            self.anchor_size_global = config_global.anchor_size
            # object information
            self.anchors_map_global = init_anchors_no_check(self.area_extents_global, self.voxel_size_global,
                                                            self.box_code_size_global, self.anchor_size_global)
            self.map_dims_global = [
                int((self.area_extents_global[0][1] - self.area_extents_global[0][0]) / self.voxel_size_global[0]), \
                int((self.area_extents_global[1][1] - self.area_extents_global[1][0]) / self.voxel_size_global[1])]
            self.reg_target_shape_global = (
            self.map_dims_global[0], self.map_dims_global[1], len(self.anchor_size_global), self.pred_len_global,
            self.box_code_size_global)
            self.dims_global = config_global.map_dims

    def __len__(self):
        return self.num_sample_seqs

    def get_one_hot(self, label, category_num):
        one_hot_label = np.zeros((label.shape[0], category_num))
        for i in range(label.shape[0]):
            one_hot_label[i][label[i]] = 1

        return one_hot_label

    def get_data_dict(self, file_path):
        file_dict = {}
        latency_lambda = self.latency_lambda
        for agent in range(len(latency_lambda)):
            file_dict[agent] = []
        for file_name in os.listdir(file_path):
            path_item = os.path.join(file_path, file_name).strip('/').split('/')
            scene_stamp, time_stamp = path_item[-1].split('_')
            if int(time_stamp) >= 10:
                agent_id = int(path_item[-2][-1])
                del path_item[-2:]
                path_pre = '/'
                for item in path_item:
                    path_pre = os.path.join(path_pre, item)
                latency_list = []
                for agent in range(len(latency_lambda)):
                    if agent == agent_id:
                        file_dict[agent_id].append(os.path.join(file_path, file_name))
                    else:
                        # latency_list.append(np.ceil(np.random.exponential(latency_lambda[agent])))
                        latency = int(np.ceil(np.random.exponential(latency_lambda[agent])))
                        delay_time_stamp = int(time_stamp) - latency
                        if delay_time_stamp <= 0:
                            delay_time_stamp = 0
                        scene_time = str(scene_stamp) + '_' + str(delay_time_stamp)
                        file_dict[agent].append(os.path.join(os.path.join(path_pre, 'agent' + str(agent)), scene_time))
            else:
                continue
        return file_dict

    def per_scene_load(self, para):
        data_return = {}
        cache = {}
        [agent, index, seq_file] = para
        if seq_file not in self.cache.keys():
            # write_flag = False
            # if index == 0:
            #     write_flag = True
            # seq_file = self.seq_files[idx]
            empty_flag = False
            gt_data_handle = np.load(os.path.join(seq_file, '0.npy'), allow_pickle=True)
            if gt_data_handle == 0:
                empty_flag = True
                data_return['current_pose_rec'] = {'token': '5ntnl2o4swb2v45g912bw6u8p15os2nb', 'timestamp': 999999, 'rotation': [0,0,0,0], 'translation': [0,0,0]}
                data_return['current_cs_rec'] = {'token': '5ntnl2o4swb2v45g912bw6u8p15os2lr', 'camera_intrinsic': [], 'sensor_token':'123123124r14512', 'rotation': [0,0,0,0], 'translation': [0,0,0]}
                padded_voxel_points = np.zeros(self.padded_voxel_points_example.shape).astype(np.float32)
                label_one_hot = np.zeros(self.label_one_hot_example.shape).astype(np.float32)
                reg_target = np.zeros(self.reg_target_example.shape).astype(np.float32)
                anchors_map = np.zeros(self.anchors_map_example.shape).astype(np.float32)
                motion_one_hot = np.zeros(self.motion_one_hot_example.shape).astype(np.float32)
                vis_maps = np.zeros(self.vis_maps_example.shape).astype(np.float32)
                reg_loss_mask = np.zeros(self.reg_loss_mask_example.shape).astype(self.reg_loss_mask_example.dtype)
                motion_mask = np.zeros(self.motion_mask_example.shape).astype(self.motion_mask_example.dtype)
                if self.val:
                    data_return['padded_voxel_points'] = (padded_voxel_points)
                    data_return['label_one_hot'] = label_one_hot
                    data_return['reg_target'] = reg_target
                    data_return['reg_loss_mask'] = reg_loss_mask
                    data_return['anchors_map'] = anchors_map
                    data_return['vis_maps'] = vis_maps
                    data_return['gt_max_iou'] = [{"gt_box": [[0, 0, 0, 0], [0, 0, 0, 0]]}]
                    data_return['filename'] = seq_file
                    data_return['target_agent_id'] = 0
                    data_return['num_sensor'] = 0
                    data_return['trans_matrices'] = np.zeros((5, 4, 4))
                    data_return['padded_voxel_points_global'] = 0
                    data_return['reg_target_global'] = 0
                    data_return['anchors_map_global'] = 0
                    data_return['gt_max_iou_global'] = 0
                    data_return['trans_matrices_map'] = 0
                    
                    # cache['padded_voxel_points'] = (padded_voxel_points)
                    # cache['label_one_hot'] = label_one_hot
                    # cache['reg_target'] = reg_target
                    # cache['reg_loss_mask'] = reg_loss_mask
                    # cache['anchors_map'] = anchors_map
                    # cache['vis_maps'] = vis_maps
                    # cache['gt_max_iou'] = [{"gt_box": [[0, 0, 0, 0], [0, 0, 0, 0]]}]
                    # cache['filename'] = seq_file
                    # cache['target_agent_id'] = 0
                    # cache['num_sensor'] = 0
                    # cache['trans_matrices'] = np.zeros((5, 4, 4))
                    # cache['padded_voxel_points_global'] = 0
                    # cache['reg_target_global'] = 0
                    # cache['anchors_map_global'] = 0
                    # cache['gt_max_iou_global'] = 0
                    # cache['trans_matrices_map'] = 0
                else:
                    data_return['trans_matrices'] = np.zeros((5, 4, 4))
                    data_return['padded_voxel_points'] = padded_voxel_points
                    data_return['filename'] = seq_file
                    data_return['target_agent_id'] = 0
                    data_return['num_sensor'] = 0

                    cache['trans_matrices'] = np.zeros((5, 4, 4))
                    cache['padded_voxel_points'] = padded_voxel_points
                    cache['filename'] = seq_file
                    cache['target_agent_id'] = 0
                    cache['num_sensor'] = 0

                data_return['padded_voxel_points_teacher'] = padded_voxel_points
                data_return['label_one_hot'] = label_one_hot
                data_return['reg_target'] = reg_target
                data_return['reg_loss_mask'] = reg_loss_mask
                data_return['anchors_map'] = anchors_map
                data_return['vis_maps'] = vis_maps
                data_return['gt_max_iou'] = [{"gt_box": [[0, 0, 0, 0], [0, 0, 0, 0]]}]
                    
                    # cache['padded_voxel_points_teacher'] = padded_voxel_points
                    # cache['label_one_hot'] = label_one_hot
                    # cache['reg_target'] = reg_target
                    # cache['reg_loss_mask'] = reg_loss_mask
                    # cache['anchors_map'] = anchors_map
                    # cache['vis_maps'] = vis_maps
                    # cache['gt_max_iou'] = [{"gt_box": [[0, 0, 0, 0], [0, 0, 0, 0]]}]
                    # data_return['trans_matrices'] = np.zeros((5, 4, 4)) 
            else:
                gt_dict = gt_data_handle.item()
                # if len(self.cache) < self.cache_size:
                #     self.cache[idx] = gt_dict
            if empty_flag == False:
                data_return['current_pose_rec'] = gt_dict['current_pose_rec']
                data_return['current_cs_rec'] = gt_dict['current_cs_rec']
                allocation_mask = gt_dict['allocation_mask'].astype(np.bool)
                reg_loss_mask = gt_dict['reg_loss_mask'].astype(np.bool)
                gt_max_iou = gt_dict['gt_max_iou']
                motion_one_hot = np.zeros(5)
                motion_mask = np.zeros(5)
                # load regression target
                reg_target_sparse = gt_dict['reg_target_sparse']
                # need to be modified Yiqi , only use reg_target and allocation_map
                reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)
                reg_target[allocation_mask] = reg_target_sparse
                reg_target[np.bitwise_not(reg_loss_mask)] = 0
                label_sparse = gt_dict['label_sparse']
                one_hot_label_sparse = self.get_one_hot(label_sparse, self.category_num)
                label_one_hot = np.zeros(self.label_one_hot_shape)
                label_one_hot[:, :, :, 0] = 1
                label_one_hot[allocation_mask] = one_hot_label_sparse
                if self.only_det:
                    reg_target = reg_target[:, :, :, :1]
                    reg_loss_mask = reg_loss_mask[:, :, :, :1]
                # only center for pred
                elif self.config.pred_type in ['motion', 'center']:
                    reg_loss_mask = np.expand_dims(reg_loss_mask, axis=-1)
                    reg_loss_mask = np.repeat(reg_loss_mask, self.box_code_size, axis=-1)
                    reg_loss_mask[:, :, :, 1:, 2:] = False
                padded_voxel_points = list()
                for i in range(self.num_past_pcs):
                    indices = gt_dict['voxel_indices_' + str(i)]
                    curr_voxels = np.zeros(self.dims, dtype=np.bool)
                    curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                    curr_voxels = np.rot90(curr_voxels, 3)
                    padded_voxel_points.append(curr_voxels)
                padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)
                anchors_map = self.anchors_map
                if self.config.use_vis:
                    vis_maps = np.zeros(
                        (self.num_past_pcs, self.config.map_dims[-1], self.config.map_dims[0], self.config.map_dims[1]))
                    vis_free_indices = gt_dict['vis_free_indices']
                    vis_occupy_indices = gt_dict['vis_occupy_indices']
                    vis_maps[
                        vis_occupy_indices[0, :], vis_occupy_indices[1, :], vis_occupy_indices[2, :], vis_occupy_indices[3,
                                                                                                    :]] = math.log(
                        0.7 / (1 - 0.7))
                    vis_maps[vis_free_indices[0, :], vis_free_indices[1, :], vis_free_indices[2, :], vis_free_indices[3,
                                                                                                    :]] = math.log(
                        0.4 / (1 - 0.4))
                    vis_maps = np.swapaxes(vis_maps, 2, 3)
                    vis_maps = np.transpose(vis_maps, (0, 2, 3, 1))
                    for v_id in range(vis_maps.shape[0]):
                        vis_maps[v_id] = np.rot90(vis_maps[v_id], 3)
                    vis_maps = vis_maps[-1]
                else:
                    vis_maps = np.zeros(0)
                trans_matrices = gt_dict['trans_matrices']
                padded_voxel_points = padded_voxel_points.astype(np.float32)
                label_one_hot = label_one_hot.astype(np.float32)
                reg_target = reg_target.astype(np.float32)
                anchors_map = anchors_map.astype(np.float32)
                motion_one_hot = motion_one_hot.astype(np.float32)
                vis_maps = vis_maps.astype(np.float32)
                target_agent_id = gt_dict['target_agent_id']
                num_sensor = gt_dict['num_sensor']
                if not self.val:
                    padded_voxel_points_teacher = list()
                    indices_teacher = gt_dict['voxel_indices_teacher']
                    curr_voxels_teacher = np.zeros(self.dims, dtype=np.bool)
                    curr_voxels_teacher[indices_teacher[:, 0], indices_teacher[:, 1], indices_teacher[:, 2]] = 1
                    curr_voxels_teacher = np.rot90(curr_voxels_teacher, 3)
                    padded_voxel_points_teacher.append(curr_voxels_teacher)
                    padded_voxel_points_teacher = np.stack(padded_voxel_points_teacher, 0).astype(np.float32)
                    padded_voxel_points_teacher = padded_voxel_points_teacher.astype(np.float32)
                if self.val:
                    trans_matrices_map = gt_dict['trans_matrices_map']
                    padded_voxel_points_global = list()
                    indices_global = gt_dict['voxel_indices_global']
                    curr_voxels_global = np.zeros(self.dims_global, dtype=np.bool)
                    curr_voxels_global[indices_global[:, 0], indices_global[:, 1], indices_global[:, 2]] = 1
                    curr_voxels_global = np.rot90(curr_voxels_global, 3)
                    padded_voxel_points_global.append(curr_voxels_global)
                    padded_voxel_points_global = np.stack(padded_voxel_points_global, 0).astype(np.float32)
                    padded_voxel_points_global = padded_voxel_points_global.astype(np.float32)
                    allocation_mask_global = gt_dict['allocation_mask_global'].astype(np.bool)
                    reg_loss_mask_global = gt_dict['reg_loss_mask_global'].astype(np.bool)
                    gt_max_iou_global = gt_dict['gt_max_iou_global']
                    # load regression target
                    reg_target_sparse_global = gt_dict['reg_target_sparse_global']
                    # need to be modified Yiqi , only use reg_target and allocation_map
                    if target_agent_id == 0:
                        reg_target_global = np.zeros(self.reg_target_shape_global).astype(reg_target_sparse_global.dtype)
                    else:
                        reg_target_global = np.zeros(self.reg_target_shape).astype(reg_target_sparse_global.dtype)
                    reg_target_global[allocation_mask_global] = reg_target_sparse_global
                    reg_target_global[np.bitwise_not(reg_loss_mask_global)] = 0
                    if self.only_det:
                        reg_target_global = reg_target_global[:, :, :, :1]
                        reg_loss_mask_global = reg_loss_mask_global[:, :, :, :1]
                    reg_target_global = reg_target_global.astype(np.float32)
                    anchors_map_global = self.anchors_map_global
                    anchors_map_global = anchors_map_global.astype(np.float32)
                    data_return['padded_voxel_points'] = padded_voxel_points
                    data_return['label_one_hot'] = label_one_hot
                    data_return['reg_target'] = reg_target
                    data_return['reg_loss_mask'] = reg_loss_mask
                    data_return['anchors_map'] = anchors_map
                    data_return['vis_maps'] = vis_maps
                    data_return['gt_max_iou'] = [ {"gt_box": gt_max_iou}]
                    data_return['filename'] =  seq_file
                    data_return['target_agent_id'] = target_agent_id
                    data_return['num_sensor'] = num_sensor
                    data_return['trans_matrices'] = trans_matrices
                    data_return['padded_voxel_points_global'] = padded_voxel_points_global
                    data_return['reg_target_global'] = reg_target_global
                    data_return['anchors_map_global'] = anchors_map_global
                    data_return['gt_max_iou_global'] = [{"gt_box_global": gt_max_iou_global}]
                    data_return['trans_matrices_map'] = trans_matrices_map

                    # cache['padded_voxel_points'] = padded_voxel_points
                    # cache['label_one_hot'] = label_one_hot
                    # cache['reg_target'] = reg_target
                    # cache['reg_loss_mask'] = reg_loss_mask
                    # cache['anchors_map'] = anchors_map
                    # cache['vis_maps'] = vis_maps
                    # cache['gt_max_iou'] = [ {"gt_box": gt_max_iou}]
                    # cache['filename'] =  seq_file
                    # cache['target_agent_id'] = target_agent_id
                    # cache['num_sensor'] = num_sensor
                    # cache['trans_matrices'] = trans_matrices
                    # cache['padded_voxel_points_global'] = padded_voxel_points_global
                    # cache['reg_target_global'] = reg_target_global
                    # cache['anchors_map_global'] = anchors_map_global
                    # cache['gt_max_iou_global'] = [{"gt_box_global": gt_max_iou_global}]
                    # cache['trans_matrices_map'] = trans_matrices_map
                else:
                    # return padded_voxel_points, padded_voxel_points_teacher, label_one_hot, reg_target, reg_loss_mask, anchors_map, vis_maps, \
                    #     target_agent_id, num_sensor, trans_matrices
                    data_return['padded_voxel_points'] = padded_voxel_points
                    data_return['filename'] =  seq_file 
                    data_return['target_agent_id'] = target_agent_id
                    data_return['num_sensor'] = num_sensor
                    data_return['trans_matrices'] = trans_matrices

                    # cache['padded_voxel_points'] = padded_voxel_points
                    # cache['filename'] =  seq_file 
                    # cache['target_agent_id'] = target_agent_id
                    # cache['num_sensor'] = num_sensor
                    # cache['trans_matrices'] = trans_matrices
                    
                # if write_flag == True:
                data_return['padded_voxel_points_teacher'] = padded_voxel_points_teacher
                data_return['label_one_hot'] = label_one_hot
                data_return['reg_target'] = reg_target
                data_return['reg_loss_mask'] = reg_loss_mask
                data_return['anchors_map'] = anchors_map
                data_return['vis_maps'] = vis_maps
                data_return['gt_max_iou'] = [{"gt_box": [[0, 0, 0, 0], [0, 0, 0, 0]]}]

                    # cache['padded_voxel_points_teacher'] = padded_voxel_points_teacher
                    # cache['label_one_hot'] = label_one_hot
                    # cache['reg_target'] = reg_target
                    # cache['reg_loss_mask'] = reg_loss_mask
                    # cache['anchors_map'] = anchors_map
                    # cache['vis_maps'] = vis_maps
                    # cache['gt_max_iou'] = [{"gt_box": [[0, 0, 0, 0], [0, 0, 0, 0]]}]
                # data_return['trans_matrices'] = trans_matrices
        else:
            data_return = self.cache[seq_file]
            # if self.val:
            #     data_return['padded_voxel_points'] = cache['padded_voxel_points']
            #     data_return['label_one_hot'] = cache['label_one_hot']
            #     data_return['reg_target'] = cache['reg_target']
            #     data_return['reg_loss_mask'] = cache['reg_loss_mask']
            #     data_return['anchors_map'] = cache['anchors_map']
            #     data_return['vis_maps'] = cache['vis_maps']
            #     data_return['gt_max_iou'] = cache['gt_max_iou']
            #     data_return['filename'] =  seq_file
            #     data_return['target_agent_id'] = cache['target_agent_id']
            #     data_return['num_sensor'] = cache['num_sensor']
            #     data_return['trans_matrices'] = cache['trans_matrices']
            #     data_return['padded_voxel_points_global'] = cache['padded_voxel_points_global']
            #     data_return['reg_target_global'] = cache['reg_target_global']
            #     data_return['anchors_map_global'] = cache['anchors_map_global']
            #     data_return['gt_max_iou_global'] = cache['gt_max_iou_global']
            #     data_return['trans_matrices_map'] = cache['trans_matrices_map']
            # else:
            #     # return padded_voxel_points, padded_voxel_points_teacher, label_one_hot, reg_target, reg_loss_mask, anchors_map, vis_maps, \
            #     #     target_agent_id, num_sensor, trans_matrices
            #     data_return['padded_voxel_points'] = cache['padded_voxel_points']
            #     data_return['filename'] =  seq_file 
            #     data_return['target_agent_id'] = cache['target_agent_id']
            #     data_return['num_sensor'] = cache['num_sensor']
            #     data_return['trans_matrices'] = cache['trans_matrices']
            #     if index == 0:
            #         data_return['padded_voxel_points_teacher'] = cache['padded_voxel_points_teacher']
            #         data_return['label_one_hot'] = cache['label_one_hot']
            #         data_return['reg_target'] = cache['reg_target']
            #         data_return['reg_loss_mask'] = cache['reg_loss_mask']
            #         data_return['anchors_map'] = cache['anchors_map']
            #         data_return['vis_maps'] = cache['vis_maps']
            #         data_return['gt_max_iou'] = cache['gt_max_iou']
            #     # data_return['trans_matrices'] = trans_matrices
        return data_return

    # def per_agent_load(self, agent):
        # [idx, agent, time_1] = para
        # print(load_list)
        # if self.val:
        #     data_return['padded_voxel_points'] = [None for _ in range(self.forecast_num)]
        #     data_return['label_one_hot'] = [None for _ in range(self.forecast_num)]
        #     data_return['reg_target'] = [None for _ in range(self.forecast_num)]
        #     data_return['reg_loss_mask'] = [None for _ in range(self.forecast_num)]
        #     data_return['anchors_map'] = [None for _ in range(self.forecast_num)]
        #     data_return['vis_maps'] = [None for _ in range(self.forecast_num)]
        #     data_return['gt_max_iou'] =  [None for _ in range(self.forecast_num)]
        #     data_return['filename'] = [None for _ in range(self.forecast_num)]
        #     data_return['target_agent_id'] = [None for _ in range(self.forecast_num)]
        #     data_return['num_sensor'] = [None for _ in range(self.forecast_num)]
        #     data_return['trans_matrices'] = [None for _ in range(self.forecast_num)]
        #     data_return['padded_voxel_points_global'] = [None for _ in range(self.forecast_num)]
        #     data_return['reg_target_global'] = [None for _ in range(self.forecast_num)]
        #     data_return['anchors_map_global'] = [None for _ in range(self.forecast_num)]
        #     data_return['gt_max_iou_global'] = [None for _ in range(self.forecast_num)]
        #     data_return['trans_matrices_map'] = [None for _ in range(self.forecast_num)]
        # else:
        #     data_return['padded_voxel_points'] = [None for _ in range(self.forecast_num)]
        #     # data_return['padded_voxel_points_teacher'] = []
        #     # data_return['label_one_hot'] = []
        #     # data_return['reg_target'] = []
        #     # data_return['reg_loss_mask'] = []
        #     # data_return['anchors_map'] = []
        #     # data_return['vis_maps'] = []
        #     # data_return['gt_max_iou'] =  []
        #     data_return['trans_matrices'] = [None for _ in range(self.forecast_num)]
        #     data_return['filename'] = [None for _ in range(self.forecast_num)]
        #     data_return['target_agent_id'] = [None for _ in range(self.forecast_num)]
        #     data_return['num_sensor'] = [None for _ in range(self.forecast_num)]

        # para_list = []
        # write_flag_list = [True, False, False, False]
        # agent_list = [agent, agent, agent, agent]
        # for para in zip(load_list, write_flag_list, range(self.forecast_num), agent_list):
        #     para_list.append(para)
        # # for seq_file in load_list:
        # with futures.ProcessPoolExecutor(None) as executor:
        #     res = executor.map(self.per_scene_load, sorted(para_list), chunksize=1)
        # time_3 = time.time()
        # # print(agent, '号计时点3:', time_3 - time_1, "非零数量：", data_return['num_sensor'])
        # data_return['padded_voxel_points'] = torch.stack(tuple(torch.Tensor(data_return['padded_voxel_points'])),0)
        # # data_return['filename'] = data_return['filename'].strip(',')
        # data_return['target_agent_id'] = torch.Tensor(data_return['target_agent_id'])
        # data_return['agent'] = agent           
        # data_return['trans_matrices'] = torch.stack(tuple(torch.Tensor(data_return['trans_matrices'])),0)
        # time_4 = time.time()
        # filename_str = ''
        # for file in data_return['filename']:
        #     filename_str += file
        #     filename_str += ','
        # data_return['filename'] = filename_str.strip(',')
        # # print(agent, '号计时点4:', time_4 - time_1)
        # # self.cache[agent][idx] = data_return
        # return data_return

    # def late_processing(self, agent):
    #     data_return['padded_voxel_points'] = torch.stack(tuple(torch.Tensor(data_return['padded_voxel_points'])),0)
    #     # data_return['filename'] = data_return['filename'].strip(',')
    #     data_return['target_agent_id'] = torch.Tensor(data_return['target_agent_id'])
    #     data_return['agent'] = agent           
    #     data_return['trans_matrices'] = torch.stack(tuple(torch.Tensor(data_return['trans_matrices'])),0)
    #     time_4 = time.time()
    #     filename_str = ''
    #     for file in data_return['filename']:
    #         filename_str += file
    #         filename_str += ','
    #     data_return['filename'] = filename_str.strip(',')

    def __getitem__(self, idx):
        time_start = time.time()
        # if idx in self.cache:
        #     gt_dict = self.cache[idx]
        # else:
        if idx == 0:        
            self.seq_dict = {}
            # for agent_num in range(self.num_agent):
            for agent_num in range(1):
                self.dataset_root_peragent = self.dataset_root + '/agent0'
                self.seq_dict[agent_num] = self.get_data_dict(self.dataset_root_peragent)
        time_2 = time.time()
        # try:
        #     print('迭代时间:',time_2 - self.time_4)
        # except:
        #     print('wait')
        data_return = {}
        load_list = []
        for agent in range(len(self.latency_lambda)):
            # data_return = Manager().dict()
            # seq_file_new = self.seq_dict[self.center_agent][agent][idx]
            seq_file_new = self.seq_dict[0][agent][idx]
            scene_id, inscene_id = seq_file_new.split('/')[-1].split('_')   
            inscene_id = int(inscene_id)
            seq_scene_root = seq_file_new.strip(seq_file_new.split('/')[-1])
            for iter in range(self.forecast_num):
                new_inscene_id = inscene_id - self.forecast_num + iter + 1
                if new_inscene_id <= 0:
                    new_inscene_id = 0
                load_list.append([agent, iter, os.path.join(seq_scene_root, scene_id + '_' + str(new_inscene_id))])
            # self.per_agent_load(agent)

        # if idx not in self.cache.keys():
        time_1 = time.time()
        # self.center_agent = int(idx / self.num_sample_seqs * self.num_agent)
        self.center_agent = 0
        # idx = idx % int(self.num_sample_seqs / self.num_agent)
        # seq_file_list = []
        # data_return = {}
        data_return['center_agent'] = self.center_agent
        # para_list = []
        time_1 = time.time()
        # for agent in range(len(self.latency_lambda)):
        #     temp_tuple = [idx, agent, time_1]
        #     para_list.append(temp_tuple)
        time_pas = time.time()
        workers = len(self.latency_lambda * self.forecast_num)
        # res = [None for _ in range(workers)]
        with futures.ThreadPoolExecutor(None) as executor:
            # for i in range(workers):
            #     res[i] = executor.submit(self.per_scene_load, load_list[i])
            res = executor.map(self.per_scene_load, sorted(load_list), chunksize=workers)
            res = list(res)
        # res = []
        # for para_item in load_list:
        #     res.append(self.per_scene_load(para_item))
        # print('并行处理时间:',time.time() - time_pas)
        for agent in range(len(self.latency_lambda)):
            data_return[agent] = {}
            if self.val:
                data_return[agent]['padded_voxel_points'] = []
                data_return[agent]['label_one_hot'] = []
                data_return[agent]['reg_target'] = []
                data_return[agent]['reg_loss_mask'] = []
                data_return[agent]['anchors_map'] = []
                data_return[agent]['vis_maps'] = []
                data_return[agent]['gt_max_iou'] = []
                data_return[agent]['filename'] =  []
                data_return[agent]['target_agent_id'] = []
                data_return[agent]['num_sensor'] = []
                data_return[agent]['trans_matrices'] = []
                data_return[agent]['current_pose_rec'] = []
                data_return[agent]['current_cs_rec'] = []
                data_return[agent]['padded_voxel_points_global'] = []
                data_return[agent]['reg_target_global'] = []
                data_return[agent]['anchors_map_global'] = []
                data_return[agent]['gt_max_iou_global'] = []
                data_return[agent]['trans_matrices_map'] = []
            else:
                # return padded_voxel_points, padded_voxel_points_teacher, label_one_hot, reg_target, reg_loss_mask, anchors_map, vis_maps, \
                #     target_agent_id, num_sensor, trans_matrices
                data_return[agent]['padded_voxel_points'] = []
                data_return[agent]['filename'] =  []
                data_return[agent]['target_agent_id'] = []
                data_return[agent]['num_sensor'] = []
                data_return[agent]['trans_matrices'] = []
                data_return[agent]['current_pose_rec'] = []
                data_return[agent]['current_cs_rec'] = []

            for iter in range(self.forecast_num):
                temp =  res[agent * self.forecast_num + iter]
                # if load_list[agent * self.forecast_num + iter][0] not in self.cache.keys():
                #     self.cache[load_list[agent * self.forecast_num + iter][2]] = temp
                if self.val:
                    data_return[agent]['padded_voxel_points'].append(temp[agent]['padded_voxel_points'])
                    data_return[agent]['label_one_hot'].append(temp[agent]['label_one_hot'])
                    data_return[agent]['reg_target'].append(temp[agent]['reg_target'])
                    data_return[agent]['reg_loss_mask'].append(temp[agent]['reg_loss_mask'])
                    data_return[agent]['anchors_map'].append(temp[agent]['anchors_map'])
                    data_return[agent]['vis_maps'].append(temp[agent]['vis_maps'])
                    data_return[agent]['gt_max_iou'].append(temp[agent]['gt_max_iou'])
                    data_return[agent]['filename'].append(temp[agent]['filename'])
                    data_return[agent]['target_agent_id'].append(temp[agent]['target_agent_id'])
                    data_return[agent]['num_sensor'].append(temp[agent]['num_sensor'])
                    data_return[agent]['trans_matrices'].append(temp[agent]['trans_matrices'])
                    data_return[agent]['padded_voxel_points_global'].append(temp[agent]['padded_voxel_points_global'])
                    data_return[agent]['reg_target_global'].append(temp[agent]['reg_target_global'])
                    data_return[agent]['anchors_map_global'].append(temp[agent]['anchors_map_global'])
                    data_return[agent]['gt_max_iou_global'].append(temp[agent]['gt_max_iou_global'])
                    data_return[agent]['trans_matrices_map'].append(temp[agent]['trans_matrices_map'])
                else:
                    # return padded_voxel_points, padded_voxel_points_teacher, label_one_hot, reg_target, reg_loss_mask, anchors_map, vis_maps, \
                    #     target_agent_id, num_sensor, trans_matrices
                    data_return[agent]['padded_voxel_points'].append(temp['padded_voxel_points'])
                    data_return[agent]['filename'].append(temp['filename'])
                    data_return[agent]['target_agent_id'] .append(temp['target_agent_id'])
                    data_return[agent]['num_sensor'].append(temp['num_sensor'])
                    data_return[agent]['trans_matrices'].append(temp['trans_matrices'])
                    data_return[agent]['current_pose_rec'].append(temp['current_pose_rec'])
                    data_return[agent]['current_cs_rec'].append(temp['current_cs_rec'])
                    if iter == self.forecast_num - 1:
                        data_return[agent]['padded_voxel_points_teacher'] = temp['padded_voxel_points_teacher']
                        data_return[agent]['label_one_hot'] = temp['label_one_hot']
                        data_return[agent]['reg_target'] = temp['reg_target']
                        data_return[agent]['reg_loss_mask'] = temp['reg_loss_mask']
                        data_return[agent]['anchors_map'] = temp['anchors_map']
                        data_return[agent]['vis_maps'] = temp['vis_maps']
                        data_return[agent]['gt_max_iou'] = temp['gt_max_iou']
        time_3 = time.time()
        # print('前读取过程:',time_3-time_2)
        for agent in range(len(self.latency_lambda)):
            # data_return[agent]['padded_voxel_points'] = torch.stack(tuple(data_return[agent]['padded_voxel_points']),0)
            data_return[agent]['padded_voxel_points'] = np.stack(data_return[agent]['padded_voxel_points'],0)
            # time_bs = time.time()
            # data_return['filename'] = data_return['filename'].strip(',')
            data_return[agent]['target_agent_id'] = torch.tensor(data_return[agent]['target_agent_id'])
            # time_as = time.time()
            # print('stack时间:',time_as-time_bs)
            data_return[agent]['agent'] = agent           
            data_return[agent]['trans_matrices'] = np.stack(data_return[agent]['trans_matrices'],0)
            filename_str = ''
            for file in data_return[agent]['filename']:
                filename_str += file
                filename_str += ','
            data_return[agent]['filename'] = filename_str.strip(',')
        # print('后处理时间:',time.time()-time_3)

        # agent_list = [0,1,2,3,4]
        # with futures.ProcessPoolExecutor(None) as executor:
        #     res = executor.map(self.late_processing, sorted(agent_list), chunksize=1)
        # with futures.ProcessPoolExecutor(workers) as executor:
        # # 添加任务, 假设我们要添加6个任务，由于进程池大小为2，每次能只有2个任务并行执行，其他任务排队
        #       for i in range(len(para_list)):
        #           future = executor.submit(self.per_agent_load, sorted(para_list)[i])
        #           res .append(future.result())

        # for i in range(len(res)):
        #     data_return[i] = res[i]
        # self.cache[idx] = data_return   
        self.time_4 = time.time()
        # print('后处理部分:',self.time_4-time_3)

        # print("一个item读取运行时间:",time.time() - time_start )
        return data_return
        # else:
        #     data_return = self.cache[idx]
        #     return data_return

if __name__ == "__main__":
    split = 'train'
    config = Config(binary=True,split = split)
    config_global = ConfigGlobal(binary=True,split = split)
    data_nuscenes = NuscenesDataset(dataset_root='./dataset/val/agent2',split = split,config=config,val=True)

    padded_voxel_points, label_one_hot, reg_target, reg_loss_mask,anchors_map,\
    motion_one_hot,motion_mask,vis_maps,gt_max_iou,_ = data_nuscenes[0]
    data_carscenes = CarscenesDataset(padded_voxel_points, label_one_hot,\
            reg_target, reg_loss_mask, anchors_map, motion_one_hot, motion_mask, vis_maps, dataset_root='./dataset/val/agent2',split = split,config=config,config_global=config_global,val=True)

    for idx in range(len(data_carscenes)):
        padded_voxel_points, label_one_hot, reg_target, reg_loss_mask,anchors_map,
        motion_one_hot,motion_mask,vis_maps,gt_max_iou,_,_,_,_,_ = data_carscenes[idx]
        anchor_corners_list = get_anchor_corners_list(anchors_map,data_carscenes.box_code_size)
        anchor_corners_map = anchor_corners_list.reshape(data_carscenes.map_dims[0],data_carscenes.map_dims[1],len(data_carscenes.anchor_size),4,2)
        gt_max_iou_idx = gt_max_iou[0]['gt_box']

        plt.clf()
        for p in range(data_carscenes.pred_len):

            plt.xlim(0,256)
            plt.ylim(0,256)
            for k in range(len(gt_max_iou_idx)):

                anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]
                encode_box = reg_target[tuple(gt_max_iou_idx[k][:-1])+(p,)]
                decode_box = bev_box_decode_np(encode_box,anchor)
                decode_corner = center_to_corner_box2d(np.asarray([anchor[:2]]),np.asarray([anchor[2:4]]),np.asarray([anchor[4:]]))[0]
                
                corners = coor_to_vis(decode_corner,data_carscenes.area_extents,data_carscenes.voxel_size)
                c_x,c_y = np.mean(corners,axis=0)
                corners = np.concatenate([corners,corners[[0]]])
                
                plt.plot(corners[:,0], corners[:,1], c='g',linewidth=2.0,zorder=20)
                plt.scatter(c_x, c_y, s=3,c = 'g',zorder=20)
                plt.plot([c_x,(corners[1][0]+corners[0][0])/2.],[c_y,(corners[1][1]+corners[0][1])/2.],linewidth=2.0,c='g',zorder=20)
        
            
            occupy = np.max(vis_maps,axis=-1)
            m = np.stack([occupy,occupy,occupy],axis=-1)
            m[m>0] = 0.99
            occupy = (m*255).astype(np.uint8)
            #-----------#
            free = np.min(vis_maps,axis=-1)
            m = np.stack([free,free,free],axis=-1)
            m[m<0] = 0.5
            free = (m*255).astype(np.uint8)
            #-----------#
            plt.imshow(np.max(padded_voxel_points.reshape(256, 256, 13), axis=2),alpha=1.0,zorder=12)
            #plt.imshow(occupy,alpha=0.5,zorder=10)#free
            plt.pause(0.1)          
        #break
    plt.show()
