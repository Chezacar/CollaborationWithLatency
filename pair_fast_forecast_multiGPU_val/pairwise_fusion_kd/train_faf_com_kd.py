#   Copyright 2021 MediaBrain Group of CMIC, Shanghai Jiao Tong University. All right reserved.
# The software, documentation and/or data in this file is provided on an "as is" basis, 
# and MediaBrain Group has no obligations to provide maintenance, support, updates, enhancements or modifications. 
# MediaBrain Group specifically disclaims any warranties, including, but not limited to, 
# the implied warranties of merchantability and fitness for any particular purpose. 
# In no event shall MediaBrain Group be liable to any party for direct, indirect, special, incidental, 
# or consequential damages, including lost profits, arising out of the use of this software 
# and its documentation, even if MediaBrain Group has been advised of the possibility of such damages. 
# As more fully described in the license agreement that was required in order to download this software, 
# documentation and/or data, permission to use, copy and modify this software without fee is granted, 
# but only for educational, research and non-commercial purposes.
# from utils.model import forecast_lstm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from utils.model import MotionNet
from utils.FaFModule import *
from utils.loss import *
from data.data_com_parallel import NuscenesDataset, CarscenesDataset
from data.config_com import Config, ConfigGlobal
from utils.mean_ap import eval_map
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

# def process_supervise_data(raw_data):
#     process_data = {}
#     process_data['bev_seq'] = []
#     process_data['trans_matrices'] = []
#     for i in range(len(raw_data)):
#         process_data['bev_seq'].append(raw_data[i]['padded_voxel_points'])
#         process_data['trans_matrices'].append(raw_data[i]['trans_matrices'])
#     process_data['bev_seq'] = torch.cat(tuple(process_data['bev_seq']), 0)
#     process_data['trans_matrices'] = torch.cat(tuple(process_data['trans_matrices']), 0)
#     return process_data


def main(config, config_global, args):
    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    only_load_model = args.model_only
    forecast_num = args.forecast_num
    latency_lambda = args.latency_lambda
    config.forecast = args.forecast_model
    config.forecast_loss = args.forecast_loss 
    if args.mode == 'train':
        config.forecast_KD = args.forecast_KD
    untrainable_list = args.utp
    start_epoch = 1

    # communicate a single layer [0: 32*256*256, 1: 64*128*128, 2: 128*64*64, 3: 256*32*32,  4: 512*16*16] [C, W, H]
    layer = args.layer
    batch_size = args.batch

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)
    print("latency_lambda: ",latency_lambda)
    for i in latency_lambda:
        print('latency: ', i)
    if args.mode == 'train':
        # Whether to log the training information
        if need_log:
            logger_root = args.logpath if args.logpath != '' else 'logs'
            time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

            if args.resume == '':
                model_save_path = check_folder(logger_root)
                model_save_path = check_folder(os.path.join(model_save_path, 'train_single_seq'))
                model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

                log_file_name = os.path.join(model_save_path, 'log.txt')
                saver = open(log_file_name, "w")
                saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
                saver.flush()

                # Logging the details for this experiment
                saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
                saver.write(args.__repr__() + "\n\n")
                saver.flush()

                # Copy the code files as logs
                copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'))
                copytree('data', os.path.join(model_save_path, 'data'))
                python_files = [f for f in os.listdir('.') if f.endswith('.py')]
                for f in python_files:
                    copy(f, model_save_path)
            else:
                # model_save_path = args.resume[:args.resume.rfind('/')]
                torch.load(args.resume)  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"
                model_save_path = args.logpath
                log_file_name = os.path.join(model_save_path, 'log.txt')
                saver = open(log_file_name, "a")
                saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
                saver.flush()

                # Logging the details for this experiment
                saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
                saver.write(args.__repr__() + "\n\n")
                saver.flush()

        # load data from multiple agents
        data_nuscenes = NuscenesDataset(dataset_root=args.data + '/agent0', split='train', config=config)

        padded_voxel_points_example, label_one_hot_example, reg_target_example, reg_loss_mask_example, \
        anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example = data_nuscenes[0]

        trainset = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
                                     reg_loss_mask_example, \
                                     anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
                                     dataset_root=args.data, config=config, config_global=config_global, agent_list = ['/agent0', '/agent1', '/agent2', '/agent3', '/agent4'],
                                     split='train', forecast_num = forecast_num,latency_lambda = latency_lambda)

        # trainset0 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                              reg_loss_mask_example, \
        #                              anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                              dataset_root=args.data + '/agent0', config=config, config_global=config_global,
        #                              split='train', center_agent = 0)

        # trainset1 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                              reg_loss_mask_example, \
        #                              anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                              dataset_root=args.data + '/agent1', config=config, config_global=config_global,
        #                              split='train', center_agent = 1)

        # trainset2 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                              reg_loss_mask_example, \
        #                              anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                              dataset_root=args.data + '/agent2', config=config, config_global=config_global,
        #                              split='train', center_agent = 2)

        # trainset3 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                              reg_loss_mask_example, \
        #                              anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                              dataset_root=args.data + '/agent3', config=config, config_global=config_global,
        #                              split='train', center_agent = 3)

        # trainset4 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                              reg_loss_mask_example, \
        #                              anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                              dataset_root=args.data + '/agent4', config=config, config_global=config_global,
        #                              split='train', center_agent = 4)

        print("Training dataset size:", len(trainset))

    if args.mode == 'val':
        data_nuscenes = NuscenesDataset(dataset_root=args.data + '/agent0', config=config, split='val', val=True)

        padded_voxel_points_example, label_one_hot_example, reg_target_example, reg_loss_mask_example, \
        anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example, _, _ = data_nuscenes[0]


        valset = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
                                     reg_loss_mask_example, \
                                     anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
                                     dataset_root=args.data, config=config, config_global=config_global, agent_list = ['/agent0', '/agent1', '/agent2', '/agent3', '/agent4'],
                                     split='val', val=True, forecast_num = forecast_num,latency_lambda=latency_lambda)
        # valset0 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                            reg_loss_mask_example, \
        #                            anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                            dataset_root=args.data + '/agent0', config=config, config_global=config_global,
        #                            split='val', val=True)

        # valset1 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                            reg_loss_mask_example, \
        #                            anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                            dataset_root=args.data + '/agent1', config=config, config_global=config_global,
        #                            split='val', val=True)

        # valset2 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                            reg_loss_mask_example, \
        #                            anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                            dataset_root=args.data + '/agent2', config=config, config_global=config_global,
        #                            split='val', val=True)

        # valset3 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                            reg_loss_mask_example, \
        #                            anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                            dataset_root=args.data + '/agent3', config=config, config_global=config_global,
        #                            split='val', val=True)

        # valset4 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example, reg_target_example,
        #                            reg_loss_mask_example, \
        #                            anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example,
        #                            dataset_root=args.data + '/agent4', config=config, config_global=config_global,
        #                            split='val', val=True)

        valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
        # valloader0 = torch.utils.data.DataLoader(valset0, batch_size=1, shuffle=False, num_workers=1)
        # valloader1 = torch.utils.data.DataLoader(valset1, batch_size=1, shuffle=False, num_workers=1)
        # valloader2 = torch.utils.data.DataLoader(valset2, batch_size=1, shuffle=False, num_workers=1)
        # valloader3 = torch.utils.data.DataLoader(valset3, batch_size=1, shuffle=False, num_workers=1)
        # valloader4 = torch.utils.data.DataLoader(valset4, batch_size=1, shuffle=False, num_workers=1)

        print("Validation dataset size:", len(valset))

    # build model
    if config.MGDA:
        encoder = FeatEncoder()
        encoder = nn.DataParallel(encoder)
        encoder = encoder.to(device)
        optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr)
        head = FaFMGDA(config)
        head = nn.DataParallel(head)
        head = head.to(device)
        optimizer_head = optim.Adam(head.parameters(), lr=args.lr)

        model = [encoder, head]
        optimizer = [optimizer_encoder, optimizer_head]
    elif config.MIMO:
        if layer == 0:
            model = FaFMIMONet_32_256_256(config)
        elif layer == 1:
            model = FaFMIMONet_64_128_128(config)
        elif layer == 2:
            if config.KD:
                model = FaFMIMONet_128_64_64_KD(config)
            else:
                model = FaFMIMONet_128_64_64(config)
        elif layer == 3:
            if config.KD:
                model = FaFMIMONet_256_32_32_KD(config)
            else:
                model = FaFMIMONet_256_32_32(config, forecast_num = forecast_num)
                # model = nn.DataParallel(model)
        else:
            if config.KD:
                model = FaFMIMONet_512_16_16_KD(config)
            else:
                model = FaFMIMONet_512_16_16(config)

        model = nn.DataParallel(model)
        model = model.to(device)
        # specify optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        model = FaFNet(config)
        model = nn.DataParallel(model)
        model = model.to(device)
        # specify optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # specify creterion
    criterion = {'cls': SoftmaxFocalClassificationLoss(), 'loc': WeightedSmoothL1LocalizationLoss()}

    if config.KD:
        teacher = FaFNet(config)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)

        fafmodule = FaFModuleKD(model, teacher, config, optimizer, criterion)
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher['epoch']
        fafmodule.teacher.load_state_dict(checkpoint_teacher['model_state_dict'])
        print("Load teacher model from {}, at epoch {}".format(args.resume_teacher, start_epoch_teacher))
    else:
        fafmodule = FaFModule(model, config, optimizer, criterion)
    if args.resume != '' or args.mode == 'val':
        checkpoint = torch.load(args.resume)
        model_save_path = args.resume[:args.resume.rfind('/')]
        start_epoch = checkpoint['epoch'] + 1
        if only_load_model:
            start_epoch = 0
        if config.MGDA:
            fafmodule.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            fafmodule.head.load_state_dict(checkpoint['head_state_dict'])

            if not only_load_model:
                fafmodule.scheduler_encoder.load_state_dict(checkpoint['scheduler_encoder_state_dict'])
                fafmodule.optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
                fafmodule.scheduler_head.load_state_dict(checkpoint['scheduler_head_state_dict'])
                fafmodule.optimizer_head.load_state_dict(checkpoint['optimizer_head_state_dict'])
        else:
            # net1 = fafmodule.model
            # model1_dict = net1.state_dict()
            model1_dict = fafmodule.model.state_dict()
            state_1_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model1_dict.keys()}
            for key in state_1_dict.keys():
                print(key)
            print(len(state_1_dict.keys()))
            print(len(model1_dict.keys()))
            model1_dict.update(state_1_dict)
            fafmodule.model.load_state_dict(model1_dict)
            for key in checkpoint['model_state_dict'].keys():
                if key not in model1_dict.keys():
                    print(key)
            if not only_load_model:
                net2 = fafmodule.scheduler
                model2_dict = net2.state_dict()
                state_2_dict = {k:v for k,v in checkpoint['scheduler_state_dict'].items() if k in model2_dict.keys()}
                model2_dict.update(state_2_dict)
                fafmodule.scheduler.load_state_dict(model2_dict)

                # net3 = fafmodule.optimizer
                # model3_dict = net3.state_dict()
                # state_3_dict = {k.strip('module.'):v for k,v in checkpoint['optimizer_state_dict'].items() if k.strip('module.') in model3_dict.keys()}
                # model3_dict.update(state_3_dict)
                # fafmodule.optimizer.load_state_dict(model3_dict)
                # fafmodule.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # fafmodule.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

        if args.mode == 'train':
            for k,v in fafmodule.model.named_parameters(): 
                for para in untrainable_list:
                    if para in k: #or ('decoder' in k): 
                        v.requires_grad = False
            for k,v in fafmodule.model.named_parameters(): 
                if v.requires_grad == True:
                    print(k,': ', v.requires_grad)
                # print(k)
                # if 'encoder' in k: 
                #     v.requires_grad=False
        
        fafmodule.optimizer = optim.Adam(filter(lambda p: p.requires_grad, fafmodule.model.parameters()), lr=args.lr)

    if args.mode == 'train':

        n_train = len(trainset)
        indices = list(range(n_train))
        data_cache = {}
        for epoch in range(start_epoch, num_epochs + 1):
            trainset.seq_dict[0] = trainset.get_data_dict(trainset.dataset_root_peragent)
            if config.MGDA:
                lr = fafmodule.optimizer_head.param_groups[0]['lr']
            else:
                lr = fafmodule.optimizer.param_groups[0]['lr']
            print("Epoch {}, learning rate {}".format(epoch, lr))

            if need_log:
                saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
                saver.flush()

            running_loss_disp = AverageMeter('Total loss', ':.6f')  # for motion prediction error
            running_loss_class = AverageMeter('classification Loss', ':.6f')  # for cell classification error
            running_loss_loc = AverageMeter('Localization Loss', ':.6f')  # for state estimation error
            if config.forecast_loss == 'True':
                running_loss_fore = AverageMeter('Forecast Loss', ':.6f')  # for forecast estimation error

            if config.MGDA:
                fafmodule.scheduler_encoder.step()
                fafmodule.encoder.train()
                fafmodule.scheduler_head.step()
                fafmodule.head.train()
            else:
                fafmodule.scheduler.step()
                fafmodule.model.train()
                # fafmodule.model.eval()
                if "decoder" in untrainable_list:
                    fafmodule.model.module.decoder.eval()
                if "encoder" in untrainable_list:
                    fafmodule.model.module.decoder.eval()
                if "adafusion" in untrainable_list:
                    fafmodule.model.module.adafusion.eval()
                if "regression" in untrainable_list:
                    fafmodule.model.module.decoder.eval()
                if "classification" in untrainable_list:
                    fafmodule.model.module.decoder.eval()
                if "forecast" in untrainable_list:
                    fafmodule.model.module.decoder.eval()
                    
                
            step_ct = 1
            t = time.time()

            random.shuffle(indices)
            train_sampler = torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.SubsetRandomSampler(indices),
                                                                  batch_size=batch_size, drop_last=False)

            trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=train_sampler, num_workers=num_workers)
            # trainloader1 = torch.utils.data.DataLoader(trainset1, shuffle=False, batch_sampler=train_sampler, num_workers=num_workers)
            # trainloader2 = torch.utils.data.DataLoader(trainset2, shuffle=False, batch_sampler=train_sampler, num_workers=num_workers)
            # trainloader3 = torch.utils.data.DataLoader(trainset3, shuffle=False, batch_sampler=train_sampler, num_workers=num_workers)
            # trainloader4 = torch.utils.data.DataLoader(trainset4, shuffle=False, batch_sampler=train_sampler, num_workers=num_workers)

            # for sample0, sample1, sample2, sample3, sample4 in tqdm(zip(trainloader0, trainloader1, trainloader2, trainloader3, trainloader4)):
            time_10 = time.time()
            for sample in tqdm(trainloader):
                time_t0 = time.time()
                print("????????????", time_t0 - trainset.time_4)
                # padded_voxel_points0, padded_voxel_points_teacher0, label_one_hot0, reg_target0, reg_loss_mask0, anchors_map0, vis_maps0, target_agent_id0, num_sensor0, trans_matrices0,center_agent  = sample
                padded_voxel_points0, padded_voxel_points_teacher0, label_one_hot0, reg_target0, reg_loss_mask0, anchors_map0, vis_maps0, target_agent_id0, num_sensor0, trans_matrices0, filename0, current_pose_rec0, current_cs_rec0 = sample[0]['padded_voxel_points'] ,sample[0]['padded_voxel_points_teacher'] ,sample[0]['label_one_hot'] ,sample[0]['reg_target'] ,sample[0]['reg_loss_mask'] ,sample[0]['anchors_map'] ,sample[0]['vis_maps'] ,sample[0]['target_agent_id'] ,sample[0]['num_sensor'] ,sample[0]['trans_matrices'], sample[0]['filename'], sample[0]['current_pose_rec'], sample[0]['current_cs_rec']
                padded_voxel_points1, padded_voxel_points_teacher1, label_one_hot1, reg_target1, reg_loss_mask1, anchors_map1, vis_maps1, target_agent_id1, num_sensor1, trans_matrices1, filename1, current_pose_rec1, current_cs_rec1 = sample[1]['padded_voxel_points'] ,sample[1]['padded_voxel_points_teacher'] ,sample[1]['label_one_hot'] ,sample[1]['reg_target'] ,sample[1]['reg_loss_mask'] ,sample[1]['anchors_map'] ,sample[1]['vis_maps'] ,sample[1]['target_agent_id'] ,sample[1]['num_sensor'] ,sample[1]['trans_matrices'], sample[1]['filename'], sample[1]['current_pose_rec'], sample[1]['current_cs_rec']
                padded_voxel_points2, padded_voxel_points_teacher2, label_one_hot2, reg_target2, reg_loss_mask2, anchors_map2, vis_maps2, target_agent_id2, num_sensor2, trans_matrices2, filename2, current_pose_rec2, current_cs_rec2 = sample[2]['padded_voxel_points'] ,sample[2]['padded_voxel_points_teacher'] ,sample[2]['label_one_hot'] ,sample[2]['reg_target'] ,sample[2]['reg_loss_mask'] ,sample[2]['anchors_map'] ,sample[2]['vis_maps'] ,sample[2]['target_agent_id'] ,sample[2]['num_sensor'] ,sample[2]['trans_matrices'], sample[2]['filename'], sample[2]['current_pose_rec'], sample[2]['current_cs_rec']
                padded_voxel_points3, padded_voxel_points_teacher3, label_one_hot3, reg_target3, reg_loss_mask3, anchors_map3, vis_maps3, target_agent_id3, num_sensor3, trans_matrices3, filename3, current_pose_rec3, current_cs_rec3 = sample[3]['padded_voxel_points'] ,sample[3]['padded_voxel_points_teacher'] ,sample[3]['label_one_hot'] ,sample[3]['reg_target'] ,sample[3]['reg_loss_mask'] ,sample[3]['anchors_map'] ,sample[3]['vis_maps'] ,sample[3]['target_agent_id'] ,sample[3]['num_sensor'] ,sample[3]['trans_matrices'], sample[3]['filename'], sample[3]['current_pose_rec'], sample[3]['current_cs_rec']
                padded_voxel_points4, padded_voxel_points_teacher4, label_one_hot4, reg_target4, reg_loss_mask4, anchors_map4, vis_maps4, target_agent_id4, num_sensor4, trans_matrices4, filename4, current_pose_rec4, current_cs_rec4 = sample[4]['padded_voxel_points'] ,sample[4]['padded_voxel_points_teacher'] ,sample[4]['label_one_hot'] ,sample[4]['reg_target'] ,sample[4]['reg_loss_mask'] ,sample[4]['anchors_map'] ,sample[4]['vis_maps'] ,sample[4]['target_agent_id'] ,sample[4]['num_sensor'] ,sample[4]['trans_matrices'], sample[4]['filename'], sample[4]['current_pose_rec'], sample[4]['current_cs_rec']
                center_agent = sample['center_agent']
                
                supervise_data = sample['supervise']
                
                # supervise_data = process_supervise_data(supervise_data)

                time_t1 = time.time()
                print("?????????1", time_t1 - time_t0)
                current_pose_rec_list = [current_pose_rec0, current_pose_rec1, current_pose_rec2, current_pose_rec3, current_pose_rec4]
                current_cs_rec0_list = [current_cs_rec0, current_cs_rec1, current_cs_rec2, current_cs_rec3, current_cs_rec4]
                padded_voxel_points_list = [padded_voxel_points0, padded_voxel_points1, padded_voxel_points2, padded_voxel_points3, padded_voxel_points4]
                # label_one_hot_list = [label_one_hot0, label_one_hot1, label_one_hot2, label_one_hot3, label_one_hot4]
                # reg_target_list = [reg_target0, reg_target1, reg_target2, reg_target3, reg_target4]
                # reg_loss_mask_list = [reg_loss_mask0, reg_loss_mask1, reg_loss_mask2, reg_loss_mask3, reg_loss_mask4]
                # anchors_map_list = [anchors_map0, anchors_map1, anchors_map2, anchors_map3, anchors_map4]
                label_one_hot_list = [label_one_hot0]
                reg_target_list = [reg_target0]
                # reg_target_list = [reg_target0, reg_target1, reg_target2, reg_target3, reg_target4]
                reg_loss_mask_list = [reg_loss_mask0]
                # reg_loss_mask_list = [reg_loss_mask0, reg_loss_mask1, reg_loss_mask2, reg_loss_mask3, reg_loss_mask4]
                # anchors_map_list = [anchors_map0, anchors_map1, anchors_map2, anchors_map3, anchors_map4]
                anchors_map_list = [anchors_map0]
                vis_maps_list = [vis_maps0, vis_maps1, vis_maps2, vis_maps3, vis_maps4]
                time_t2 = time.time()
                print("?????????2", time_t2 - time_t1)
                padded_voxel_points = torch.cat(tuple(padded_voxel_points_list), 0)# ???????????????tensor???list ???????????? 
                label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
                reg_target = torch.cat(tuple(reg_target_list), 0)
                reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
                anchors_map = torch.cat(tuple(anchors_map_list), 0)
                vis_maps = torch.cat(tuple(vis_maps_list), 0)
                time_t3 = time.time()
                print("?????????3", time_t3 - time_t2)
                target_agent_id_list = [target_agent_id0, target_agent_id1, target_agent_id2, target_agent_id3, target_agent_id4]
                num_agent_list = [num_sensor0[-1], num_sensor1[-1], num_sensor2[-1], num_sensor3[-1], num_sensor4[-1]]
                trans_matrices_list = [trans_matrices0, trans_matrices1, trans_matrices2, trans_matrices3, trans_matrices4]

                trans_matrices = torch.stack(tuple(trans_matrices_list), 1)  #
                target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
                num_agent = torch.stack(tuple(num_agent_list), 1)
                time_t4 = time.time()
                print("?????????4", time_t4 - time_t3)
                data = {}
                data['current_pose_rec'] = current_pose_rec_list
                data['current_cs_rec'] = current_cs_rec0_list
                data['file_name'] = [filename0, filename1, filename2, filename3, filename4]
                data['bev_seq'] = padded_voxel_points.to(device)
                time_t5_0 = time.time()
                print("?????????5_0", time_t5_0 - time_t4)
                data['labels'] = label_one_hot.to(device)
                data['reg_targets'] = reg_target.to(device)
                data['anchors'] = anchors_map.to(device)
                data['reg_loss_mask'] = reg_loss_mask.to(device).type(dtype=torch.bool)
                data['vis_maps'] = vis_maps.to(device)
                time_t5_1 = time.time()
                print("?????????5_1", time_t5_1 - time_t5_0)
                # data['target_agent_ids'] = target_agent_ids.to(device)
                data['num_agent'] = num_agent.to(device)
                data['trans_matrices'] = trans_matrices
                data['supervise'] = supervise_data
                data['teacher'] = padded_voxel_points_teacher0
                data['filename'] = filename0
                time_8 = time.time()
                time_c = time_8- time_10
                time_t5 = time.time()
                print("?????????5", time_t5 - time_t4)
                print("??????????????????", time_c)
                print("???loader?????????", time_8-trainset.time_4)
                time_9 = time.time()
                if config.KD:
                    padded_voxel_points_list_teacher = [padded_voxel_points_teacher0, padded_voxel_points_teacher1, padded_voxel_points_teacher2, padded_voxel_points_teacher3, padded_voxel_points_teacher4]
                    padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_list_teacher), 0)
                    data['bev_seq_teacher'] = padded_voxel_points_teacher.to(device)
                    data['kd_weight'] = args.kd
                    data['layer'] = layer

                if config.KD:
                    loss, cls_loss, loc_loss,kd_loss = fafmodule.step(data, batch_size, center_agent)
                else:
                    if config.forecast_loss == 'True' and config.forecast_KD == 'False':
                        loss, cls_loss, loc_loss, fore_loss = fafmodule.step(data, batch_size, center_agent, forecast_num, config)
                    elif config.forecast_loss == 'True' and config.forecast_KD == 'True':
                        loss, cls_loss, loc_loss, fore_loss, kd_loss = fafmodule.step(data, batch_size, center_agent, forecast_num, config)
                    else: 
                        loss, cls_loss, loc_loss = fafmodule.step(data, batch_size, center_agent, forecast_num, config)
                running_loss_disp.update(loss)
                running_loss_class.update(cls_loss)
                running_loss_loc.update(loc_loss)
                if config.forecast_loss == 'True':
                    running_loss_fore.update(fore_loss)
                time_10 = time.time()
                print("total_time:", time_10 - time_9)
                step_ct += 1
                print("\nEpoch {}, Step {}".format(epoch, step_ct))
                print("Running total loss: {}".format(running_loss_disp.avg))
                print("Running total cls loss: {}".format(running_loss_class.avg))
                print("Running total loc loss: {}".format(running_loss_loc.avg))
                print("Running total KD loss: {}".format(kd_loss))
                if config.forecast_loss == 'True':
                    print("Running total fore loss: {}".format(running_loss_fore.avg))
            if config.forecast_loss == 'True':
                print("{}\t{}\t{}\t{}\t Takes {} s\n".format(running_loss_disp, running_loss_class, running_loss_loc,running_loss_fore,
                                                        str(time.time() - t)))
            else:
                print("{}\t{}\t{}\t Takes {} s\n".format(running_loss_disp, running_loss_class, running_loss_loc,
                                                        str(time.time() - t)))
            # save model
            if need_log:
                if config.KD:
                    saver.write("{}\t{}\t{}\tkd loss:{} Take {} s\n".format(running_loss_disp,running_loss_class,running_loss_loc,kd_loss,str(time.time()-t)))
                else:
                    saver.write("{}\t{}\t{}\tTake {} s\n".format(running_loss_disp,running_loss_class,running_loss_loc,str(time.time()-t)))

                saver.flush()
                if config.MGDA:
                    save_dict = {'epoch': epoch,
                                 'encoder_state_dict': fafmodule.encoder.state_dict(),
                                 'optimizer_encoder_state_dict': fafmodule.optimizer_encoder.state_dict(),
                                 'scheduler_encoder_state_dict': fafmodule.scheduler_encoder.state_dict(),
                                 'head_state_dict': fafmodule.head.state_dict(),
                                 'optimizer_head_state_dict': fafmodule.optimizer_head.state_dict(),
                                 'scheduler_head_state_dict': fafmodule.scheduler_head.state_dict(),
                                 'loss': running_loss_disp.avg}
                else:
                    save_dict = {'epoch': epoch,
                                 'model_state_dict': fafmodule.model.state_dict(),
                                 'optimizer_state_dict': fafmodule.optimizer.state_dict(),
                                 'scheduler_state_dict': fafmodule.scheduler.state_dict(),
                                 'loss': running_loss_disp.avg}
                torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    elif args.mode == 'val':
        # model_save_path = model_save_path + '/epoch_' + str(start_epoch - 1)
        # check_folder(model_save_path)
        # save_fig_path0 = os.path.join(model_save_path, 'vis_result_agent0')
        # save_fig_path1 = os.path.join(model_save_path, 'vis_result_agent1')
        # save_fig_path2 = os.path.join(model_save_path, 'vis_result_agent2')
        # save_fig_path3 = os.path.join(model_save_path, 'vis_result_agent3')
        # save_fig_path4 = os.path.join(model_save_path, 'vis_result_agent4') 
        # check_folder(save_fig_path0)
        # check_folder(save_fig_path1)
        # check_folder(save_fig_path2)
        # check_folder(save_fig_path3)
        # check_folder(save_fig_path4)
        # save_fig_path = [save_fig_path0, save_fig_path1, save_fig_path2, save_fig_path3, save_fig_path4]

        if config.MGDA:
            fafmodule.encoder.eval()
            fafmodule.head.eval()
        else:
            fafmodule.model.eval()

        running_loss_disp = AverageMeter('Total loss', ':.6f')  # for motion prediction error
        running_loss_class = AverageMeter('classification Loss', ':.6f')  # for cell classification error
        running_loss_loc = AverageMeter('Localization Loss', ':.6f')  # for state estimation error

        # for local and global mAP evaluation
        det_results_local = [[] for i in range(5)]
        annotations_local = [[] for i in range(5)]

        # for sample0, sample1, sample2, sample3, sample4 in zip(valloader0, valloader1, valloader2, valloader3,
        #                                                        valloader4):
        for sample in tqdm(valloader):    
            t = time.time()
            # center_agent = sample['center_agent']
            # padded_voxel_points0, label_one_hot0, reg_target0, reg_loss_mask0, anchors_map0, vis_maps0, gt_max_iou0, filename0, \
            # target_agent_id0, num_sensor0, trans_matrices0, padded_voxel_points_global, reg_target_global, anchors_map_global, gt_max_iou_global, trans_matrices_map = sample[0]['padded_voxel_points'] ,sample[0]['label_one_hot'] ,sample[0]['reg_target'] ,sample[0]['reg_loss_mask'] ,sample[0]['anchors_map'] ,sample[0]['vis_maps'], sample[0]['gt_max_iou'], sample[0]['filename'], sample[0]['target_agent_id'] ,sample[0]['num_sensor'] ,sample[0]['trans_matrices'],sample[0]['padded_voxel_points_global'],sample[0]['reg_target_global'],sample[0]['anchors_map_global'],sample[0]['gt_max_iou_global'],sample[0]['trans_matrices_map']
            # padded_voxel_points1, label_one_hot1, reg_target1, reg_loss_mask1, anchors_map1, vis_maps1, gt_max_iou1, filename1, target_agent_id1, num_sensor1, trans_matrices1, _, _, _, _, _ = sample[1]['padded_voxel_points'], sample[1]['label_one_hot'] ,sample[1]['reg_target'] ,sample[1]['reg_loss_mask'] ,sample[1]['anchors_map'] ,sample[1]['vis_maps'], sample[1]['gt_max_iou'], sample[1]['filename'], sample[1]['target_agent_id'] ,sample[1]['num_sensor'] ,sample[1]['trans_matrices'],sample[1]['padded_voxel_points_global'],sample[1]['reg_target_global'],sample[1]['anchors_map_global'],sample[1]['gt_max_iou_global'],sample[1]['trans_matrices_map']
            # padded_voxel_points2, label_one_hot2, reg_target2, reg_loss_mask2, anchors_map2, vis_maps2, gt_max_iou2, filename2, target_agent_id2, num_sensor2, trans_matrices2, _, _, _, _, _ = sample[2]['padded_voxel_points'], sample[2]['label_one_hot'] ,sample[2]['reg_target'] ,sample[2]['reg_loss_mask'] ,sample[2]['anchors_map'] ,sample[2]['vis_maps'], sample[2]['gt_max_iou'], sample[2]['filename'], sample[2]['target_agent_id'] ,sample[2]['num_sensor'] ,sample[2]['trans_matrices'],sample[2]['padded_voxel_points_global'],sample[2]['reg_target_global'],sample[2]['anchors_map_global'],sample[2]['gt_max_iou_global'],sample[2]['trans_matrices_map']
            # padded_voxel_points3, label_one_hot3, reg_target3, reg_loss_mask3, anchors_map3, vis_maps3, gt_max_iou3, filename3, target_agent_id3, num_sensor3, trans_matrices3, _, _, _, _, _ = sample[3]['padded_voxel_points'] ,sample[3]['label_one_hot'] ,sample[3]['reg_target'] ,sample[3]['reg_loss_mask'] ,sample[3]['anchors_map'] ,sample[3]['vis_maps'], sample[3]['gt_max_iou'], sample[3]['filename'], sample[3]['target_agent_id'] ,sample[3]['num_sensor'] ,sample[3]['trans_matrices'],sample[3]['padded_voxel_points_global'],sample[3]['reg_target_global'],sample[3]['anchors_map_global'],sample[3]['gt_max_iou_global'],sample[3]['trans_matrices_map']
            # padded_voxel_points4, label_one_hot4, reg_target4, reg_loss_mask4, anchors_map4, vis_maps4, gt_max_iou4, filename4, target_agent_id4, num_sensor4, trans_matrices4, _, _, _, _, _ = sample[4]['padded_voxel_points'],sample[4]['label_one_hot'] ,sample[4]['reg_target'] ,sample[4]['reg_loss_mask'] ,sample[4]['anchors_map'] ,sample[4]['vis_maps'], sample[4]['gt_max_iou'], sample[4]['filename'], sample[4]['target_agent_id'] ,sample[4]['num_sensor'] ,sample[4]['trans_matrices'],sample[4]['padded_voxel_points_global'],sample[4]['reg_target_global'],sample[4]['anchors_map_global'],sample[4]['gt_max_iou_global'],sample[4]['trans_matrices_map']
            center_agent = sample['center_agent']
            padded_voxel_points0, label_one_hot0, reg_target0, reg_loss_mask0, anchors_map0, vis_maps0, gt_max_iou0, filename0, \
            target_agent_id0, num_sensor0, trans_matrices0, trans_matrices_map, filename0, current_pose_rec0, current_cs_rec0  = sample[0]['padded_voxel_points'] ,sample[0]['label_one_hot'] ,sample[0]['reg_target'] ,sample[0]['reg_loss_mask'] ,sample[0]['anchors_map'] ,sample[0]['vis_maps'], sample[0]['gt_max_iou'], sample[0]['filename'], sample[0]['target_agent_id'] ,sample[0]['num_sensor'] ,sample[0]['trans_matrices'],sample[0]['trans_matrices_map'], sample[0]['filename'], sample[0]['current_pose_rec'], sample[0]['current_cs_rec']
            padded_voxel_points1, label_one_hot1, reg_target1, reg_loss_mask1, anchors_map1, vis_maps1, gt_max_iou1, filename1, target_agent_id1, num_sensor1, trans_matrices1, _, filename1, current_pose_rec1, current_cs_rec1 = sample[1]['padded_voxel_points'], sample[1]['label_one_hot'] ,sample[1]['reg_target'] ,sample[1]['reg_loss_mask'] ,sample[1]['anchors_map'] ,sample[1]['vis_maps'], sample[1]['gt_max_iou'], sample[1]['filename'], sample[1]['target_agent_id'] ,sample[1]['num_sensor'] ,sample[1]['trans_matrices'],sample[1]['trans_matrices_map'], sample[1]['filename'], sample[1]['current_pose_rec'], sample[1]['current_cs_rec']
            padded_voxel_points2, label_one_hot2, reg_target2, reg_loss_mask2, anchors_map2, vis_maps2, gt_max_iou2, filename2, target_agent_id2, num_sensor2, trans_matrices2, _, filename2, current_pose_rec2, current_cs_rec2 = sample[2]['padded_voxel_points'], sample[2]['label_one_hot'] ,sample[2]['reg_target'] ,sample[2]['reg_loss_mask'] ,sample[2]['anchors_map'] ,sample[2]['vis_maps'], sample[2]['gt_max_iou'], sample[2]['filename'], sample[2]['target_agent_id'] ,sample[2]['num_sensor'] ,sample[2]['trans_matrices'],sample[2]['trans_matrices_map'], sample[2]['filename'], sample[2]['current_pose_rec'], sample[2]['current_cs_rec']
            padded_voxel_points3, label_one_hot3, reg_target3, reg_loss_mask3, anchors_map3, vis_maps3, gt_max_iou3, filename3, target_agent_id3, num_sensor3, trans_matrices3, _, filename3, current_pose_rec3, current_cs_rec3 = sample[3]['padded_voxel_points'] ,sample[3]['label_one_hot'] ,sample[3]['reg_target'] ,sample[3]['reg_loss_mask'] ,sample[3]['anchors_map'] ,sample[3]['vis_maps'], sample[3]['gt_max_iou'], sample[3]['filename'], sample[3]['target_agent_id'] ,sample[3]['num_sensor'] ,sample[3]['trans_matrices'],sample[3]['trans_matrices_map'], sample[3]['filename'], sample[3]['current_pose_rec'], sample[3]['current_cs_rec']
            padded_voxel_points4, label_one_hot4, reg_target4, reg_loss_mask4, anchors_map4, vis_maps4, gt_max_iou4, filename4, target_agent_id4, num_sensor4, trans_matrices4, _, filename4, current_pose_rec4, current_cs_rec4 = sample[4]['padded_voxel_points'],sample[4]['label_one_hot'] ,sample[4]['reg_target'] ,sample[4]['reg_loss_mask'] ,sample[4]['anchors_map'] ,sample[4]['vis_maps'], sample[4]['gt_max_iou'], sample[4]['filename'], sample[4]['target_agent_id'] ,sample[4]['num_sensor'] ,sample[4]['trans_matrices'],sample[4]['trans_matrices_map'], sample[4]['filename'], sample[4]['current_pose_rec'], sample[4]['current_cs_rec']
            padded_voxel_points_list = [padded_voxel_points0, padded_voxel_points1, padded_voxel_points2,
                                        padded_voxel_points3, padded_voxel_points4]
            # label_one_hot_list = [label_one_hot0, label_one_hot1, label_one_hot2, label_one_hot3, label_one_hot4]
            # reg_target_list = [reg_target0, reg_target1, reg_target2, reg_target3, reg_target4]
            # reg_loss_mask_list = [reg_loss_mask0, reg_loss_mask1, reg_loss_mask2, reg_loss_mask3, reg_loss_mask4]
            # anchors_map_list = [anchors_map0, anchors_map1, anchors_map2, anchors_map3, anchors_map4]
            label_one_hot_list = [label_one_hot0]
            reg_target_list = [reg_target0]
            current_pose_rec_list = [current_pose_rec0, current_pose_rec1, current_pose_rec2, current_pose_rec3, current_pose_rec4]
            current_cs_rec0_list = [current_cs_rec0, current_cs_rec1, current_cs_rec2, current_cs_rec3, current_cs_rec4]
            # reg_target_list = [reg_target0, reg_target1, reg_target2, reg_target3, reg_target4]
            reg_loss_mask_list = [reg_loss_mask0]
            # reg_loss_mask_list = [reg_loss_mask0, reg_loss_mask1, reg_loss_mask2, reg_loss_mask3, reg_loss_mask4]
            # anchors_map_list = [anchors_map0, anchors_map1, anchors_map2, anchors_map3, anchors_map4]
            anchors_map_list = [anchors_map0]
            vis_maps_list = [vis_maps0, vis_maps1, vis_maps2, vis_maps3, vis_maps4]
            gt_max_iou = [gt_max_iou0, gt_max_iou1, gt_max_iou2, gt_max_iou3, gt_max_iou4]
            target_agent_id_list = [target_agent_id0, target_agent_id1, target_agent_id2, target_agent_id3,
                                    target_agent_id4]
            num_agent_list = [torch.tensor(num_sensor0), torch.tensor(num_sensor1), torch.tensor(num_sensor2), torch.tensor(num_sensor3), torch.tensor(num_sensor4)]

            trans_matrices_list = [trans_matrices0, trans_matrices1, trans_matrices2, trans_matrices3, trans_matrices4]
            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)  #
            target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
            num_agent = torch.stack(tuple(num_agent_list), 1)

            padded_voxel_points = torch.cat(tuple(padded_voxel_points_list), 0)
            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {}
            data['current_pose_rec'] = current_pose_rec_list
            data['current_cs_rec'] = current_cs_rec0_list
            data['bev_seq'] = padded_voxel_points.to(device)
            data['labels'] = label_one_hot.to(device)
            data['reg_targets'] = reg_target.to(device)
            data['anchors'] = anchors_map.to(device)
            data['vis_maps'] = vis_maps.to(device)
            data['reg_loss_mask'] = reg_loss_mask.to(device).type(dtype=torch.bool)
            data['file_name'] = [filename0, filename1, filename2, filename3, filename4]
            data['target_agent_ids'] = target_agent_ids.to(device)
            data['num_agent'] = num_agent.to(device)
            data['trans_matrices'] = trans_matrices
            data['filename'] = filename0

            loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, True, center_agent, forecast_num, config)
            
            # local qualitative evaluation
            # for k in range(int(num_sensor0[0])):
            for k in range(1):
                data_agents = {}
                data_agents['bev_seq'] = torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1)
                data_agents['reg_targets'] = torch.unsqueeze(reg_target[k, :, :, :, :, :], 0)
                data_agents['anchors'] = torch.unsqueeze(anchors_map[k, :, :, :, :], 0)
                temp = gt_max_iou[k]
                data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
                # data_agents['gt_max_iou'] = temp[0]['gt_box'][0]
                # data_agents['gt_max_iou'] = [0]
                result_temp = result[k]

                temp = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp[0][0],
                        'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
                        'anchors_map': data_agents['anchors'].cpu().numpy()[0],
                        'gt_max_iou': data_agents['gt_max_iou']}
                det_results_local[k], annotations_local[k], det_results_local_temp, annotations_local_temp = cal_local_mAP(config, temp, det_results_local[k], annotations_local[k])
                temp['det_results_local_temp'] = det_results_local_temp
                temp['annotations_local_temp'] = annotations_local_temp
                map_result_0_5, _ = eval_map([det_results_local_temp], [annotations_local_temp], scale_ranges=None,iou_thr=0.5,dataset=None, logger=None)

                map_result_0_7, _ = eval_map([det_results_local_temp], [annotations_local_temp], scale_ranges=None,iou_thr=0.7,dataset=None, logger=None)

                filename = str(filename0[0][0])
                cut = filename[filename.rfind('agent') + 7:]
                seq_name = cut[:cut.rfind('_')]
                idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
                # seq_save = os.path.join(save_fig_path[k], seq_name)
                # check_folder(seq_save)
                idx = filename0[-1].split('/')[-1]
                scene = idx.split('_')[0]
                idx_save = str(idx) + '.png'
                pth_save = str(idx)
                if args.visualization:
                    seq_save = args.resume.strip('.pth') +  '_' + str(latency_lambda[0])
                    check_folder(seq_save)
                    seq_save = seq_save  + '/' + scene
                    check_folder(seq_save)
                    # np.save(os.path.join(seq_save, pth_save), temp)
                    visualization(config, temp, os.path.join(seq_save, idx_save),map_result_0_5, map_result_0_7)

            print("Validation scene {}, at frame {}".format(seq_name, idx))
            running_loss_disp.update(loss)
            running_loss_class.update(cls_loss)
            running_loss_loc.update(loc_loss)
            print("{}\t{}\t{}\t Takes {} s\n".format(running_loss_disp, running_loss_class, running_loss_loc,
                                                     str(time.time() - t)))

        print("Quantitative evaluation results of model from {}, at epoch {}".format(args.resume, start_epoch - 1))

        log_results_file = args.logname + '.txt'
        saver_val = open(log_results_file,'w')

        det_results_all_local = det_results_local[0] + det_results_local[1] + det_results_local[2] + det_results_local[3]
        annotations_all_local = annotations_local[0] + annotations_local[1] + annotations_local[2] + annotations_local[3]
        saver_val.write('\noverall local results@iou0.5\n')

        mean_ap_local_average, _ = eval_map(det_results_all_local,annotations_all_local,scale_ranges=None,iou_thr=0.5,dataset=None,logger=None)
        print(mean_ap_local_average)
        saver_val.write(str(mean_ap_local_average))

        saver_val.write('\noverall local results@iou0.7\n')
        mean_ap_local_average, _ = eval_map(det_results_all_local,annotations_all_local,scale_ranges=None,iou_thr=0.7,dataset=None,logger=None)
        print(mean_ap_local_average)
        saver_val.write(str(mean_ap_local_average))

        #local mAP evaluation
        for k in range(1):
            saver_val.write('\nlocal{} results@iou0.5\n'.format(k+1))
            mean_ap, _ = eval_map(det_results_local[k],annotations_local[k],scale_ranges=None,iou_thr=0.5,dataset=None,logger=None)
            print(mean_ap)
            saver_val.write(str(mean_ap))
            saver_val.write('\nlocal{} results@iou0.7\n'.format(k+1))

            mean_ap, _ = eval_map(det_results_local[k],annotations_local[k],scale_ranges=None,iou_thr=0.7,dataset=None,logger=None)
            print(mean_ap)
            saver_val.write(str(mean_ap))

    else:
        print('Not implemented yet.')
    # if need_log:
    #     saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--resume_teacher', default='/DATA_SSD/slren/teacher_aug_batch_4_epoch_100.pth', type=str, help='The path to the saved teacher model that is loaded to resume training')
    parser.add_argument('--kd', default=100000, type=float, help='kd_weight')
    parser.add_argument('--model_only', action='store_true', help='only load model')
    parser.add_argument('--batch', default=2, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--layer', default=3, type=int, help='Communicate which layer')
    parser.add_argument('--nworker', default=0, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./log', help='The path to the output log file')
    parser.add_argument('--mode', default=None, help='Train/Val mode')
    parser.add_argument('--visualization', default=False, help='Visualize validation result')
    parser.add_argument('--latency_lambda', default=[1,1,1,1,1], nargs='+', type=int, help='How many frames do you want to use in forecast')
    parser.add_argument('--binary', default=True, type=bool, help='Only detect car')
    parser.add_argument('--only_det', default=True, type=bool, help='Only do detection')
    parser.add_argument('--logname', default='./log', type=str, help='log the detection performance')
    parser.add_argument('--forecast_model', default='MotionLSTM', type=str, help='Forecast model')
    parser.add_argument('--forecast_loss', default='True', type=str, help='Whether to use Forecast Loss')
    parser.add_argument('--forecast_KD', default='False', type=str, help='Whether to use Forecast KD')
    parser.add_argument('--forecast_num', default=4, type=int, help='How many frames do you want to use in forecast')
    parser.add_argument('--utp', default=['encoder','decoder','classification','regression'], nargs='+', type=str, help='untrain 6able parameters')
    
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()
    print(args)
    config = Config('train', binary=args.binary, only_det=args.only_det)
    config_global = ConfigGlobal('train', binary=args.binary, only_det=args.only_det)
    main(config, config_global, args)