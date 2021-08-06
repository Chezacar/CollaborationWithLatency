# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
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
from data.Dataset_com import NuscenesDataset, CarscenesDataset
from data.config_com import Config, ConfigGlobal
from utils.mean_ap import eval_map

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


def main(config,config_global,args):
    need_log = args.log
    only_load_model = args.model_only
    start_epoch = 1

    # communicate a single layer [0: 32*256*256, 1: 64*128*128, 2: 128*64*64, 3: 256*32*32,  4: 512*16*16] [C, W, H]
    layer = args.layer

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.mode == 'val':
        data_nuscenes = NuscenesDataset(dataset_root=args.data+'/agent0',config=config,split ='val',val=True)

        padded_voxel_points_example, label_one_hot_example, reg_target_example, reg_loss_mask_example,\
        anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example, _, _ = data_nuscenes[0]

        valset0 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example,reg_target_example, reg_loss_mask_example, \
        anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example, dataset_root=args.data+'/agent0',config=config,config_global = config_global,split ='val',val=True)

        valset1 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example,reg_target_example, reg_loss_mask_example, \
        anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example, dataset_root=args.data+'/agent1',config=config,config_global = config_global,split ='val',val=True)

        valset2 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example,reg_target_example, reg_loss_mask_example, \
        anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example, dataset_root=args.data+'/agent2',config=config,config_global = config_global,split ='val',val=True)

        valset3 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example,reg_target_example, reg_loss_mask_example, \
        anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example, dataset_root=args.data+'/agent3',config=config,config_global = config_global,split ='val',val=True)

        valset4 = CarscenesDataset(padded_voxel_points_example, label_one_hot_example,reg_target_example, reg_loss_mask_example, \
        anchors_map_example, motion_one_hot_example, motion_mask_example, vis_maps_example, dataset_root=args.data+'/agent4',config=config,config_global = config_global,split ='val',val=True)

        valloader0 = torch.utils.data.DataLoader(valset0, batch_size=1, shuffle=False, num_workers=1)
        valloader1 = torch.utils.data.DataLoader(valset1, batch_size=1, shuffle=False, num_workers=1)
        valloader2 = torch.utils.data.DataLoader(valset2, batch_size=1, shuffle=False, num_workers=1)
        valloader3 = torch.utils.data.DataLoader(valset3, batch_size=1, shuffle=False, num_workers=1)
        valloader4 = torch.utils.data.DataLoader(valset4, batch_size=1, shuffle=False, num_workers=1)

        print("Validation dataset size:", len(valset0))

    #build model
    if config.MGDA:
        encoder = FeatEncoder()
        encoder = nn.DataParallel(encoder)
        encoder = encoder.to(device)
        optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr)
        head = FaFMGDA(config)
        head = nn.DataParallel(head)
        head = head.to(device)
        optimizer_head = optim.Adam(head.parameters(), lr=args.lr)

        model = [encoder,head]
        optimizer =[optimizer_encoder,optimizer_head]
    elif config.MIMO:
        if layer == 0:
            model = FaFMIMONet_32_256_256(config)
        elif layer == 1:
            model = FaFMIMONet_64_128_128(config)
        elif layer == 2:
            model = FaFMIMONet_128_64_64(config)
        elif layer == 3:
            model = FaFMIMONet_256_32_32(config)
        else:
            model = FaFMIMONet_512_16_16(config)
        model = nn.DataParallel(model)
        model = model.to(device)   
        #specify optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)     
    else:
        model = FaFNet(config)
        model = nn.DataParallel(model)
        model = model.to(device)
        #specify optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #specify creterion
    criterion = {'cls': SoftmaxFocalClassificationLoss(),'loc':WeightedSmoothL1LocalizationLoss() }
    fafmodule = FaFModule(model, config, optimizer, criterion)


    if args.resume != '' or args.mode=='val':
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

            fafmodule.model.load_state_dict(checkpoint['model_state_dict'])
            if not only_load_model:
                fafmodule.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                fafmodule.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    if args.mode=='val':
        model_save_path = model_save_path + '/epoch_' + str(start_epoch-1)
        check_folder(model_save_path)
        save_fig_path_global = os.path.join(model_save_path,'vis_result_global')
        save_fig_path0_cs = os.path.join(model_save_path,'vis_result_agent0_cs')
        save_fig_path1_cs = os.path.join(model_save_path,'vis_result_agent1_cs')
        save_fig_path2_cs = os.path.join(model_save_path,'vis_result_agent2_cs')
        save_fig_path3_cs = os.path.join(model_save_path,'vis_result_agent3_cs')
        save_fig_path4_cs = os.path.join(model_save_path,'vis_result_agent4_cs')


        check_folder(save_fig_path_global)
        check_folder(save_fig_path0_cs)
        check_folder(save_fig_path1_cs)
        check_folder(save_fig_path2_cs)
        check_folder(save_fig_path3_cs)
        check_folder(save_fig_path4_cs)

        save_fig_path_cs = [save_fig_path0_cs, save_fig_path1_cs, save_fig_path2_cs, save_fig_path3_cs, save_fig_path4_cs]

        if config.MGDA:
            fafmodule.encoder.eval()
            fafmodule.head.eval()
        else:
            fafmodule.model.eval()
        running_loss_disp = AverageMeter('Total loss', ':.6f')  # for motion prediction error
        running_loss_class = AverageMeter('classification Loss', ':.6f')  # for cell classification error
        running_loss_loc = AverageMeter('Localization Loss', ':.6f')  # for state estimation error

        # for local and global mAP evaluation
        det_results_local_consensus = [[] for i in range(5)]
        annotations_local_consensus = [[] for i in range(5)]
        det_results_global = []
        annotations_global = []

        total_message_size = 0


        for sample0, sample1, sample2, sample3, sample4 in zip(valloader0, valloader1, valloader2, valloader3, valloader4):
            t = time.time()
            padded_voxel_points0, label_one_hot0, reg_target0, reg_loss_mask0, anchors_map0, vis_maps0, gt_max_iou0, filename0,\
            target_agent_id0, num_sensor0, trans_matrices0, padded_voxel_points_global, reg_target_global, anchors_map_global, gt_max_iou_global, trans_matrices_map = sample0

            padded_voxel_points1, label_one_hot1, reg_target1, reg_loss_mask1, anchors_map1, vis_maps1, gt_max_iou1, filename1, target_agent_id1, num_sensor1, trans_matrices1, _, _, _, _, _ = sample1
            padded_voxel_points2, label_one_hot2, reg_target2, reg_loss_mask2, anchors_map2, vis_maps2, gt_max_iou2, filename2, target_agent_id2, num_sensor2, trans_matrices2, _, _, _, _, _ = sample2
            padded_voxel_points3, label_one_hot3, reg_target3, reg_loss_mask3, anchors_map3, vis_maps3, gt_max_iou3, filename3, target_agent_id3, num_sensor3, trans_matrices3, _, _, _, _, _ = sample3
            padded_voxel_points4, label_one_hot4, reg_target4, reg_loss_mask4, anchors_map4, vis_maps4, gt_max_iou4, filename4, target_agent_id4, num_sensor4, trans_matrices4, _, _, _, _, _ = sample4

            padded_voxel_points_list = [padded_voxel_points0, padded_voxel_points1, padded_voxel_points2, padded_voxel_points3, padded_voxel_points4]
            label_one_hot_list = [label_one_hot0, label_one_hot1, label_one_hot2, label_one_hot3, label_one_hot4]
            reg_target_list = [reg_target0, reg_target1, reg_target2, reg_target3, reg_target4]
            reg_loss_mask_list = [reg_loss_mask0, reg_loss_mask1, reg_loss_mask2, reg_loss_mask3, reg_loss_mask4]
            anchors_map_list = [anchors_map0, anchors_map1, anchors_map2, anchors_map3, anchors_map4]
            vis_maps_list = [vis_maps0, vis_maps1, vis_maps2, vis_maps3, vis_maps4]

            padded_voxel_points = torch.cat(tuple(padded_voxel_points_list), 0)
            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            gt_max_iou = [gt_max_iou0, gt_max_iou1, gt_max_iou2, gt_max_iou3, gt_max_iou4]
            target_agent_id_list = [target_agent_id0, target_agent_id1, target_agent_id2, target_agent_id3, target_agent_id4]
            num_agent_list = [num_sensor0, num_sensor1, num_sensor2, num_sensor3, num_sensor4]
            trans_matrices_list = [trans_matrices0, trans_matrices1, trans_matrices2, trans_matrices3, trans_matrices4]

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            trans_matrices_map = torch.squeeze(trans_matrices_map)
            target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
            num_agent = torch.stack(tuple(num_agent_list), 1)

            data = {}
            data['bev_seq'] = padded_voxel_points.to(device)
            data['labels'] = label_one_hot.to(device)
            data['reg_targets'] = reg_target.to(device)
            data['anchors'] = anchors_map.to(device)
            data['vis_maps'] = vis_maps.to(device)
            data['reg_loss_mask'] = reg_loss_mask.to(device).type(dtype=torch.bool)

            data['target_agent_ids'] = target_agent_ids.to(device)
            data['num_agent'] = num_agent.to(device)
            data['trans_matrices'] = trans_matrices

            loss, cls_loss, loc_loss, local_results_af_local_nms, global_result, sample_bandwidth_two_nms =  fafmodule.predict_all_with_box_com(data, trans_matrices_map)
            total_message_size = total_message_size + sample_bandwidth_two_nms

            # consensus-based local agents qualitative evaluation
            for k in range(num_sensor0):
                data_agents = {}
                data_agents['bev_seq'] = torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1)
                data_agents['reg_targets'] = torch.unsqueeze(reg_target[k, :, :, :, :, :], 0)
                data_agents['anchors'] = torch.unsqueeze(anchors_map[k, :, :, :, :], 0)
                temp = gt_max_iou[k]
                data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
                result_temp = local_results_af_local_nms[k]

                temp = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp[0], \
                        'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
                        'anchors_map': data_agents['anchors'].cpu().numpy()[0], \
                        'gt_max_iou': data_agents['gt_max_iou']}
                det_results_local_consensus[k], annotations_local_consensus[k] = cal_local_mAP(config, temp, det_results_local_consensus[k], annotations_local_consensus[k])

                filename = str(filename0[0][0])
                cut = filename[filename.rfind('agent') + 7:]
                seq_name = cut[:cut.rfind('_')]
                idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
                seq_save = os.path.join(save_fig_path_cs[k], seq_name)
                check_folder(seq_save)
                idx_save = str(idx) + '.png'

                if args.visualization:
                    visualization(config, temp, os.path.join(seq_save, idx_save))

            # global qualitative evaluation
            global_temp = {'bev_seq': padded_voxel_points_global[0, -1].cpu().numpy(), 'result': global_result[0],
                            'reg_targets': reg_target_global[0], \
                            'anchors_map': anchors_map_global[0],
                            'gt_max_iou': gt_max_iou_global[0]['gt_box_global'][0, :, :]}
            det_results_global, annotations_global = cal_global_mAP(config_global, global_temp, det_results_global,
                                                                        annotations_global)

            filename = str(filename0[0][0])
            cut = filename[filename.rfind('agent') + 7:]
            seq_name = cut[:cut.rfind('_')]
            idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
            seq_save = os.path.join(save_fig_path_global, seq_name)
            check_folder(seq_save)
            idx_save = str(idx) + '.png'

            if args.visualization:
                visualization(config_global, global_temp, os.path.join(seq_save, idx_save))

            print("Validation scene {}, at frame {}".format(seq_name, idx))

            running_loss_disp.update(loss)
            running_loss_class.update(cls_loss)
            running_loss_loc.update(loc_loss)
            print("{}\t{}\t{}\t Takes {} s\n".format(running_loss_disp, running_loss_class, running_loss_loc, str(time.time() - t)))

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                              finished all the epoch, calculate the mAP

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        # start to print the mAP
        print("Average communication message size in 1100 samples {}".format(total_message_size / 1100))
        print("Quantitative evaluation results of model from {}, at epoch {}".format(args.resume, start_epoch - 1))

        # local mAP evaluation with consensus communication
        for k in range(4):
            mean_ap, _ = eval_map(det_results_local_consensus[k], annotations_local_consensus[k], scale_ranges=None, iou_thr=0.5, dataset=None, logger=None)
            print("Local mAP@0.5 of agent {} after box communication".format(k))
            print(mean_ap)
            mean_ap, _ = eval_map(det_results_local_consensus[k], annotations_local_consensus[k], scale_ranges=None, iou_thr=0.7, dataset=None, logger=None)
            print("Local mAP@0.7 of agent {} after box communication".format(k))
            print(mean_ap)

        # average local consensus-based mAP evaluation
        det_results_all_local_cs = det_results_local_consensus[0] + det_results_local_consensus[1] + det_results_local_consensus[2] + det_results_local_consensus[3]
        annotations_all_local_cs = annotations_local_consensus[0] + annotations_local_consensus[1] + annotations_local_consensus[2] + annotations_local_consensus[3]
        mean_ap_local_average_cs, _ = eval_map(det_results_all_local_cs, annotations_all_local_cs, scale_ranges=None, iou_thr=0.5, dataset=None, logger=None)
        print("Average local mAP@0.5 after box communication")
        print(mean_ap_local_average_cs)
        mean_ap_local_average_cs, _ = eval_map(det_results_all_local_cs, annotations_all_local_cs, scale_ranges=None, iou_thr=0.7, dataset=None, logger=None)
        print("Average local mAP@0.7 after box communication")
        print(mean_ap_local_average_cs)

        # global mAP@0.5 evaluation
        mean_ap_global, _ = eval_map(det_results_global, annotations_global, scale_ranges=None, iou_thr=0.5, dataset=None, logger=None)
        print("Global mAP@0.5 after box consensus")
        print(mean_ap_global)
        # global mAP@0.7 evaluation
        mean_ap_global, _ = eval_map(det_results_global, annotations_global, scale_ranges=None, iou_thr=0.7, dataset=None, logger=None)
        print("Global mAP@0.7 after box consensus")
        print(mean_ap_global)

    else:
        print('Not implemented yet.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--model_only', action='store_true',help='only load model')
    parser.add_argument('--batch', default=2, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--nworker', default=2, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.000001, type=float, help='Initial learning rate')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./log', help='The path to the output log file')
    parser.add_argument('--mode', default=None, help='Train/Val mode')
    parser.add_argument('--visualization', default=True, help='Visualize validation result')
    parser.add_argument('--binary', default=True,type=bool, help='Only detect car')
    parser.add_argument('--only_det', default=True,type=bool, help='Only do detection')
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()
    print(args)
    config = Config('train',binary = args.binary,only_det=args.only_det)
    config_global = ConfigGlobal('train',binary = args.binary,only_det=args.only_det)
    main(config,config_global,args)