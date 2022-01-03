from posixpath import join
from numpy.lib.npyio import save
import torch
import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from utils.postprocess import *
import torch.nn.functional as F
from data.obj_util import *
from torch import stack as tstack
import argparse
# from utils.mean_ap import eval_map
from utils.mean_ap import *
from data.config_com import Config, ConfigGlobal
from tqdm import tqdm


def eval_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             logger=None,
             nproc=4):
    """Evaluate mAP of a dataset.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        if dataset in ['det', 'vid']:
            tpfp_func = tpfp_imagenet
        else:
            tpfp_func = tpfp_default
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (
                    bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    # print_map_summary(
        # mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def visualization(config, data, data_pre, savename=None):

    map_result_0_5, _ = eval_map([data['det_results_local_temp']], [data['annotations_local_temp']], scale_ranges=None,iou_thr=0.5,dataset=None, logger=None)

    map_result_0_7, _ = eval_map([data['det_results_local_temp']], [data['annotations_local_temp']], scale_ranges=None,iou_thr=0.7,dataset=None, logger=None)

    map_result_0_5_pre, _ = eval_map([data_pre['det_results_local_temp']], [data_pre['annotations_local_temp']], scale_ranges=None,iou_thr=0.5,dataset=None, logger=None)

    map_result_0_7_pre, _ = eval_map([data_pre['det_results_local_temp']], [data_pre['annotations_local_temp']], scale_ranges=None,iou_thr=0.7,dataset=None, logger=None)
    
    voxel_size = config.voxel_size
    area_extents = config.area_extents
    anchor_size = config.anchor_size
    map_dims = config.map_dims
    pred_len = 1
    box_code_size = 6  # (x,y,w,h,sin,cos)

    # voxel map visualization
    voxel = data['bev_seq']
    maps = np.max(voxel, axis=-1)

    anchors_map = data['anchors_map']
    # print(anchors_map.shape)
    # anchor_corners_list = get_anchor_corners_list(anchors_map,box_code_size)
    # anchor_corners_map = anchor_corners_list.reshape(map_dims[0],map_dims[1],len(anchor_size),4,2)
    reg_targets = data['reg_targets']

    # 'pred':box_corners[selected_idx],'score': cls_preds[selected_idx,i]},'selected_idx': selected_idx
    pred_selected = data['result']
    pred_selected_pre = data_pre['result']

    gt_max_iou_idx = data['gt_max_iou']
    # gt_max_iou_idx_pre = data['gt_max_iou']
    # if anchors_map.shape[2] < 7:#binary classification only has 4 anchors
    #	anchors_map = np.concatenate([anchors_map[:,:,:2],np.zeros_like(anchors_map[:,:,:3]),anchors_map[:,:,2:]],axis=2)

    #	reg_targets = np.concatenate([reg_targets[:,:,:2],np.zeros_like(reg_targets[:,:,:3]),reg_targets[:,:,2:]],axis=2)

    plt.clf()
    if config.pred_type == 'motion':
        cur_det = []
    for p in range(pred_len):
        # p=0
        for k in range(len(pred_selected)):

            cls_pred_corners = pred_selected[k]['pred'][:, p]
            cls_pred_scores = pred_selected[k]['score']

            cls_pred_corners_pre = pred_selected_pre[k]['pred'][:, p]
            cls_pred_scores_pre = pred_selected_pre[k]['score']

            # cls_pred_idx = pred_selected[k]['selected_idx']
            if config.motion_state:
                cls_pred_state = pred_selected[k]['motion']
                cls_pred_state_pre = pred_selected_pre[k]['motion']

            for corner_id in range(cls_pred_corners.shape[0]):
                corner_box = cls_pred_corners[corner_id]

                corners = coor_to_vis(corner_box, area_extents=area_extents, voxel_size=voxel_size)
                # corners = corner_box
                c_x, c_y = np.mean(corners, axis=0)
                corners = np.concatenate([corners, corners[[0]]])

                if p == 0:
                    if config.motion_state:
                        if cls_pred_state[corner_id] == 0:
                            color = 'y'
                        else:
                            color = 'r'
                    else:
                        color = 'r'
                    plt.plot(corners[:, 0], corners[:, 1], c=color, linewidth=0.8, zorder=15)
                    plt.scatter(c_x, c_y, s=3, c=color, zorder=15)
                    # plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                    plt.plot([c_x, (corners[-2][0] + corners[0][0]) / 2.], [c_y, (corners[-2][1] + corners[0][1]) / 2.],
                             linewidth=0.8, c=color, zorder=15)
                else:
                    color = 'r'
                    if config.motion_state:
                        if cls_pred_state[corner_id] == 0:
                            continue
                    plt.scatter(c_x, c_y, s=3, c=color, zorder=15)

            for corner_id_pre in range(cls_pred_corners_pre.shape[0]):
                corner_box_pre = cls_pred_corners_pre[corner_id_pre]

                corners_pre = coor_to_vis(corner_box_pre, area_extents=area_extents, voxel_size=voxel_size)
                # corners_pre = corner_box_pre
                c_x, c_y = np.mean(corners_pre, axis=0)
                corners_pre = np.concatenate([corners_pre, corners_pre[[0]]])

                if p == 0:
                    if config.motion_state:
                        if cls_pred_state_pre[corner_id] == 0:
                            color = 'y'
                        else:
                            color = 'b'
                    else:
                        color = 'b'
                    plt.plot(corners_pre[:, 0], corners_pre[:, 1], c=color, linewidth=0.8, zorder=15, alpha = 0.9)
                    plt.scatter(c_x, c_y, s=3, c=color, zorder=15, alpha = 0.9)
                    # plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                    plt.plot([c_x, (corners_pre[-2][0] + corners_pre[0][0]) / 2.], [c_y, (corners_pre[-2][1] + corners_pre[0][1]) / 2.],
                             linewidth=0.8, c=color, zorder=15, alpha = 0.9)
                else:
                    color = 'b'
                    if config.motion_state:
                        if cls_pred_state_pre[corner_id] == 0:
                            continue
                    plt.scatter(c_x, c_y, s=3, c=color, zorder=15)

        for k in range(len(gt_max_iou_idx)):

            anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]

            encode_box = reg_targets[tuple(gt_max_iou_idx[k][:-1]) + (p,)]
            if config.code_type[0] == 'f':
                if config.pred_type == 'motion':

                    # motion a
                    '''
					if p ==0:
						decode_box = bev_box_decode_np(encode_box,anchor)
						cur_det.append(decode_box)
					else:
						decode_box = cur_det[k].copy()
						decode_box[:2] += encode_box[:2]
					'''

                    # motion b
                    if p == 0:
                        decode_box = bev_box_decode_np(encode_box, anchor)
                        cur_det.append(decode_box)
                    else:
                        decode_box = cur_det[k].copy()
                        decode_box[:2] += encode_box[:2]
                        cur_det[k] = decode_box.copy()

                else:
                    decode_box = bev_box_decode_np(encode_box, anchor)
                # print(decode_box)
                decode_corner = center_to_corner_box2d(np.asarray([decode_box[:2]]), np.asarray([decode_box[2:4]]),
                                                       np.asarray([decode_box[4:]]))[0]
            # print(decode_corner)
            # exit()
            # decode_corner = center_to_corner_box2d(np.asarray([anchor[:2]]),np.asarray([anchor[2:4]]),np.asarray([anchor[4:]]))[0]
            elif config.code_type[0] == 'c':
                decoded_corner = (encode_box + anchor).reshape(-1, 4, 2)

            corners = coor_to_vis(decode_corner, area_extents=area_extents, voxel_size=voxel_size)
            c_x, c_y = np.mean(corners, axis=0)
            corners = np.concatenate([corners, corners[[0]]])

            if p == 0:

                plt.plot(corners[:, 0], corners[:, 1], c='g', linewidth=0.8, zorder=5)
                plt.scatter(c_x, c_y, s=5, c='g', zorder=5)
                # plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                plt.plot([c_x, (corners[-2][0] + corners[0][0]) / 2.], [c_y, (corners[-2][1] + corners[0][1]) / 2.],
                         linewidth=0.8, c='g', zorder=5)
            else:
                plt.scatter(c_x, c_y, s=5, linewidth=0.8, c='g', zorder=5)

    m = np.stack([maps, maps, maps], axis=-1)
    m[m == 0] = 0.99
    m[m == 1] = 0.5
    maps = (m * 255).astype(np.uint8)
    # maps = maps.reshape(256,256,3)
    maps = maps.reshape(256,256,3)
    plt.text(-50, -15, "Baseline local results@iou0.5: " + str(int(map_result_0_5 * 100)) + "%", c = 'r')
    plt.text(-50, -5, "Baseline local results@iou0.7: " + str(int(map_result_0_7 * 100)) + "%", c ='r')
    plt.text(150, -15, "MotionNet local results@iou0.5: " + str(int(map_result_0_5_pre * 100)) + "%", c = 'b')
    plt.text(150, -5, "MotionNet local results@iou0.7: " + str(int(map_result_0_7_pre * 100)) + "%", c = 'b')
    plt.imshow(maps, zorder=0)
    if not savename is None:
        plt.savefig(savename)
    else:
        plt.show()
        plt.pause(1)

def main(args, config):
    datapath = args.data_path
    data_pre_path = args.data_pre_path
    save_path = args.save_path
    model_name = datapath.split('/')[-3]
    model_time = datapath.split('/')[-2]
    model_epoch = datapath.split('/')[-1]
    model_pre_name = data_pre_path.split('/')[-3]
    model_pre_time = data_pre_path.split('/')[-2]
    model_pre_epoch = data_pre_path.split('/')[-1]
    save_path = save_path + model_name +'_' + model_pre_name + '/'
    check_folder(save_path)
    save_path = save_path + model_time + '_' + model_pre_time + '/'
    check_folder(save_path)
    save_path = save_path + model_epoch + '_' + model_pre_epoch + '/'
    check_folder(save_path)
    for data_folder in os.listdir(datapath):
        # if 1:
        if data_folder == '86':
            data_folder_path = os.path.join(datapath, data_folder)
            data_pre_folder_path = os.path.join(data_pre_path, data_folder)
            png_save_folder = save_path + data_folder + '/'
            check_folder(png_save_folder)
            for file in tqdm(os.listdir(data_folder_path)):
                if file[-4:] == '.npy' and file[-6:-4] == '34':
                    npy_path = os.path.join(data_folder_path, file)
                    npy_pre_path = os.path.join(data_pre_folder_path, file)
                    data = np.load(npy_path, allow_pickle=True).item()
                    data_pre = np.load(npy_pre_path, allow_pickle=True).item()
                    png_save_path = png_save_folder + file[:-4] + '.png'
                    visualization(config, data, data_pre, png_save_path)
                else:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', default='/GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/baseline/raw/epoch_100_10', type=str, help='')
    parser.add_argument('--data_pre_path', default='/GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/motionNet_l5_f3/11301932/epoch_395_10', type=str, help='')
    parser.add_argument('-s', '--save_path', default='/GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_multiGPU_val/pairwise_fusion_kd/visualization/', type=str, help='')
    parser.add_argument('--binary', default=True, type=bool, help='Only detect car')
    parser.add_argument('--only_det', default=True, type=bool, help='Only do detection')


    args = parser.parse_args()
    print(args)
    config = Config('train', binary=args.binary, only_det=args.only_det)
    main(args, config)