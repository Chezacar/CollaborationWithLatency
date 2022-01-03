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
                corner_box_pre = cls_pred_corners_pre[corner_id]

                corners_pre = coor_to_vis(corner_box_pre, area_extents=area_extents, voxel_size=voxel_size)

                c_x, c_y = np.mean(corners_pre, axis=0)
                corners = np.concatenate([corners_pre, corners_pre[[0]]])

                if p == 0:
                    if config.motion_state:
                        if cls_pred_state[corner_id] == 0:
                            color = 'y'
                        else:
                            color = 'b'
                    else:
                        color = 'b'
                    plt.plot(corners_pre[:, 0], corners_pre[:, 1], c=color, linewidth=0.8, zorder=15)
                    plt.scatter(c_x, c_y, s=3, c=color, zorder=15)
                    # plt.scatter(corners[0,0], corners[0,1], s=10,c = 'r')
                    plt.plot([c_x, (corners_pre[-2][0] + corners_pre[0][0]) / 2.], [c_y, (corners_pre[-2][1] + corners_pre[0][1]) / 2.],
                             linewidth=0.8, c=color, zorder=15)
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
    maps = maps.reshape(256,256,3)
    plt.text(-50, -15, "Baseline local results@iou0.5: " + str(int(map_result_0_5 * 100)) + "%")
    plt.text(-50, -5, "Baseline local results@iou0.7: " + str(int(map_result_0_7 * 100)) + "%")
    plt.text(150, -15, "MotionNet local results@iou0.5: " + str(int(map_result_0_5_pre * 100)) + "%")
    plt.text(150, -5, "MotionNet local results@iou0.7: " + str(int(map_result_0_7_pre * 100)) + "%")
    plt.imshow(maps, zorder=0)
    if not savename is None:
        plt.savefig(savename)
    else:
        plt.show()
        plt.pause(1)

