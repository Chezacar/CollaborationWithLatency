    @classmethod
    def get_instance_boxes_multisweep_sample_data(cls,
                                                  nusc: 'NuScenes',
                                                  ref_sd_rec: Dict,
                                                  instance_token: str,
                                                  nsweeps_back: int = 5,
                                                  nsweeps_forward: int = 5) -> \
            Tuple[List['Box'], np.array, List[str], List[str]]:
        """
        Return the bounding boxes associated with the given instance. The bounding boxes are across different sweeps.
        For each bounding box, we need to map its (global) coordinates to the reference frame.
        For this function, the reference sweep is supposed to be from sample data record (not sample. ie, keyframe).
        :param nusc: A NuScenes instance.
        :param ref_sd_rec: The current sample data record.
        :param instance_token: The current selected instance.
        :param nsweeps_back: Number of sweeps to aggregate. The sweeps trace back.
        :param nsweeps_forward: Number of sweeps to aggregate. The sweeps are obtained from the future.
        :return: (list of bounding boxes, the time stamps of bounding boxes, attribute list, category list)
        """

        # Init
        box_list = list()
        all_times = list()
        attr_list = list()  # attribute list
        cat_list = list()  # category list

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Get the bounding boxes across different sweeps
        boxes = list()

        # Move backward to get the past annotations
        current_sd_rec = ref_sd_rec
        for _ in range(nsweeps_back):
            box, attr, cat = nusc.get_instance_box(current_sd_rec['token'], instance_token)
            boxes.append(box)  # It is possible the returned box is None
            attr_list.append(attr)
            cat_list.append(cat)

            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # positive difference
            all_times.append(time_lag)

            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        # Move forward to get the future annotations
        current_sd_rec = ref_sd_rec

        # Abort if there are no future sweeps.
        if current_sd_rec['next'] != '':
            current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

            for _ in range(nsweeps_forward):
                box, attr, cat = nusc.get_instance_box(current_sd_rec['token'], instance_token)
                boxes.append(box)  # It is possible the returned box is None
                attr_list.append(attr)
                cat_list.append(cat)

                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # negative difference
                all_times.append(time_lag)

                if current_sd_rec['next'] == '':
                    break
                else:
                    current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

        # Map the bounding boxes to the local sensor coordinate
        for box in boxes:
            if box is not None:
                # Move box to ego vehicle coord system
                box.translate(-np.array(ref_pose_rec['translation']))
                box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)

                # Move box to sensor coord system
                box.translate(-np.array(ref_cs_rec['translation']))
                box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)

                # caused by coordinate inconsistency of nuscene-toolkit
                box.center[0] = - box.center[0]

#                # debug
                shift = [box.center[0],box.center[1],box.center[2]]
                box.translate(-np.array(shift))
                box.rotate(Quaternion([0,1,0,0]).inverse)
                box.translate(np.array(shift))

            box_list.append(box)
        #print(temp)
        return box_list, all_times, attr_list, cat_list