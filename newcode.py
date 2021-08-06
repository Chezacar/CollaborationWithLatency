def get_data_dict(file_name, latency_lamda):
    file_dict = {}
    path_item = file_name.split('/')
    scene_stamp, time_stamp = path_item[-1].split('_')
    agent_id = int(path_item[-2][-1])
    del path_item[-2:]
    path_pre = '/'
    for item in path_item:
        path_pre = os.path.join(path_pre, item)
    latency_list = []
    for agent in range(len(latency_lamda)):
        if agent == agent_id:
            file_dict[agent_id] = file_name
        else:
            # latency_list.append(np.ceil(np.random.exponential(latency_lamda[agent])))
            latency = np.ceil(np.random.exponential(latency_lamda[agent]))
            delay_time_stamp = int(time_stamp) + latency
            if delay_time_stamp >= 99:
                delay_time_stamp = 99
            scene_time = str(scene_stamp) + '_' + str(delay_time_stamp)
            file_dict[agent] = os.path.join(os.path.join(path_pre, 'agent' + str(agent)), scene_time)
    return file_dict

def delay_seq(seq, mu = 0, variance = 0):
    new_seq_dirs = {}
    bias_gaussan = mu + variance * np.random.rand(1)
    bias = np.round(bias_gaussan)
    for item in seq:
        temp_item = item.split("/")[-1]
        temp_id_list = temp_item.split("_")
        inscene_id = int(temp_id_list[1])
        if inscene_id >= 100:
            inscene_id = 99
        if inscene_id <= 0:
            inscene_id = 0
        temp_id = str(int(temp_id_list[0]) * 100  + inscene_id).zfill(4)
        new_seq_dirs[temp_id] = item
    new_seq_list = []
    for i in range(0, len(seq)):
    # for i in range(len(seq)):
        temp_id = str(i).zfill(4)
        temp2_item = new_seq_dirs[temp_id]
        inscene_id_2 = int(temp2_item.split("/")[-1].split("_")[1])
        if inscene_id_2 + bias <= 99:
            temp_id2 = str(int(int(temp_id) + bias)).zfill(4)
            new_seq_list.append(new_seq_dirs[temp_id2])
        else:
            new_seq_list.append(new_seq_dirs[temp_id])     
    return new_seq_list