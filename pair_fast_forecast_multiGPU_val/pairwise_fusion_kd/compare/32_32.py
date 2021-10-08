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
            tg_agent = []
            tg_agent.append(local_com_mat[b, i])
            all_warp = trans_matrices[b, i, -1] # transformation [2 5 5 4 4]
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