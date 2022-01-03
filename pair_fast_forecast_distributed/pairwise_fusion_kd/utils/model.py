# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
from numpy.core import shape_base
from torch._C import device
import torch.nn.functional as F
import torch.nn as nn
import torch
import ipdb
import math 

class Predict_Conv(nn.Module):
    def __init__(self, input_size=256, height_feat_size=64, forecast_num = 3):
        super(Predict_Conv, self).__init__()
        self.conv1 = nn.Conv2d(3*height_feat_size, 4*height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4*height_feat_size, 4*height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(2*height_feat_size, 1*height_feat_size)
        self.bn1 = nn.BatchNorm2d(4*height_feat_size)
        self.bn2 = nn.BatchNorm2d(4*height_feat_size)
        self.bn3 = nn.BatchNorm2d(2*height_feat_size)
        self.bn4 = nn.BatchNorm2d(1*height_feat_size)
    def forward(self, x, m):
        x_m = torch.cat([x,m.squeeze(0)], 0)
        x_m = x_m.unsqueeze(0)
        x_m = F.relu(self.bn1(self.conv1(x_m)))
        x_m = F.relu(self.bn2(self.conv2(x_m)))
        x_m = F.relu(self.bn3(self.conv3(x_m)))
        x_m = F.relu(self.bn4(self.linear1(x_m.permute(0,2,3,1)).permute(0,3,1,2)))
        return x_m


class MotionRNN(nn.Module):
    def __init__(self, channel_size = 256, motion_category_num=2, height_feat_size=64, forecast_num = 3):
        super(MotionRNN, self).__init__()
        self.height_feat_size = height_feat_size
        self.forecast_num = forecast_num
        self.ratio = int(math.sqrt(channel_size / height_feat_size))
        self.channel_size = 256
        self.motionconv = STPN_MotionNet(height_feat_size = height_feat_size,forecast_num = forecast_num)
        self.feature_prediction = Predict_Conv(height_feat_size = height_feat_size)
        self.conv_pre_1 = nn.Conv2d(self.channel_size, self.ratio * self.height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(self.ratio * self.height_feat_size, self.height_feat_size, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(self.ratio * self.height_feat_size)
        self.bn_pre_2 = nn.BatchNorm2d(self.height_feat_size)
        self.conv_after_1 = nn.Conv2d(self.height_feat_size, self.ratio * self.height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv_after_2 = nn.Conv2d(self.ratio * self.height_feat_size, self.channel_size, kernel_size=3, stride=1, padding=1)
        self.bn_after_1 = nn.BatchNorm2d(self.ratio * self.height_feat_size)
        self.bn_after_2 = nn.BatchNorm2d(self.channel_size)
        # self.feature_prediction = Predict_Conv(height_feat_size = height_feat_size, output_size = height_feat_size)
    def forward(self, x ,delta_t):
        device = x.device
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))
        a,b,c = x[0].shape
        input_d = torch.zeros((int(delta_t), 1, self.forecast_num, a, b, c)).to(device)
        h_d = torch.zeros((int(delta_t) + self.forecast_num,a,b,c)).to(device)
        h_d[0:self.forecast_num] = x.clone()
        for i in range(int(delta_t)):
            input_d[i][0] = h_d[i:i+self.forecast_num].clone()
            # a = h_d[i:i+self.forecast_num]
            m = self.motionconv(input_d[i].clone())
            h_d[i + self.forecast_num] = self.feature_prediction(h_d[i + self.forecast_num - 1].clone(), m)

        h = F.relu(self.bn_after_1(self.conv_after_1(h_d[-1].unsqueeze(0))))
        h = F.relu(self.bn_after_2(self.conv_after_2(h)))
        return h



class Motion_Prediction_LSTM(nn.Module):
    def __init__(self, channel_size = 256, spatial_size = 32, compressed_size = 256, motion_category_num=2, delta_t = 5, forecast_num = 3):
        super(Motion_Prediction_LSTM, self).__init__()
        self.ratio = int(math.sqrt(channel_size / compressed_size))
        self.delta_t = delta_t
        self.forecast_num = forecast_num
        self.spatial_size = spatial_size
        self.compressed_size = compressed_size
        self.channel_size = channel_size
        self.conv_pre_1 = nn.Conv2d(self.channel_size, self.ratio * self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(self.ratio * self.compressed_size, self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
        self.bn_pre_2 = nn.BatchNorm2d(self.compressed_size)
        # self.conv_pre_3 = nn.Conv2d(16, self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.conv_after_1 = nn.Conv2d(self.compressed_size, self.ratio * self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.conv_after_2 = nn.Conv2d(self.ratio * self.compressed_size, self.channel_size, kernel_size=3, stride=1, padding=1)
        self.bn_after_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
        self.bn_after_2 = nn.BatchNorm2d(self.channel_size)
        self.lstmcell = MotionLSTM(32, self.compressed_size)
        self.time_weight = ModulatedTime(input_channel = 512)

        
    def forward(self, x, delta_t):
        self.delta_t = delta_t
        # Cell Classification head
        # x_shape = [self.forecast_num _32 _32_256]
        # x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        # x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))
        h = x[-1]
        c = torch.zeros((x[0].shape)).to(x.device)
        
        for i in range(self.forecast_num):
            h,c = self.lstmcell(x[i], (h,c))
        # cell_class_pred = self.cell_classify(stpn_out)
        for t in range(int(self.delta_t)):
            h,c = self.lstmcell(h, (h,c))
        # Motion State Classification head
        # state_class_pred = self.state_classify(stpn_out)
        w = self.time_weight(torch.cat([x[-1].unsqueeze(0), h],1), delta_t)
        w = torch.tanh(0.1 * int(delta_t - 1) * w)
        # print("delta_t:", delta_t)
        # print("w.max:", w.max())
        if delta_t > 0:
            res = w * h + (1-w) * x[-1]
        else: 
            res = x[-1].unsqueeze(0)
        # print('res', (res - x[-1]).sum())
        # res = F.relu(self.bn_after_1(self.conv_after_1(res)))
        # res = F.relu(self.bn_after_2(self.conv_after_2(res)))
        # Motion Displacement prediction
        # disp = self.motion_pred(stpn_out)
        # disp = disp.view(-1, 2, stpn_out.size(-2), stpn_out.size(-1))
        del h,c
        # return disp, cell_class_pred, state_class_pred
        return res

class ModulatedTime(nn.Module):
    def __init__(self, input_channel = 128):
        super(ModulatedTime, self).__init__()
        self.input_channel = input_channel
        self.conv1_channel = int(self.input_channel / 2)
        self.conv2_channel = int(self.conv1_channel / 2)
        # self.ratio = math.sqrt()
        self.conv1 = nn.Conv2d(self.input_channel, self.conv1_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.conv1_channel, self.conv2_channel, kernel_size=3, stride=1, padding=1)
        self.convl1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.convl2 = nn.Conv2d(8,self.conv2_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(2 * self.conv2_channel), 8, kernel_size = 3, stride=1, padding = 1) 
        # self.linear3 = nn.Linear(16,8)
        self.convl4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        # self.bnl1 = nn.BatchNorm2d(8)
        # self.bnl2 = nn.BatchNorm2d(self.conv2_channel)
        # self.bnc1 = nn.BatchNorm2d(self.conv1_channel)
        # self.bnc2 = nn.BatchNorm2d(self.conv2_channel)
        # self.bnc3 = nn.BatchNorm2d(8)
        # self.bn4 = nn.BatchNorm2d(1)
    def forward(self, x, delta_t):
        a,b,c,d = x.size()
        y = torch.ones((1,1,c,d)).to(x.device)
        t_y = F.relu(self.convl1(y))
        t_y = F.relu(self.convl2(t_y))
        t_x = F.relu(self.conv1(x))
        t_x = F.relu(self.conv2(t_x))
        t_xy = torch.cat([t_x, t_y], 1)
        t_xy = F.relu(self.conv3(t_xy))
        # t_y = F.relu(self.bn3(self.linear3(y)))
        t_xy = torch.sigmoid(self.convl4(t_xy))
        del y
        return t_xy

class MotionLSTM(nn.Module):
    def __init__(self, spatial_size, input_channel_size, hidden_size = 0):
        super().__init__()
        self.input_channel_size = input_channel_size  # channel size
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size

        #i_t 
        # self.U_i = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        
        self.U_i = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_i = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_i = nn.Parameter(torch.Tensor(1, self.input_channel_size, self.spatial_size, self.spatial_size))
        
        # #f_t 
        # self.U_f = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_f = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_f = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_f = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_f = nn.Parameter(torch.Tensor(1, self.input_channel_size, self.spatial_size, self.spatial_size))

        # #c_t 
        # self.U_c = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_c = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_c = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_c = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_c = nn.Parameter(torch.Tensor(1, self.input_channel_size, self.spatial_size, self.spatial_size))

        # #o_t 
        # self.U_o = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_o = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_o = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_o = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_o = nn.Parameter(torch.Tensor(1, self.input_channel_size, self.spatial_size, self.spatial_size))

        # self.init_weights()

    # def init_weights(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self,x,init_states=None): 
        """ 
        assumes x.shape represents (batch_size, sequence_size, input_channel_size) 
        """ 
        h, c = init_states 
        i = torch.sigmoid(self.U_i(x) + self.V_i(h) + self.b_i) 
        f = torch.sigmoid(self.U_f(x) + self.V_f(h) + self.b_f) 
        g = torch.tanh(self.U_c(x) + self.V_c(h) + self.b_c) 
        o = torch.sigmoid(self.U_o(x) + self.V_o(x) + self.b_o) 
        c_out = f * c + i * g 
        h_out = o *  torch.tanh(c_out) 

        # hidden_seq.append(h_t.unsqueeze(0)) 

        # #reshape hidden_seq p/ retornar 
        # hidden_seq = torch.cat(hidden_seq, dim=0) 
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous() 
        return (h_out, c_out)

class STPN_MotionLSTM(nn.Module):
    def __init__(self, height_feat_size = 16):
        super(STPN_MotionLSTM, self).__init__()



        # self.conv3d_1 = Conv3D(4, 8, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = Conv3D(8, 8, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(height_feat_size, 2*height_feat_size, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(2*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(2*height_feat_size, 4*height_feat_size, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(4*height_feat_size, 4*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(6*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(2*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(3*height_feat_size , height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(height_feat_size, height_feat_size, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(2*height_feat_size)
        self.bn1_2 = nn.BatchNorm2d(2*height_feat_size)

        self.bn2_1 = nn.BatchNorm2d(4*height_feat_size)
        self.bn2_2 = nn.BatchNorm2d(4*height_feat_size)

        self.bn7_1 = nn.BatchNorm2d(2*height_feat_size)
        self.bn7_2 = nn.BatchNorm2d(2*height_feat_size)

        self.bn8_1 = nn.BatchNorm2d(1*height_feat_size)
        self.bn8_2 = nn.BatchNorm2d(1*height_feat_size)

    def forward(self, x):
        # z, h, w = x.size()
        batch = 1
        # bathc 4 32 32 
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        # x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        # x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        # x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        # x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))


        # -- STC block 4
        # x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        # x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))
        # x_4 = x_4.view(batch, -1, x_4.size(1), x_4.size(2), x_4.size(3))
        # x_4 = x_4.permute(0, 2, 1, 3, 4).contiguous()
        # x_4 = F.adaptive_max_pool3d(x_4, (1, None, None))
        # x_4 = x_4.permute(0, 2, 1, 3, 4).contiguous()
        # x_4 = x_4.view(-1, x_4.size(2), x_4.size(3), x_4.size(4)).contiguous()

        # -------------------------------- Decoder Path --------------------------------
        # x_3 = x_3.view(batch, -1, x_3.size(1), x_3.size(2), x_3.size(3))
        # x_3 = x_3.permute(0, 2, 1, 3, 4).contiguous()
        # x_3 = F.adaptive_max_pool3d(x_3, (1, None, None))
        # x_3 = x_3.permute(0, 2, 1, 3, 4).contiguous()
        # x_3 = x_3.view(-1, x_3.size(2), x_3.size(3), x_3.size(4)).contiguous()
        # x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        # x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        # x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        # x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x



class forecast_lstm(nn.Module):
    def __init__(self):
        super(forecast_lstm, self).__init__()
        self.embedding_dim = 4096
        self.hidden_size = 4096
        # self.proj_size = 32768
        self.lstm_layer = nn.LSTMCell(input_size=self.embedding_dim,hidden_size=self.hidden_size)
        # self.conv3d_1 = nn.Conv3D(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = nn.Conv3D(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(0, 0, 0))
        self.linear_1 = nn.Linear(256, 64)
        self.linear_2 = nn.Linear(64,16)
        self.linear_3 = nn.Linear(16,4)
        self.linear_4 = nn.Linear(4, 16)
        self.linear_5 = nn.Linear(16, 64)
        self.linear_6 = nn.Linear(64,256)
        # self.linear_5 = nn.Linear(4,16)
        # self.linear_6 = nn.Linear(16,4)
        
        
    def forward(self, x, delta_t):
        x = x.permute(0, 2, 3, 1)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.linear_3(x)
        x = F.relu(x)
        # x = self.linear_6(x)
        # x = F.relu(x)
        shape_a, shape_b, shape_c, shape_d = x.shape
        x = x.reshape(shape_a, 1, shape_b * shape_c * shape_d)
        for i in range(shape_a):
            x_temp = x[i]
            if i != 0:
                (h_temp, c_temp) = self.lstm_layer(x_temp, (h_temp, c_temp))
            else:
                (h_temp, c_temp) = self.lstm_layer(x_temp)

        # if delta_t < self.num_layers:
        #     x = x[1][int(delta_t)]
        # else:
        #     x = x[1][self.num_layers - 1]

        for j in range(int(delta_t)):
            x_temp = c_temp
            (h_temp, c_temp) = self.lstm_layer(x_temp, (h_temp, c_temp))
        x = h_temp
        x = x.reshape(32,32,4)
        # x = self.linear_5(x)
        # x = F.relu(x)
        x = self.linear_4(x)
        x = F.relu(x)
        x = self.linear_5(x)
        x = F.relu(x)
        x = self.linear_6(x)
        x = F.relu(x)
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, 0)
        return x

class CellClassification(nn.Module):
    def __init__(self, category_num=5):
        super(CellClassification, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class StateEstimation(nn.Module):
    def __init__(self, motion_category_num=2):
        super(StateEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, motion_category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class CatTime(nn.Module):
    def __init__(self):
        super(CatTime, self).__init__()
        self.linear1 = nn.Linear(1,32)
        self.linear2 = nn.Linear(32,1024)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x, delta_t):
        count = 0
        a, b, c, d = x.size()
        x_te = torch.ones(a,1,c,d).to(x.device)
        x_t = torch.ones(x.size())
        for i in range(len(delta_t)):
            for j in range(len(delta_t[0])):
                if delta_t[i][j] != 0:
                    t = delta_t[i][j] * torch.ones((1,1)).to(x.device)
                    # print(t[0])
                    t = F.relu(self.linear1(t))
                    t = F.relu(self.linear2(t))
                    t = t.reshape(32,32)
                    x_te[count][0] = t
                    # t = F.relu(self.conv1(t))
                    # t = F.relu(self.conv2(t))
                    # t = F.relu(self.conv3(t))
        x_te = F.relu(self.bn1(self.conv1(x_te)))
        x_te = F.relu(self.bn2(self.conv2(x_te)))
        x_te = F.relu(self.bn3(self.conv3(x_te)))
        x = F.relu(self.bn4(self.conv4(x)))
        x_t = torch.cat((x,x_te),dim = 1)

        return x_t


class MotionPrediction(nn.Module):
    def __init__(self, seq_len = 256, forecast_num = 1):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(512, seq_len, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(512, seq_len)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(seq_len)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0,2,3,1)
        x = self.linear1(x)
        x = x.permute(0,3,1,2)
        x = F.relu(self.bn3(x))
        # x = self.conv2(x)

        return x

class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x

class MapExtractor(nn.Module):
    def __init__(self, map_channel=8):
        super(MapExtractor, self).__init__()
        self.conv_pre_1 = nn.Conv2d(map_channel, 16, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(16)
        self.bn_pre_2 = nn.BatchNorm2d(16)

        self.conv1_1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # self.conv5_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        # self.conv5_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # self.conv6_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv6_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(32 + 16, 16, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.bn3_1 = nn.BatchNorm2d(128)
        self.bn3_2 = nn.BatchNorm2d(128)

        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(256)

        self.bn5_1 = nn.BatchNorm2d(128)
        self.bn5_2 = nn.BatchNorm2d(128)

        self.bn6_1 = nn.BatchNorm2d(64)
        self.bn6_2 = nn.BatchNorm2d(64)

        self.bn7_1 = nn.BatchNorm2d(32)
        self.bn7_2 = nn.BatchNorm2d(32)

        self.bn8_1 = nn.BatchNorm2d(16)
        self.bn8_2 = nn.BatchNorm2d(16)

    def forward(self, x):

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))


        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        # # -- STC block 3
        # x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        # x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # # -- STC block 4
        # x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        # x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))


        # -------------------------------- Decoder Path --------------------------------
        # x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        # x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        # x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        # x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_1, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x

class STPN(nn.Module):
    def __init__(self, height_feat_size=13):
        super(STPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

#        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
#        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))


        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x


class STPN_MotionNet(nn.Module):
    def __init__(self, height_feat_size=256, forecast_num=3):
        super(STPN_MotionNet, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, height_feat_size * 2, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(height_feat_size * 2, height_feat_size * 2, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(height_feat_size * 2)
        self.bn_pre_2 = nn.BatchNorm2d(height_feat_size * 2)

        self.conv3d_1 = Conv3D(height_feat_size * 4, height_feat_size * 4, kernel_size=(forecast_num, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(height_feat_size * 8, height_feat_size * 8, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(height_feat_size * 2, height_feat_size * 4, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(height_feat_size * 4, height_feat_size * 4, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(height_feat_size * 4, height_feat_size * 8, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(height_feat_size * 8, height_feat_size * 8, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(int(height_feat_size / 2), height_feat_size * 1, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(height_feat_size * 1, height_feat_size * 1, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(height_feat_size * 1, height_feat_size * 2, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(height_feat_size * 2, height_feat_size * 2, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(height_feat_size * 3, height_feat_size * 1, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(height_feat_size * 1, height_feat_size * 1, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(int(height_feat_size * 3 / 2), int(height_feat_size / 2), kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(int(height_feat_size / 2), int(height_feat_size / 2), kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(height_feat_size * 12, height_feat_size * 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(height_feat_size * 4, height_feat_size * 4, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(height_feat_size * 6, height_feat_size * 2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(height_feat_size * 2, height_feat_size , kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(height_feat_size * 4)
        self.bn1_2 = nn.BatchNorm2d(height_feat_size * 4)

        self.bn2_1 = nn.BatchNorm2d(height_feat_size * 8)
        self.bn2_2 = nn.BatchNorm2d(height_feat_size * 8)

        self.bn3_1 = nn.BatchNorm2d(height_feat_size * 1)
        self.bn3_2 = nn.BatchNorm2d(height_feat_size * 1)

        self.bn4_1 = nn.BatchNorm2d(height_feat_size * 2)
        self.bn4_2 = nn.BatchNorm2d(height_feat_size * 2)

        self.bn5_1 = nn.BatchNorm2d(height_feat_size * 1)
        self.bn5_2 = nn.BatchNorm2d(height_feat_size * 1)

        self.bn6_1 = nn.BatchNorm2d(int(height_feat_size / 2))
        self.bn6_2 = nn.BatchNorm2d(int(height_feat_size / 2))

        self.bn7_1 = nn.BatchNorm2d(height_feat_size * 4)
        self.bn7_2 = nn.BatchNorm2d(height_feat_size * 4)

        self.bn8_1 = nn.BatchNorm2d(height_feat_size * 2)
        self.bn8_2 = nn.BatchNorm2d(height_feat_size)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        # x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        # x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))


        # -- STC block 4
        # x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        # x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))
        # x_4 = x_4.view(batch, -1, x_4.size(1), x_4.size(2), x_4.size(3))
        # x_4 = x_4.permute(0, 2, 1, 3, 4).contiguous()
        # x_4 = F.adaptive_max_pool3d(x_4, (1, None, None))
        # x_4 = x_4.permute(0, 2, 1, 3, 4).contiguous()
        # x_4 = x_4.view(-1, x_4.size(2), x_4.size(3), x_4.size(4)).contiguous()

        # -------------------------------- Decoder Path --------------------------------
        # x_3 = x_3.view(batch, -1, x_3.size(1), x_3.size(2), x_3.size(3))
        # x_3 = x_3.permute(0, 2, 1, 3, 4).contiguous()
        # x_3 = F.adaptive_max_pool3d(x_3, (1, None, None))
        # x_3 = x_3.permute(0, 2, 1, 3, 4).contiguous()
        # x_3 = x_3.view(-1, x_3.size(2), x_3.size(3), x_3.size(4)).contiguous()
        # x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        # x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        # x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        # x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x


class STPN_KD(nn.Module):
    def __init__(self, height_feat_size=13):
        super(STPN_KD, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

#        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
#        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))


        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x, x_7, x_6, x_5, x_3


class MotionNet(nn.Module):
    def __init__(self, out_seq_len=256, motion_category_num=2, height_feat_size=256, forecast_num = 3):
        super(MotionNet, self).__init__()
        self.out_seq_len = out_seq_len
        self.forecast_num = forecast_num
        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn = STPN_MotionNet(height_feat_size=height_feat_size, forecast_num = forecast_num)
        self.cattime = CatTime()

    def forward(self, bevs, delta_t):
        # bevs = bevs.permute(0, 1, 2, 3, 4)  # (Batch, seq, z, h, w)
        

        # Backbone network
        x = self.stpn(bevs)

        # Cell Classification head
        # cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        # state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        x = self.cattime(x,delta_t)
        disp = self.motion_pred(x)
        # disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        # return disp, cell_class_pred, state_class_pred
        return disp

# For MGDA loss computation
class FeatEncoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(FeatEncoder, self).__init__()
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.stpn(bevs)

        return x


class MotionNetMGDA(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2):
        super(MotionNetMGDA, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

    def forward(self, stpn_out):
        # Cell Classification head
        cell_class_pred = self.cell_classify(stpn_out)

        # Motion State Classification head
        state_class_pred = self.state_classify(stpn_out)

        # Motion Displacement prediction
        disp = self.motion_pred(stpn_out)
        disp = disp.view(-1, 2, stpn_out.size(-2), stpn_out.size(-1))

        return disp, cell_class_pred, state_class_pred

'''''''''''''''''''''
Added by Yiming

'''''''''''''''''''''
class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))
        
        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.range(start=1, end=number_of_logits, device=input.device).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class lidar_encoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(lidar_encoder, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

#        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
#        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        return x, x_1, x_2, x_3, x_4

class lidar_decoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(lidar_decoder, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

#        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
#        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x, x_1, x_2, x_3, x_4, batch):
        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        #x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        #x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        #x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x, x_5, x_6, x_7, x_8

class lidar_decoder_kd(nn.Module):
    def __init__(self, height_feat_size=13):
        super(lidar_decoder_kd, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

#        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
#        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x, x_1, x_2, x_3, x_4, batch):
        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        #x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        #x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        #x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x, x_7, x_6, x_5



class adafusionlayer(nn.Module):
    def __init__(self,input_channel=128):
        super(adafusionlayer, self).__init__()
        self.attn = nn.Conv2d(input_channel, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self,x):

        _,c,h,w = x.size()
        num_agent = x.size()[0]

        fusion_weight = F.relu(self.bn(self.attn(x)))
        # fusion_weight = F.relu(self.attn(x))

        fusion_weight = F.softmax(fusion_weight,dim=0).cuda()


        feat = torch.zeros(x[0].size()).cuda()
        for j in range(num_agent):

            feat = feat + (x[j]*(fusion_weight[j].repeat(c,1,1)))
        
        return feat

class multifusionlayer(nn.Module):
    def __init__(self,input_channel=128):
        super(multifusionlayer,self).__init__()
        c1 = 64
        c2 = 32 
        self.attn1 = nn.Conv2d(input_channel,c1,kernel_size=1, stride=1)
        self.attn2 = nn.Conv2d(c1,c2,kernel_size=1, stride=1)
        self.attn3 = nn.Conv2d(c2,1,kernel_size=1, stride=1)
        
        self.bn_1=nn.BatchNorm2d(c1)
        self.bn_2=nn.BatchNorm2d(c2)
        self.bn_3=nn.BatchNorm2d(1)

    
    def forward(self,x):

        _,c,h,w = x.size()
        num_agent = 5

        x = F.relu(self.bn_1(self.attn1(x)))
        x = F.relu(self.bn_2(self.attn2(x)))
        x = F.relu(self.bn_3(self.attn3(x)))
        fusion_weight = F.softmax(x,dim=0).cuda()

        feat = torch.zeros(x[0].size()).cuda()
        for j in range(num_agent):

            feat = feat + (x[j]*(fusion_weight[j].repeat(c,1,1)))
        
        return feat



class sigmoidfusionlayer(nn.Module):
    def __init__(self,input_channel=128):
        super(sigmoidfusionlayer, self).__init__()
        self.attn = nn.Conv2d(input_channel, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self,x):

        _,c,h,w = x.size()
        num_agent = 5

        fusion_weight = F.sigmoid(F.relu(self.bn(self.attn(x)))).cuda()
        # fusion_weight = fusion_weight.sum(dim=0).cuda()


        feat = torch.zeros(x[0].size()).cuda()
        for j in range(num_agent):

            feat = feat + (x[j]*(fusion_weight[j]))
        
        return feat


# class MLPfusionlayer(nn.Module):
#     def __init__(self,input_channel=128):
#         super(MLPfusionlayer,self).__init__()

#         self.MLP = 


class pairfusionlayer(nn.Module):
    def __init__(self,input_channel=512):
        super(pairfusionlayer, self).__init__()
        self.attn = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self,x):

        _,c,h,w = x.size()
        num_agent = x.size()[0]
        
        cat_list=[]
        for i in range(num_agent):
            
            cat_list.append(torch.cat((x[0],x[i])))
        
        
        feat_list=torch.stack(cat_list)

        # fusion_weight = F.relu(self.bn(self.attn(x)))
        fusion_weight = F.relu(self.attn(feat_list))

        fusion_weight = F.softmax(fusion_weight,dim=0).cuda()

        feat = torch.zeros(x[0].size()).cuda()
        for j in range(num_agent):

            feat = feat + (x[j]*(fusion_weight[j].repeat(c,1,1)))
        
        return feat


class pairfusionlayer_1(nn.Module):
    def __init__(self,input_channel=512):
        super(pairfusionlayer_1, self).__init__()
        self.conv1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self,x):

        _,c,h,w = x.size()
        num_agent = x.size()[0]
        
        cat_list=[]
        for i in range(num_agent):
            
            cat_list.append(torch.cat((x[0],x[i])))
        
        
        feat_list=torch.stack(cat_list)

        # # fusion_weight = F.relu(self.bn(self.attn(x)))
        # fusion_weight = F.relu(self.attn(feat_list))
        x_1 = F.relu(self.bn1_1(self.conv1_1(feat_list)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        fusion_weight = F.relu(self.conv1_4(x_1))


        fusion_weight = F.softmax(fusion_weight,dim=0).cuda()

        feat = torch.zeros(x[0].size()).cuda()
        for j in range(num_agent):

            feat = feat + (x[j]*(fusion_weight[j].repeat(c,1,1)))
        
        return feat


class pairfusionlayer_2(nn.Module):
    def __init__(self,input_channel=512):
        super(pairfusionlayer_2, self).__init__()
        self.conv1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.bn1_4 = nn.BatchNorm2d(1)


    def forward(self,x):
        device = x.device
        _,c,h,w = x.size()
        num_agent = x.size()[0]
        
        cat_list=[]
        for i in range(num_agent):
            
            cat_list.append(torch.cat((x[0],x[i])))
        
        
        feat_list=torch.stack(cat_list)

        # # fusion_weight = F.relu(self.bn(self.attn(x)))
        # fusion_weight = F.relu(self.attn(feat_list))
        x_1 = F.relu(self.bn1_1(self.conv1_1(feat_list)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        fusion_weight = F.relu(self.bn1_4(self.conv1_4(x_1)))


        fusion_weight = F.softmax(fusion_weight,dim=0).cuda(device)

        feat = torch.zeros(x[0].size()).cuda(device)
        for j in range(num_agent):

            feat = feat + (x[j]*(fusion_weight[j].repeat(c,1,1)))
        
        return feat


class pairfusionlayer_3(nn.Module):
    def __init__(self,input_channel=512):
        super(pairfusionlayer_3, self).__init__()
        self.conv1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(8)

        # self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        # self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.bn1_4 = nn.BatchNorm2d(1)


    def forward(self,x):
        device = x.device
        _,c,h,w = x.size()
        num_agent = x.size()[0]
        
        cat_list=[]
        for i in range(num_agent):
            
            cat_list.append(torch.cat((x[0],x[i])))
        
        
        feat_list=torch.stack(cat_list)

        # # fusion_weight = F.relu(self.bn(self.attn(x)))
        # fusion_weight = F.relu(self.attn(feat_list))
        x_1 = F.relu(self.bn1_1(self.conv1_1(feat_list)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        # x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        fusion_weight = F.relu(self.bn1_4(self.conv1_4(x_1)))


        # fusion_weight = F.softmax(fusion_weight,dim=0).cuda(device, non_blocking=True)
        fusion_weight = F.softmax(fusion_weight,dim=0).cuda(device)

        # feat = torch.zeros(x[0].size()).cuda(device, non_blocking=True)
        feat = torch.zeros(x[0].size()).cuda(device)
        for j in range(num_agent):

            feat = feat + (x[j]*(fusion_weight[j].repeat(c,1,1)))
        
        return feat


class pairfusionlayer_4(nn.Module):
    def __init__(self,input_channel=512):
        super(pairfusionlayer_4, self).__init__()
        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(256)

        # self.conv1_2 = nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0)
        # self.bn1_2 = nn.BatchNorm2d(8)

        # self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        # self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.bn1_4 = nn.BatchNorm2d(1)


    def forward(self,x):

        _,c,h,w = x.size()
        num_agent = x.size()[0]
        
        cat_list=[]
        for i in range(num_agent):
            
            cat_list.append(torch.cat((x[0],x[i])))
        
        
        feat_list=torch.stack(cat_list)

        # # fusion_weight = F.relu(self.bn(self.attn(x)))
        # fusion_weight = F.relu(self.attn(feat_list))
        x_1 = F.relu(self.bn1_1(self.conv1_1(feat_list)))
        # x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        # x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        fusion_weight = F.relu(self.bn1_4(self.conv1_4(x_1)))

        # print(fusion_weight.size())


        fusion_weight = F.softmax(fusion_weight,dim=0).cuda()
        # print(fusion_weight.size())
        # ipdb.set_trace()


        feat = torch.zeros(x[0].size()).cuda()
        for j in range(num_agent):

            feat = feat + (x[j]*(fusion_weight[j].repeat(c,1,1)))
        
        return feat




