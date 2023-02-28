from torch import nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2 
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output

class CNN_BLSTM(nn.Module):
    def __init__(self):
        super(CNN_BLSTM, self).__init__()
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 3), 1), nn.ReLU())
        # re_shape = layers.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
        self.blstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.droupout = nn.Dropout(0.3)
        # FC
        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense1 = nn.Sequential(
            TimeDistributed(nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.ReLU()), batch_first=True),
            nn.Dropout(0.3))

        # frame score
        self.frame_layer = TimeDistributed(nn.Linear(128, 1), batch_first=True)
        # avg score
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, forward_input):
        forward_input = forward_input.permute(0, 1, 3, 2)
        conv1_output = self.conv1(forward_input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
#         conv4_output = conv4_output.permute(0, 2, 1, 3)
        conv4_output = torch.reshape(conv4_output, (conv4_output.shape[0], conv4_output.shape[2], 4 * 128))

        # blstm
        blstm_output, (h_n, c_n) = self.blstm1(conv4_output)
        blstm_output = self.droupout(blstm_output)

        flatten_output = self.flatten(blstm_output)
        fc_output = self.dense1(flatten_output)
        frame_score = self.frame_layer(fc_output)
        avg_score = self.average_layer(frame_score.permute(0, 2, 1))

        return frame_score#, torch.reshape(avg_score, (avg_score.shape[0], -1))

    
    
class self_att(nn.Module):
    def __init__(self, input_dim, return_att=False):
        super(self_att, self).__init__()

        self.q_weight = nn.Parameter(torch.randn(input_dim, input_dim))
        self.sigmoid = nn.Sigmoid()
        self.return_att = return_att
        
    def forward(self, input):
        #  q_{t'} = x_{t'} W_a
        query =  input @ self.q_weight
        #  e_{t, t'} = q_{t'} x_t^T 
        e = torch.bmm(query, input.permute(0, 2, 1))
        #  a_{t} = \text{Sigmoid}(e_t)
        att = self.sigmoid(e)
        #  v_{t} = \sum{a_{t} x_{t'}}
        value = torch.bmm(att, input)
        if self.return_att is True:
            return [value, att]
        else:
            return value
    
class Stoi_net(nn.Module):
    def __init__(self):
        super(Stoi_net, self).__init__()
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 3), 1), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 3), 1), nn.ReLU())
        # re_shape = layers.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
        self.blstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.droupout = nn.Dropout(0.3)
        # FC
        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense1 = nn.Sequential(
            TimeDistributed(nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.ReLU()), batch_first=True),
            nn.Dropout(0.3))

        # Attention Layer
        self.attention = self_att(128)

        # frame score
        self.frame_layer = TimeDistributed(nn.Linear(128, 1), batch_first=True)
        # avg score
        self.average_layer = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, forward_input):
        forward_input = forward_input.permute(0, 1, 3, 2)
        conv1_output = self.conv1(forward_input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
#         print(conv4_output.size())

        conv4_output = conv4_output.permute(0, 2, 1, 3)
        conv4_output = torch.reshape(conv4_output, (conv4_output.shape[0], conv4_output.shape[1], 4 * 128))

        # blstm
        blstm_output, (h_n, c_n) = self.blstm1(conv4_output)
        blstm_output = self.droupout(blstm_output)

        flatten_output = self.flatten(blstm_output)
        fc_output = self.dense1(flatten_output)

        attention_output = self.attention(fc_output)

        frame_score = self.frame_layer(attention_output)
        avg_score = self.average_layer(frame_score.permute(0, 2, 1))
        return avg_score, frame_score#, torch.reshape(avg_score, (avg_score.shape[0], -1))
    

    
    
    
class CNN_layer(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None, kernel_size=3):
        super(CNN_layer, self).__init__()    
        #CNN
        if not mid_channel:
            mid_channel = out_channel
        pad = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=(3, 3),  padding=pad)
        self.conv2 = nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=(3, 3),  padding=pad)
        self.conv3 = nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=(3, 3), padding=pad)
        self.leacky_relu = nn.LeakyReLU()
    
    def forward(self, x):
        output = self.leacky_relu(self.conv1(x))
        output = self.leacky_relu(self.conv2(output))
        output = self.leacky_relu(self.conv3(output))

        return output

class Down_Sample(nn.Module):
    def __init__(self, channel):
        super(Down_Sample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel, channel, 3, (2, 1), 1, padding_mode='reflect',bias=False),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.layer(x)

class neck(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(neck, self).__init__()
        self.blstm = nn.LSTM(input_dim, target_dim, bidirectional=True, batch_first=True)

        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense =  TimeDistributed(nn.Linear(in_features=target_dim*2, out_features=target_dim), batch_first=True)
        self.frame_layer = TimeDistributed(nn.Linear(target_dim, 1), batch_first=True)

        self.score_layer = nn.Linear(501, 1)

    def forward(self, x):
        blstm_output, (h_n, c_n) = self.blstm(x)

        fc_output = self.flatten(blstm_output)
        fc_output = self.dense(fc_output) 
        fc_output =  self.frame_layer(fc_output)    # frame_score

        final_output = fc_output.view(fc_output.shape[0], -1)
        final_output = self.score_layer(final_output)

        return final_output



class Ushape_Backbone(nn.Module):
    def __init__(self, input_channel, out_channel, num_layer):
        super(Ushape_Backbone, self).__init__()

        self.backbone_list = nn.ModuleList()
        for i in range(num_layer):
            conv_layer = CNN_layer(input_channel, out_channel)
            down_layer = Down_Sample(out_channel)
            self.backbone_list.append(conv_layer)
            self.backbone_list.append(down_layer)
            input_channel = out_channel
            out_channel = out_channel * 2

    def forward(self,x):
        for list_layer in self.backbone_list:
            x = list_layer(x)
            # print(x.shape)
        return x

class Ushape_Att_Backbone(nn.Module):
    def __init__(self, input_channels, out_channels, Ushape_layers, att_dims, att_heads=2, att_layers=2):
        super(Ushape_Att_Backbone, self).__init__()

        self.Ushape_net = Ushape_Backbone(input_channels, out_channels, Ushape_layers)

        self.atten_layer = nn.TransformerEncoderLayer(d_model=att_dims, nhead=att_heads, batch_first=True)
        self.attention = nn.TransformerEncoder(self.atten_layer, num_layers=att_layers)

        self.neck_layer = neck(att_dims, 128)

    def forward(self, forward_input):
        Ushape_output = self.Ushape_net(forward_input)
        feature_tranf = torch.flatten(Ushape_output, start_dim=1, end_dim=2)
        feature_tranf = feature_tranf.permute(0, 2, 1)
        # print(feature_tranf.shape)

        # att_output = self.attention(feature_tranf)
        # print(att_output.shape) 

        neck_output = self.neck_layer(feature_tranf)
        # print(neck_output.shape)

        return neck_output


class Dense_Ushape_Backbone(nn.Module):
    def __init__(self, input_channel, out_channel, num_layer):
        super(Dense_Ushape_Backbone, self).__init__()

        self.conv_list = nn.ModuleList()
        self.down_list = nn.ModuleList()
        for i in range(num_layer):
            conv_layer = CNN_layer(input_channel, out_channel)
            input_channel = out_channel + input_channel
            out_channel = input_channel * 2
            down_layer = Down_Sample(input_channel)
            self.conv_list.append(conv_layer)
            self.down_list.append(down_layer)
            

    def forward(self,down_samp):
        for conv_layer, down_layer in zip(self.conv_list, self.down_list):
            # print(down_samp.shape)
            conv_output = conv_layer(down_samp)
            conv_output = torch.cat((conv_output, down_samp), dim=1)
            down_samp = down_layer(conv_output)
        return down_samp


class Dense_Ushape_CNN_Backbone(nn.Module):
    def __init__(self, input_channels, out_channels, Ushape_layers, att_dims, att_heads=2, att_layers=2):
        super(Dense_Ushape_CNN_Backbone, self).__init__()

        self.Ushape_net = Dense_Ushape_Backbone(input_channels, out_channels, Ushape_layers)

#         self.atten_layer = nn.TransformerEncoderLayer(d_model=att_dims, nhead=att_heads, batch_first=True)
#         self.attention = nn.TransformerEncoder(self.atten_layer, num_layers=att_layers)

        self.neck_layer = neck(att_dims, 128)

    def forward(self, forward_input):
        Ushape_output = self.Ushape_net(forward_input)

        feature_tranf = torch.flatten(Ushape_output, start_dim=1, end_dim=2)
        feature_tranf = feature_tranf.permute(0, 2, 1)
        # print(feature_tranf.shape)

        # att_output = self.attention(feature_tranf)
        # print(att_output.shape) 

        neck_output = self.neck_layer(feature_tranf)
        # print(neck_output.shape)

        return neck_output

# 最后一层为average

class neck_with_ave(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(neck_with_ave, self).__init__()
        self.blstm = nn.LSTM(input_dim, target_dim, bidirectional=True, batch_first=True)

        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense =  TimeDistributed(nn.Linear(in_features=target_dim*2, out_features=target_dim), batch_first=True)
        self.frame_layer = TimeDistributed(nn.Linear(target_dim, 1), batch_first=True)

#         self.score_layer = nn.Linear(501, 1)
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        blstm_output, (h_n, c_n) = self.blstm(x)

        fc_output = self.flatten(blstm_output)
        fc_output = self.dense(fc_output) 
        frame_score = self.frame_layer(fc_output)    # frame_score

#         final_output = fc_output.view(fc_output.shape[0], -1)
#         final_output = self.score_layer(final_output)
        avg_score = self.average_layer(frame_score.permute(0, 2, 1))

#         return final_output
        return torch.reshape(avg_score, (avg_score.shape[0], -1)), frame_score

# dense with average

class Dense_Ushape_CNN_Backbone_With_Ave(nn.Module):
    def __init__(self, input_channels, out_channels, Ushape_layers, att_dims, att_heads=2, att_layers=2):
        super(Dense_Ushape_CNN_Backbone_With_Ave, self).__init__()

        self.Ushape_net = Dense_Ushape_Backbone(input_channels, out_channels, Ushape_layers)

#         self.atten_layer = nn.TransformerEncoderLayer(d_model=att_dims, nhead=att_heads, batch_first=True)
#         self.attention = nn.TransformerEncoder(self.atten_layer, num_layers=att_layers)

        self.neck_layer = neck_with_ave(att_dims, 128)

    def forward(self, forward_input):
        Ushape_output = self.Ushape_net(forward_input)

        feature_tranf = torch.flatten(Ushape_output, start_dim=1, end_dim=2)
        feature_tranf = feature_tranf.permute(0, 2, 1)
        # print(feature_tranf.shape)

        # att_output = self.attention(feature_tranf)
        # print(att_output.shape) 

        avg_score, frame_score = self.neck_layer(feature_tranf)
        # print(neck_output.shape)

        return avg_score, frame_score    

# cnn with average
    
class Ushape_Att_Backbone_With_Ave(nn.Module):
    def __init__(self, input_channels, out_channels, Ushape_layers, att_dims, att_heads=2, att_layers=2):
        super(Ushape_Att_Backbone_With_Ave, self).__init__()

        self.Ushape_net = Ushape_Backbone(input_channels, out_channels, Ushape_layers)
        self.neck_layer = neck_with_ave(att_dims, 128)

    def forward(self, forward_input):
        Ushape_output = self.Ushape_net(forward_input)
        feature_tranf = torch.flatten(Ushape_output, start_dim=1, end_dim=2)
        feature_tranf = feature_tranf.permute(0, 2, 1)
        # print(feature_tranf.shape)

        # att_output = self.attention(feature_tranf)
        # print(att_output.shape) 

        avg_score, frame_score = self.neck_layer(feature_tranf)
        # print(neck_output.shape)

        return avg_score, frame_score  

# blstm with average

class Blstm_With_Ave(nn.Module):
    def __init__(self, input_channels, out_channels, Ushape_layers, att_dims, att_heads=2, att_layers=2):
        super(Blstm_With_Ave, self).__init__()

#         self.Ushape_net = Dense_Ushape_Backbone(input_channels, out_channels, Ushape_layers)

#         self.atten_layer = nn.TransformerEncoderLayer(d_model=att_dims, nhead=att_heads, batch_first=True)
#         self.attention = nn.TransformerEncoder(self.atten_layer, num_layers=att_layers)

        self.neck_layer = neck_with_ave(att_dims, 128)

    def forward(self, forward_input):
#         Ushape_output = self.Ushape_net(forward_input)

        feature_tranf = torch.flatten(forward_input, start_dim=1, end_dim=2)
        feature_tranf = feature_tranf.permute(0, 2, 1)
        # print(feature_tranf.shape)

        # att_output = self.attention(feature_tranf)
        # print(att_output.shape) 

        avg_score, frame_score = self.neck_layer(feature_tranf)
        # print(neck_output.shape)

        return avg_score, frame_score  

    
# 注意力
    
class neck_with_Mulhead(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(neck_with_Mulhead, self).__init__()
        self.blstm = nn.LSTM(input_dim, target_dim, bidirectional=True, batch_first=True)

        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense =  TimeDistributed(nn.Linear(in_features=target_dim*2, out_features=target_dim), batch_first=True)
        self.frame_layer = TimeDistributed(nn.Linear(target_dim, 1), batch_first=True)

        
        self.atten = nn.MultiheadAttention(501, 1)
        self.score_layer = nn.Linear(501, 1)

    def forward(self, x):
        blstm_output, (h_n, c_n) = self.blstm(x)

        fc_output = self.flatten(blstm_output)
        fc_output = self.dense(fc_output)

        fc_output =  self.frame_layer(fc_output)

        att_output = fc_output.view(fc_output.shape[0], -1)
        att_output = self.atten(att_output, att_output, att_output)
#         print(att_output)
        final_output = self.score_layer(att_output[0])


        return final_output

class Dense_Ushape_CNN_Backbone_With_MulHead(nn.Module):
    def __init__(self, input_channels, out_channels, Ushape_layers, att_dims, att_heads=2, att_layers=2):
        super(Dense_Ushape_CNN_Backbone_With_MulHead, self).__init__()

        self.Ushape_net = Dense_Ushape_Backbone(input_channels, out_channels, Ushape_layers)

        self.neck_layer = neck_with_Mulhead(att_dims, 128)

        

    def forward(self, forward_input):
        Ushape_output = self.Ushape_net(forward_input)

        feature_tranf = torch.flatten(Ushape_output, start_dim=1, end_dim=2)
        feature_tranf = feature_tranf.permute(0, 2, 1)
        # print(feature_tranf.shape)

        # att_output = self.attention(feature_tranf)
        # print(att_output.shape) 

        neck_output = self.neck_layer(feature_tranf)
        # print(neck_output.shape)

        return neck_output


class neck_with_Mulhead_Without_FC(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(neck_with_Mulhead_Without_FC, self).__init__()
        self.blstm = nn.LSTM(input_dim, target_dim, bidirectional=True, batch_first=True)

        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense = TimeDistributed(nn.Linear(in_features=target_dim * 2, out_features=target_dim), batch_first=True)
        self.frame_layer = TimeDistributed(nn.Linear(target_dim, 1), batch_first=True)

        self.atten = nn.MultiheadAttention(501, 1)

    def forward(self, x):
        blstm_output, (h_n, c_n) = self.blstm(x)

        fc_output = self.flatten(blstm_output)
        fc_output = self.dense(fc_output)

        fc_output = self.frame_layer(fc_output)

        att_output = fc_output.view(fc_output.shape[0], -1)
        att_output = self.atten(att_output, att_output, att_output)

        return att_output[0][0]


class Dense_Ushape_CNN_Backbone_With_MulHead_Without_FC(nn.Module):
    def __init__(self, input_channels, out_channels, Ushape_layers, att_dims, att_heads=2, att_layers=2):
        super(Dense_Ushape_CNN_Backbone_With_MulHead_Without_FC, self).__init__()

        self.Ushape_net = Dense_Ushape_Backbone(input_channels, out_channels, Ushape_layers)

        self.neck_layer = neck_with_Mulhead(att_dims, 128)

    def forward(self, forward_input):
        Ushape_output = self.Ushape_net(forward_input)

        feature_tranf = torch.flatten(Ushape_output, start_dim=1, end_dim=2)
        feature_tranf = feature_tranf.permute(0, 2, 1)
        # print(feature_tranf.shape)

        # att_output = self.attention(feature_tranf)
        # print(att_output.shape)

        neck_output = self.neck_layer(feature_tranf)
        # print(neck_output.shape)

        return neck_output

class neck_with_MulAtt(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(neck_with_MulAtt, self).__init__()
        self.blstm = nn.LSTM(input_dim, target_dim, bidirectional=True, batch_first=True)

        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense = TimeDistributed(nn.Linear(in_features=target_dim * 2, out_features=target_dim), batch_first=True)
        self.frame_layer = TimeDistributed(nn.Linear(target_dim, 1), batch_first=True)
        self.atten = self_att(501)
        self.score_layer = nn.Linear(501, 1)

    def forward(self, x):
        blstm_output, (h_n, c_n) = self.blstm(x)

        fc_output = self.flatten(blstm_output)
        fc_output = self.dense(fc_output)

        fc_output = self.frame_layer(fc_output)

        att_output = fc_output.view(fc_output.shape[0], -1)
        att_output = self.atten(att_output, att_output, att_output)
        final_output = self.score_layer(att_output)

        return final_output


class Dense_Ushape_CNN_Backbone_With_MulAtt(nn.Module):
    def __init__(self, input_channels, out_channels, Ushape_layers, att_dims, att_heads=2, att_layers=2):
        super(Dense_Ushape_CNN_Backbone_With_MulAtt, self).__init__()

        self.Ushape_net = Dense_Ushape_Backbone(input_channels, out_channels, Ushape_layers)

        self.neck_layer = neck_with_Mulhead(att_dims, 128)

    def forward(self, forward_input):
        Ushape_output = self.Ushape_net(forward_input)

        feature_tranf = torch.flatten(Ushape_output, start_dim=1, end_dim=2)
        feature_tranf = feature_tranf.permute(0, 2, 1)
        # print(feature_tranf.shape)

        # att_output = self.attention(feature_tranf)
        # print(att_output.shape)

        neck_output = self.neck_layer(feature_tranf)
        # print(neck_output.shape)

        return neck_output

if __name__ == "__main__":
    input_data = torch.randn(3, 2, 257,501)
    model = Dense_Ushape_CNN_Backbone_With_MulHead(2, 4, 2)
    output_data = model(input_data)
    print(output_data.shape)