import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
sys.path.append('../')
from variables import RootVariables
from flownet2pytorch.networks import FlowNetS
#from skimage.transform import rotate

class All_Models:
    def __init__(self):
        print("Setting up all models .. ")

    def get_model(self, model_index, vision_model_depth, test_folder):
        if model_index == 0:
            return IMU_PIPELINE(), 'signal_checkpoint_' + test_folder[5:] + '.pth'
        elif model_index == 1:
            return VISION_PIPELINE(vision_model_depth), 'vision_checkpoint_' + test_folder[5:] + '.pth'
        elif model_index == 2:
            return FusionPipeline(vision_model_depth, test_folder), 'pipeline_checkpoint_' + test_folder[5:] + '.pth'

class VISION_PIPELINE(nn.Module):
    def __init__(self):
        super(VISION_PIPELINE, self).__init__()
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        self.net = FlowNetS.FlowNetS(input_channels=6, batchNorm=False)
        # dict = torch.load(checkpoint_path)
        # self.net.load_state_dict(dict["state_dict"])
        # self.net = nn.Sequential(*list(self.net.children())[:])
        #
        # self.net[0][0] = nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1)
        # self.encoder1 = self.net[0]
        # self.encoder2 = self.net[1]
        # self.encoder3 = self.net[2]
        # self.encoder4 = self.net[3]
        #
        # self.resnet_block = self.net[4]
        #
        # self.decoder1 = self.net[5]
        # self.decoder1 = nn.Sequential(*list(self.decoder1.children())[:])
        # self.decoder1[0] = nn.Conv2d(1024, 256, kernel_size=3, stride=2, padding=1)
        # del self.decoder1[1:3]
        #
        # self.decoder2 = self.net[6]
        # self.decoder2 = nn.Sequential(*list(self.decoder2.children())[:])
        # self.decoder2[0] = nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1)
        # del self.decoder2[1:3]
        #
        # self.decoder3 = self.net[7]
        # self.decoder3 = nn.Sequential(*list(self.decoder3.children())[:])
        # self.decoder3[0] = nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1)
        # del self.decoder3[1:3]
        #
        # self.decoder4 = self.net[8]
        # self.decoder4 = nn.Sequential(*list(self.decoder4.children())[:])
        # self.decoder4[0] = nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1)
        # del self.decoder4[1:3]

        self.tensorboard_folder = ''

        for params in self.net.parameters():
            params.requires_grad = True

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def forward(self, input_img):
        # encoder
        return self.net(input_img)
        # skip_connections = {}
        # inputs = self.encoder1(input_img)
        # skip_connections['skip0'] = inputs.clone()
        # inputs = self.encoder2(inputs)
        # skip_connections['skip1'] = inputs.clone()
        # inputs = self.encoder3(inputs)
        # skip_connections['skip2'] = inputs.clone()
        # inputs = self.encoder4(inputs)
        # skip_connections['skip3'] = inputs.clone()
        #
        # # transition
        # inputs = self.resnet_block(inputs)
        #
        # # decoder
        # print(inputs.shape, skip_connections['skip3'].shape)
        # flow_dict = {}
        # inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        # inputs = self.decoder1(inputs)
        # # flow_dict['flow0'] = flow.clone()
        #
        # print(inputs.shape, skip_connections['skip2'].shape)
        #
        # inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        # inputs = self.decoder2(inputs)
        # # flow_dict['flow1'] = flow.clone()
        #
        # print(inputs.shape, skip_connections['skip1'].shape)
        #
        # inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        # inputs = self.decoder3(inputs)
        # # flow_dict['flow2'] = flow.clone()
        #
        # print(inputs.shape, skip_connections['skip0'].shape)
        #
        # inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        # inputs = self.decoder4(inputs)
        # flow_dict['flow3'] = flow.clone()

        # return flow_dict

    def get_original_coordinates(self, pred, labels):
        return pred*self.orig_tensor, labels*self.orig_tensor

class IMU_PIPELINE(nn.Module):
    def __init__(self):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.55, bidirectional=True).to(self.device)
        self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.dropout = nn.Dropout(0.45)
        self.activation = nn.Sigmoid()
        self.orig_tensor = torch.tensor([3.75, 2.8125]).to(self.device)

        self.tensorboard_folder = '' #'BLSTM_signal_outputs_sell1/'

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0] - label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()
        # return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def forward(self, x):
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(out[:,-1,:]))
        return out

    def get_original_coordinates(self, pred, labels):
        return pred*self.orig_tensor, labels*self.orig_tensor

class FusionPipeline(nn.Module):
    def __init__(self, resnet_depth, test_folder):
        super(FusionPipeline, self).__init__()
        torch.manual_seed(2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var = RootVariables()
        self.activation = nn.Sigmoid()
        self.temporalSeq = 32
        self.temporalSize = 16
        self.trim_frame_size = 150
        self.imuCheckpoint_file = 'signal_checkpoint0_' + test_folder[5:] + '.pth'
        self.frameCheckpoint_file = 'vision_checkpointAdam9CNN_' + test_folder[5:] +'.pth'
        self.orig_tensor = torch.tensor([3.75, 2.8125]).to(self.device)

        ## IMU Models
        self.imuModel = IMU_ENCODER()
        imuCheckpoint = torch.load(self.var.root + 'datasets/' + test_folder[5:] + '/' + self.imuCheckpoint_file,  map_location="cuda:0")
        self.imuModel.load_state_dict(imuCheckpoint['model_state_dict'])
        for params in self.imuModel.parameters():
             params.requires_grad = True

        ## FRAME MODELS
        self.frameModel =  VIS_ENCODER(resnet_depth)
        frameCheckpoint = torch.load(self.var.root + 'datasets/' + test_folder[5:] + '/' + self.frameCheckpoint_file,  map_location="cuda:0")
        self.frameModel.load_state_dict(frameCheckpoint['model_state_dict'])
        for params in self.frameModel.parameters():
            params.requires_grad = True

        ## TEMPORAL MODELS
 #       self.temporalModel = TEMP_ENCODER(self.temporalSize)

#        self.fc1 = nn.Linear(self.var.hidden_size, 2).to("cuda:2")
        self.dropout = nn.Dropout(0.35)
        self.fc0 = nn.Linear(512, 256).to(self.device)
        self.fc1 = nn.Linear(256, 2).to(self.device)
#        self.fc2 = nn.Linear(128, 2).to("cuda:2")
        ##OTHER
        self.imu_encoder_params = None
        self.frame_encoder_params = None
        self.imuBN = nn.BatchNorm1d(self.var.hidden_size*2, affine=True).to(self.device)
        self.frameBN = nn.BatchNorm1d(self.var.hidden_size*2, affine=True).to(self.device)
        self.fcBN = nn.BatchNorm1d(256).to(self.device)
        self.tensorboard_folder = ''

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        self.imu_encoder_params = F.relu(self.imuBN(self.imuModel(imu_BatchData.float()))).to(self.device)
        self.frame_encoder_params = F.relu(self.frameBN(self.frameModel(frame_BatchData.float()))).to(self.device)
#        self.frame_encoder_params = F.leaky_relu(self.dropout(self.downsample(self.frame_encoder_params)), 0.1).to("cuda:1")
        return self.imu_encoder_params, self.frame_encoder_params

    def fusion_network(self, imu_params, frame_params):
        return torch.cat((frame_params, imu_params), dim=1).to(self.device)

    def temporal_modelling(self, fused_params):
 #       newParams = fused_params.reshape(fused_params.shape[0], self.temporalSeq, self.temporalSize)
 #       tempOut = self.temporalModel(newParams.float()).to("cuda:2")
 #       gaze_pred = self.fc1(tempOut).to("cuda:2")
 #       print(fused_params, self.fc0.weight)
        gaze_pred = F.relu(self.fcBN(self.fc0(self.dropout(fused_params)))).to(self.device)
        gaze_pred = F.relu(self.fc1(self.dropout(gaze_pred))).to(self.device)
#        gaze_pred = self.fc2(self.dropout(gaze_pred)).to("cuda:2")

        return gaze_pred

    def forward(self, batch_frame_data, batch_imu_data):
        imu_params, frame_params = self.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = self.fusion_network(imu_params, frame_params)
        coordinate = self.temporal_modelling(fused)

        for index, val in enumerate(coordinate):
            if coordinate[index][0] > 512.0:
                coordinate[index][0] = 512.0
            if coordinate[index][1] > 384.0:
                coordinate[index][1] = 384.0

        return coordinate

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def get_original_coordinates(self, pred, labels):
        return pred*self.orig_tensor, labels*self.orig_tensor

class IMU_ENCODER(nn.Module):
    def __init__(self):
        super(IMU_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.65, bidirectional=True).to(self.device)
        # self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)

    def forward(self, x):
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)

        # x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        return out[:,-1,:]

class TEMP_ENCODER(nn.Module):
    def __init__(self, input_size):
        super(TEMP_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size, int(self.var.hidden_size/2), int(self.var.num_layers/2), batch_first=True, dropout=0.45, bidirectional=True).to(self.device)

    def forward(self, x):
        # hidden = (h0, c0)
        h0 = torch.randn(self.var.num_layers, self.var.batch_size, int(self.var.hidden_size/2), requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers, self.var.batch_size, int(self.var.hidden_size/2), requires_grad=True).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        # out = self.activation(self.fc1(out[:,-1,:]))
        return out[:,-1,:]

class VIS_ENCODER(nn.Module):
    def __init__(self, resnet_depth):
        super(VIS_ENCODER, self).__init__()

        self.var = RootVariables()
        torch.manual_seed(1)
        self.net = resnet.generate_model(50)

        dict = torch.load(self.var.root + 'r3d' + str(resnet_depth) + '_KM_200ep.pth')
        self.net.load_state_dict(dict["state_dict"])

        for params in self.net.parameters():
            params.requires_grad = True

    def forward(self, input_img):
        out = self.net(input_img)

        return out

if __name__ == '__main__':
    from EVFlowNetpytorch.src.config import configs
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    args = configs()
    model = VISION_PIPELINE()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    var = RootVariables()
    f0 = var.root + 'testing_images/image_1.jpg'
    f1 = var.root + 'testing_images/image_2.jpg'
    frame1 = cv2.imread(f0)
    t0 = transforms.ToTensor()(Image.open(f0))
    t1 = transforms.ToTensor()(Image.open(f1))
    print(model)
    model.eval()
    t = torch.cat((t0, t1), dim=0)
    t = t.unsqueeze(dim=0)
    x = model(t).to(device)
    x = x.squeeze(dim=0)
    c, h, w = x.shape
    y = torch.ones((1, h, w))
    x = torch.cat((x, y), dim=0)
    x = x.permute(1, 2, 0)
    print(x.shape)
    x = x.detach().cpu().numpy()
    mag, ang = cv2.cartToPolar(x[...,0], x[...,1])

    hsv = np.zeros_like(x)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    print(hsv.shape)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    plt.imshow(x)
    plt.show()
