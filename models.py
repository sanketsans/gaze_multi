import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
sys.path.append('../')
from resnetpytorch.models import resnet
from variables import RootVariables
from FlowNetPytorch.models import FlowNetS
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
    def __init__(self, resnet_depth):
        super(VISION_PIPELINE, self).__init__()
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        self.net = FlowNetS.FlowNetS(batch_norm)

        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.net = nn.Sequential(*list(self.net.children())[0:9]).to("cuda:0")
        for i in range(len(self.net) - 1):
             self.net[i][1] = nn.ReLU()

        self.fc1 = nn.Linear(1024*6*8, 4096).to("cuda:0")
        self.fc2 = nn.Linear(4096,256).to("cuda:0")
        self.fc3 = nn.Linear(256, 2).to("cuda:0")

        self.fc1 = nn.Linear(1039, 256).to(self.device)
        self.fc2 = nn.Linear(256,2).to(self.device)
        self.dropout = nn.Dropout(0.35)
        self.activation = nn.Sigmoid()
        self.tensorboard_folder = ''

        for params in self.net.parameters():
            params.requires_grad = True

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def forward(self, input_img):
        out = self.net(input_img).to(self.device)
        out = F.relu(self.dropout(self.fc1(out))).to(self.device)
        out = F.relu(self.dropout(self.fc2(out))).to(self.device)

        # for index, val in enumerate(out):
        #     if out[index][0] > 512.0:
        #         out[index][0] = 512.0
        #     if out[index][1] > 384.0:
        #         out[index][1] = 384.0

        return out

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
