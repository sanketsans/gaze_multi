import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
sys.path.append('../')
from variables import RootVariables
from flownet2pytorch.networks import FlowNetS
from submodules import *
#from skimage.transform import rotate

class All_Models:
    def __init__(self):
        print("Setting up all models .. ")

    def get_model(self, model_index, test_folder):
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
        self.input_channels = 6
        self.batchNorm = False
        self.net = FlowNetS.FlowNetS(input_channels=self.input_channels, batchNorm=False)
        dict = torch.load(self.var.root + 'FlowNet2-S_checkpoint.pth.tar', map_location="cpu")
        self.net.load_state_dict(dict["state_dict"])
        self.net = self.net = nn.Sequential(*list(self.net.children()))
        self.conv1 = self.net[0]
        self.conv2 = self.net[1]
        self.conv3 = self.net[2]
        self.conv3_1 = self.net[3]
        self.conv4 = conv(self.batchNorm, 256, 512, stride=1)
        self.conv4_1 = self.net[5]
        self.conv5 = conv(self.batchNorm, 512, 512, stride=1)
        self.conv5_1 = self.net[7]
        self.conv6 = self.net[8]
        self.conv6_1 = self.net[9]

        self.deconv5 = self.net[10]
        self.deconv4 = deconv(1024, 256, kernel_size=3, stride=1)
        self.deconv3 = deconv(768, 128, kernel_size=3, stride=1)
        self.deconv2 = deconv(384, 64, kernel_size=4)
        self.deconv1 = deconv(192, 32, kernel_size=(4, 4), stride=2)
        self.deconv0 = deconv(96, self.input_channels, kernel_size=4)

        self.predict_flow6 = self.net[14]
        self.predict_flow5 = predict_flow(1024)
        self.predict_flow4 = predict_flow(768)
        self.predict_flow3 = predict_flow(384)
        self.predict_flow2 = predict_flow(192)
        self.predict_flow1 = predict_flow(96)
        self.predict_flow0 = predict_flow(6, channels=3)

        self.upsampled_flow6_to_5 = self.net[19]
        self.upsampled_flow5_to_4 = self.net[20]
        self.upsampled_flow4_to_3 = self.net[21]
        self.upsampled_flow3_to_2 = self.net[22]
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.tensorboard_folder = ''
        self.activation = nn.Softmax2d()

        for params in self.net.parameters():
            params.requires_grad = True

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def forward(self, input_img):
        # encoder & decoder
        out_conv1 = self.conv1(input_img)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # flow6       = self.predict_flow6(out_conv6)
        # flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5),1)
        # flow5       = self.predict_flow5(concat5)
        # flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4,out_deconv4),1)
        # flow4       = self.predict_flow4(concat4)
        # flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3,out_deconv3),1)
        # flow3       = self.predict_flow3(concat3)
        # flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2),1)
        # flow2       = self.predict_flow2(concat2)
        # flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        # flow1       = self.predict_flow1(concat1)
        # flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        flow = self.predict_flow0(out_deconv0)
        flow0 = self.activation(flow)

        return flow0

        # if self.training:
        #     return flow0, flow1, flow2, flow3, flow4, flow5, flow6
        # else:
        #     return flow0

    def get_original_coordinates(self, pred, labels):
        return pred*self.orig_tensor, labels*self.orig_tensor

class IMU_PIPELINE(nn.Module):
    def __init__(self):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.55, bidirectional=True).to(self.device)
        # self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.deconv = deconv(1, 3, kernel_size=4, stride=2, padding=1)
        self.deconv1 = deconv(3, 3, kernel_size=4, stride=2, padding=1)
        self.deconv2 = deconv(3, 3, kernel_size=4, stride=2, padding=1)
        self.deconv3 = deconv(3, 3, kernel_size=(3, 4), stride=2, padding=1)
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

        # x = self.fc0(x)
        print(x.shape)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:,-1,:].reshape(-1, 1, 16, 32)
        out = self.deconv(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        print(out.shape)
        # out = F.relu(self.fc1(out[:,-1,:]))
        # return out

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
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    from helpers import Helpers
    import cv2
    import numpy as np
    from create_dataset import All_Dataset

    # test_folder = 'test_CoffeeVendingMachine_S1​'
    # utils = Helpers(test_folder)
    # imu_training, imu_testing, training_target, testing_target = utils.load_datasets(1, repeat=0)
    #
    # All_Dataset = All_Dataset()
    # trainDataset = All_Dataset.get_dataset('trainImg', imu_testing, 'heatmap_testImg', 0)
    # a, _ = trainDataset[0]
    # a = a.unsqueeze(dim=0)
    # print(a.shape)
    # model = IMU_PIPELINE()
    # _ = model(a.float())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    var = RootVariables()
    criterion = nn.KLDivLoss(reduction='batchmean')
    f0 = var.root + 'testing_images/test_sanket_Interaction_S3/image_18345.jpg'
    f1 = var.root + 'testing_images/test_sanket_Interaction_S3/image_18346.jpg'
    gt = var.root + 'heatmap_testing_images/test_sanket_Interaction_S3/image_0.jpg'

    # frame1 = cv2.imread(f0)
    # frame1 = cv2.resize(frame1, (32, 16))
    # plt.imshow(frame1)
    # plt.show()
    t0 = transforms.ToTensor()(Image.open(f0))
    t1 = transforms.ToTensor()(Image.open(f1))

    gt = transforms.ToTensor()(Image.open(gt))
    gt = gt.unsqueeze(dim=0)
    act = nn.Softmax2d()
    gt = act(gt)
    gt = gt.squeeze(dim=0)
    print(torch.argmax(gt, dim=1), torch.argmax(gt, dim=0).shape, torch.max(torch.argmax(gt, dim=1)), torch.max(torch.argmax(gt, dim=2)))
    gt = gt.permute(1, 2, 0)
    gt = gt.detach().cpu().numpy()
    # plt.imshow(gt)
    # plt.show()

    # model = VISION_PIPELINE()
    # model.eval()
    # t = torch.cat((t0, t1), dim=0)
    # t = t.unsqueeze(dim=0)
    # x = model(t).to(device)
    # print(x)
    # loss = criterion(x, gt)
    # print(loss)
    # print(loss.shape)
    # print(x.shape)
    # x = x.squeeze(dim=0)
    # x = x.permute(1, 2, 0)
    # x = x.detach().cpu().numpy()
    # # x = cv2.GaussianBlur(x, (0, 0), 10)
    # # x /= np.max(x)  # keep the max to 1
    # plt.imshow(x)
    # plt.show()
    # print(x.shape)
    # mag, ang = cv2.cartToPolar(x[...,0], x[...,1])
    #
    # hsv = np.zeros_like(x)
    # hsv[...,1] = 255
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # plt.imshow(x)
    # plt.show()
