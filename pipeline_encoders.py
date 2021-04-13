import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys, os, ast
sys.path.append('../')
from variables import RootVariables
from FlowNetPytorch.models import FlowNetS

device = torch.device("cpu")

class IMU_ENCODER(nn.Module):
    def __init__(self):
        super(IMU_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.65, bidirectional=True).to("cuda:0")
        # self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)

    def forward(self, x):
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        # x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        return out[:,-1,:]

class TEMP_ENCODER(nn.Module):
    def __init__(self, input_size):
        super(TEMP_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size, int(self.var.hidden_size/2), int(self.var.num_layers/2), batch_first=True, dropout=0.45, bidirectional=True).to("cuda:0")

    def forward(self, x):
        # hidden = (h0, c0)
        h0 = torch.randn(self.var.num_layers, self.var.batch_size, int(self.var.hidden_size/2), requires_grad=True).to("cuda:0")
        c0 = torch.randn(self.var.num_layers, self.var.batch_size, int(self.var.hidden_size/2), requires_grad=True).to("cuda:0")
        out, _ = self.lstm(x, (h0, c0))
        # out = self.activation(self.fc1(out[:,-1,:]))
        return out[:,-1,:]

class VIS_ENCODER(nn.Module):
    def __init__(self, checkpoint_path, input_channels=6, batch_norm=False):
        super(VIS_ENCODER, self).__init__()

        self.var = RootVariables()
        torch.manual_seed(1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = FlowNetS.FlowNetS(batch_norm)
        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.net = nn.Sequential(*list(self.net.children())[0:9]).to(self.device)
        for i in range(len(self.net) - 1):
            self.net[i][1] = nn.ReLU()

        self.fc1 = nn.Linear(1024*6*8, 256).to(self.device)
#        self.fc2 = nn.Linear(4096, 256).to(self.device)
        self.fc3 = nn.Linear(256, 2).to(self.device)
        self.dropout = nn.Dropout(0.3)
        # self.net[8][1] = nn.ReLU(inplace=False)
        self.net[8] = self.net[8][0]

#        for params in self.net.parameters():
#            params.requires_grad = True

    def forward(self, input_img):
        out = self.net(input_img)
        out = out.reshape(-1, 1024*6*8)
        out = F.relu(self.dropout(self.fc1(out)))
        #out = F.leaky_relu(self.dropout(self.fc2(out)), 0.1)
        # out = self.activation(self.fc3(out))

        return out

## PREPARING THE DATA
# folder = sys.argv[1]
# dataset_folder = '/home/sans/Downloads/gaze_data/'
# os.chdir(dataset_folder + folder + '/' if folder[-1]!='/' else (dataset_folder + folder))
if __name__ == "__main__":
    folder = sys.argv[1]
    device = torch.device("cpu")

    var = RootVariables()
    os.chdir(var.root + folder)
    # dataset = FRAME_IMU_DATASET(var.root, folder, 150, device)
    # trainLoader = torch.utils.data.DataLoader(dataset, batch_size=var.batch_size, drop_last=True)
    # a = iter(trainLoader)
    # f, g, i = next(a)
    # # print(data.shape, data)
    # print(i.shape) # [batch_size, sequence_length, input_size]
    # i = i.reshape(i.shape[0], i.shape[2], -1)
    # print(i.shape)

    model = IMU_ENCODER(var.imu_input_size ,device).to(device)
    imuCheckpoint_file = 'hidden_256_60e_signal_pipeline_checkpoint.pth'
    imuCheckpoint = torch.load(var.root + imuCheckpoint_file)
    model.load_state_dict(imuCheckpoint['model_state_dict'])
    print(model)
    # scores = model(data.float())
    # print(model, scores.shape)
    # scores = scores.unsqueeze(dim = 1)
    # newscore = scores.reshape(scores.shape[0], 4, 32)
    # print(newscore.shape)
    # print(newscore)

