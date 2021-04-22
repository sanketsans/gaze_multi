import sys, os
import numpy as np
import pandas as pd
import torch.nn as nn
import cv2
import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Dataset
from torchvision import transforms
sys.path.append('../')
# from FlowNetPytorch.models import FlowNetS
from variables import RootVariables
from helpers import standarization
from build_dataset import BUILDING_DATASETS
from PIL import Image
from resnetpytorch.models import resnet

class All_Dataset:
    def __init__(self):
        self.var = RootVariables()

    def get_dataset(self, folder_type, feat, labels, index):
        if index == 0:
            return self.SIG_FINAL_DATASET(feat, labels)
        elif index == 1:
            return self.VIS_FINAL_DATASET(folder_type, labels)
        else:
            return self.FusionPipeline(folder_type, feat, labels)

    class FUSION_DATASET(Dataset):
        def __init__(self, csv_file_name, imu_feat, labels):
            self.imu_data, self.gaze_data = [], []
            self.indexes = []
            self.imgs_path = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/' + csv_file_name + '.csv')
            checkedLast = False
            name_index = 6 if len(self.imgs_path.iloc[0, 1].split('/')) > 4 else 3
            subfolder = self.imgs_path.iloc[0, 1].split('/')[name_index]
            contNone, f_index = 3, 0
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                imu_check = np.isnan(imu_feat[index])
                if check.any() or imu_check.any():
                    contNone += 1
                    continue
                else:
                    f_index += 1 + contNone
                    if self.imgs_path.iloc[f_index, 1].split('/')[name_index] == subfolder:
                        self.indexes.append(f_index)
                        contNone = 0
                    else:
                        f_index += 4
                        contNone = 0
                        self.indexes.append(f_index)
                        subfolder = self.imgs_path.iloc[f_index, 1].split('/')[name_index]
                    self.imu_data.append(imu_feat[index])
                    self.gaze_data.append(labels[index])

            self.imu_data = standarization(self.imu_data)

            assert len(self.imu_data) == len(self.indexes)
            assert len(self.gaze_data) == len(self.indexes)

            self.transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
            ##Imgs
            for i in range(f_index, f_index-5, -1):
                img = torch.cat((img, self.transforms(Image.open(self.imgs_path.iloc[i, 1])).unsqueeze(dim=3)), axis=3) if i < f_index else self.transforms(Image.open(self.imgs_path.iloc[i, 1])).unsqueeze(dim=3)

            targets = self.gaze_data[index]

            targets[:,0] *= 512.0
            targets[:,1] *= 384.0

            return (img).to("cuda:0"), torch.from_numpy(self.imu_data[index]).to("cuda:0"), torch.from_numpy(targets).to("cuda:0")

    class VIS_FINAL_DATASET(Dataset):
        def __init__(self, csv_file_name, labels):
            self.gaze_data = []
            self.indexes = []
            self.imgs_path = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/' + csv_file_name + '.csv')
            checkedLast = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            name_index = 6 if len(self.imgs_path.iloc[0, 1].split('/')) > 7 else 3
            subfolder = self.imgs_path.iloc[0, 1].split('/')[name_index]
            contNone, f_index = 3, 0
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                if check.any():
                    contNone += 1
                    continue
                else:
                    f_index += 1 + contNone
                    if self.imgs_path.iloc[f_index, 1].split('/')[name_index] == subfolder:
                        self.indexes.append(f_index)
                        contNone = 0
                    else:
                        f_index += 4
                        contNone = 0
                        self.indexes.append(f_index)
                        subfolder = self.imgs_path.iloc[f_index, 1].split('/')[name_index]
                    self.gaze_data.append(labels[index])

            self.transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            assert len(self.gaze_data) == len(self.indexes)

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
            ##Imgs
            for i in range(f_index, f_index-5, -1):
                img = torch.cat((img, self.transforms(Image.open(self.imgs_path.iloc[i, 1])).unsqueeze(dim=3)), axis=3) if i < f_index else self.transforms(Image.open(self.imgs_path.iloc[i, 1])).unsqueeze(dim=3)

            targets = self.gaze_data[index]
            #targets[:,0] *= 0.2667
            #targets[:,1] *= 0.3556

            targets[:,0] *= 512.0
            targets[:,1] *= 384.0

            return (img).to(self.device), torch.from_numpy(targets).to(self.device)

    class SIG_FINAL_DATASET(Dataset):
        def __init__(self, feat, labels):
            self.gaze_data, self.imu_data = [], []
            checkedLast = False
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                imu_check = np.isnan(feat[index])
                if check.any() or imu_check.any():
                    continue
                else:
                    self.gaze_data.append(labels[index])
                    self.imu_data.append(feat[index])

            self.imu_data = standarization(self.imu_data)

            assert len(self.imu_data) == len(self.gaze_data)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def __len__(self):
            return len(self.gaze_data) # len(self.labels)

        def __getitem__(self, index):
            targets = self.gaze_data[index]
            targets[:,0] *= 512.0
            targets[:,1] *= 384.0

            return torch.from_numpy(self.imu_data[index]).to(self.device), torch.from_numpy(targets).to(self.device)


if __name__ =='__main__':
    var = RootVariables()

    # dataframes = BUILDING_DATASETS('train_Lift_S1')
    # #
    # dataframes.load_unified_frame_dataset(reset_dataset=1)
    # labels, _ = dataframes.load_unified_gaze_dataset()
    # labels = labels.reshape(-1, 4, 2)
    #
    # ad = All_Dataset()
    # dataset = ad.VIS_FINAL_DATASET('trainImg', labels)
    # i, l = dataset[400]
    # model = resnet.generate_model(50)
    # # model.fc = nn.Linear(2048, 1039)
    # # dict = torch.load(var.root + 'r3d50_KM_200ep.pth')
    # # model.load_state_dict(dict["state_dict"])
    # print(model.avgpool)
    #
    # i = i.unsqueeze(dim=0)
    # x = model(i)
    # fc = nn.Linear(400, 2)
    # a = nn.Sigmoid()
    # y = a(fc(x))
    # print(x.shape)
    # print(y.shape)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
