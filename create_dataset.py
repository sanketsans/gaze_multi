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
# from helpers import Helpers

class All_Dataset:
    def __init__(self):
        self.var = RootVariables()

    def get_dataset(self, csv_file_name, feat, heatmap_img_csv, index):
        if index == 0:
            return self.SIG_FINAL_DATASET(feat, heatmap_img_csv)
        elif index == 1:
            return self.VIS_FINAL_DATASET(csv_file_name, heatmap_img_csv)
        else:
            return self.FusionPipeline(csv_file_name, feat, heatmap_img_csv)

    class FUSION_DATASET(Dataset):
        def __init__(self, original_img_csv, imu_feat, heatmap_img_csv):
            self.imu_data, self.gaze_data = [], []
            self.indexes = []
            self.ori_imgs_path = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/' + original_img_csv + '.csv')
            self.heat_imgs_path = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/' + heatmap_img_csv + '.csv')
            name_index = 6 if len(self.ori_imgs_path.iloc[0, 1].split('/')) > 4 else 3
            subfolder = self.ori_imgs_path.iloc[0, 1].split('/')[name_index]
            f_index = 0
            for index in range(len(self.heat_imgs_path)):
                imu_check = np.isnan(imu_feat[index])
                if imu_check.any():
                    continue
                else:
                    f_index += 1
                    if self.ori_imgs_path.iloc[f_index, 1].split('/')[name_index] == subfolder:
                        self.indexes.append(f_index)
                    else:
                        f_index += 1
                        self.indexes.append(f_index)
                        subfolder = self.ori_imgs_path.iloc[f_index, 1].split('/')[name_index]

            self.imu_data = standarization(self.imu_data)

            assert len(self.imu_data) == len(self.indexes)
            assert len(self.heat_imgs_path) == len(self.indexes)

            self.transforms = transforms.Compose([
                                            transforms.ToTensor()])

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
            ##Imgs
            for i in range(f_index, f_index-2, -1):
                img = torch.cat((img, self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=3)), axis=3) if i < f_index else self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=3)

            return (img).to("cuda:0"), torch.from_numpy(self.imu_data[index]).to(self.device), transforms.ToTensor()(Image.open(self.heat_imgs_path.iloc[f_index, 1])).to(self.device)

    class VIS_FINAL_DATASET(Dataset):
        def __init__(self, original_img_csv, heatmap_img_csv):
            self.indexes = []
            self.var = RootVariables()
            # os.path.dirname(os.path.realpath(__file__))
            self.ori_imgs_path = pd.read_csv(self.var.root + original_img_csv + '.csv')
            self.heat_imgs_path = pd.read_csv(self.var.root + heatmap_img_csv + '.csv')
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            name_index = 6 if len(self.ori_imgs_path.iloc[0, 1].split('/')) > 7 else 4
            subfolder = self.ori_imgs_path.iloc[0, 1].split('/')[name_index]
            f_index = 0
            for index in range(len(self.heat_imgs_path)):
                f_index += 1
                if self.ori_imgs_path.iloc[f_index, 1].split('/')[name_index] == subfolder:
                    self.indexes.append(f_index)
                else:
                    f_index += 1
                    self.indexes.append(f_index)
                    subfolder = self.ori_imgs_path.iloc[f_index, 1].split('/')[name_index]

            self.transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            assert len(self.heat_imgs_path) == len(self.indexes)

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
            ##Imgs
            for i in range(f_index, f_index-2, -1):
                # print(self.ori_imgs_path.iloc[i, 1])
                img = torch.cat((img, self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1]))), 0) if i < f_index else self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1]))
                # img = torch.cat((img, self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=3)), axis=3) if i < f_index else self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=3)
            # print(self.heat_imgs_path.iloc[index, 1])
            # print('\n')
            return (img).to(self.device), (transforms.ToTensor()(Image.open(self.heat_imgs_path.iloc[index, 1]))).to(self.device)

    class SIG_FINAL_DATASET(Dataset):
        def __init__(self, imu_feat, heatmap_img_csv):
            self.imu_data, self.indexes = [], []
            self.heat_imgs_path = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/' + heatmap_img_csv + '.csv')
            checkedLast = False
            for index in range(len(self.heat_imgs_path)):
                imu_check = np.isnan(imu_feat[index])
                if imu_check.any():
                    continue
                else:
                    self.indexes.append(index)
                    self.imu_data.append(imu_feat[index])

            self.imu_data = standarization(self.imu_data)

            assert len(self.imu_data) == len(self.indexes)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
            return torch.from_numpy(self.imu_data[index]).to(self.device), transforms.ToTensor()(Image.open(self.heat_imgs_path.iloc[f_index, 1])).to(self.device)


if __name__ =='__main__':
    var = RootVariables()
    alld = All_Dataset()

    os.chdir(var.root)
    v = alld.VIS_FINAL_DATASET('trainImg', 'heatmap_trainImg')
    print(v[5998])

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
