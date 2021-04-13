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
        def __init__(self, folder_type, imu_feat, labels):
            self.imu_data = []
            self.indexes = []
            self.folder_type = folder_type
            checkedLast = False
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                imu_check = np.isnan(imu_feat[index])
                if check.any() or imu_check.any():
                    continue
                else:
                    self.indexes.append(index)
                    self.imu_data.append(imu_feat[index])

            self.imu_data = standarization(self.imu_data)

            self.transforms = transforms.Compose([transforms.ToTensor()])

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
    #        img = self.frames[f_index]
            img =  np.load(self.var.root + self.folder_type + '/frames_' + str(f_index) +'.npy')
            targets = self.gaze_data[f_index]
            targets[:,0] *= 512.0
            targets[:,1] *= 384.0

            return self.transforms(img).to("cuda:0"), torch.from_numpy(self.imu_data[index]).to("cuda:0"), torch.from_numpy(targets).to("cuda:0")

    class VIS_FINAL_DATASET(Dataset):
        def __init__(self, csv_file_name, labels):
            self.labels = labels
            self.indexes = []
            self.imgs_path = pd.read_csv('/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/' + csv_file_name + '.csv')
            checkedLast = False
            subfolder = self.imgs_path.iloc[0, 1].split('/')[6]
            contNone, f_index = 3, 0
            for index in range(len(self.labels)):
                check = np.isnan(self.labels[index])
                if check.any():
                    contNone += 1
                    continue
                else:
                    f_index += 1 + contNone
                    if self.imgs_path.iloc[f_index, 1].split('/')[6] == subfolder:
                        self.indexes.append(f_index)
                        contNone = 0
                    else:
                        f_index += 4
                        contNone = 0
                        self.indexes.append(f_index)
                        subfolder = self.imgs_path.iloc[f_index, 1].split('/')[6]

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

            targets = self.labels[index]
            #targets[:,0] *= 0.2667
            #targets[:,1] *= 0.3556

            targets[:,0] *= 512.0
            targets[:,1] *= 384.0

            return (img), torch.from_numpy(targets)

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
    # dataset_folder = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'
    # os.chdir(dataset_folder)
    dataframes = BUILDING_DATASETS('train_Lift_S1')

    dataframes.load_unified_frame_dataset(reset_dataset=0)
    labels, _ = dataframes.load_unified_gaze_dataset()
    labels = labels.reshape(-1, 4, 2)

    ad = All_Dataset()
    dataset = ad.VIS_FINAL_DATASET('trainImg', labels)
    i, l = dataset[1341]
    model = resnet.generate_model(50)
    # print(model)
    i = i.unsqueeze(dim=0)
    x = model(i)
    print(x.shape)
    print(i.shape)
