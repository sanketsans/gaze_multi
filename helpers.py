import math, os, cv2
from variables import RootVariables
from torch.utils.data import Dataset
import numpy as np
from prepare_dataset import IMU_GAZE_FRAME_DATASET
from pathlib import Path

class ALIGN_DATASET(Dataset):
    def __init__(self, imu_data, gaze_data):
        self.imu_data = imu_data
        self.gaze_data = gaze_data
        self.per_file_imu = []
        self.per_file_gaze = []
        checkedLast = False
        for i in range(len(self.gaze_data)): #-1
            index = i #+ 1
            imu_index = 25 + index
            catIMUData = self.imu_data[imu_index-15]
            for i in range(15):#15
                catIMUData = np.concatenate((catIMUData, self.imu_data[imu_index-3+i]), axis=0)
            for i in range(1, 6):#6
                catIMUData = np.concatenate((catIMUData, self.imu_data[imu_index+i]), axis=0)

            self.per_file_imu.append(catIMUData)
            self.per_file_gaze.append(self.gaze_data[index])

        self.per_file_imu = np.array(self.per_file_imu)
        self.per_file_gaze = np.array(self.per_file_gaze)

    def __len__(self):
        return len(self.gaze_data) #- 1

    def __getitem__(self, index):
        return self.per_file_imu[index], self.per_file_gaze[index]

def standarization(datas):
    datas = np.array(datas)
    seq = datas.shape[1]
    datas = datas.reshape(-1, datas.shape[-1])
    rows, cols = datas.shape
    for i in range(cols):
        mean = np.mean(datas[:,i])
        std = np.std(datas[:,i])
        datas[:,i] = (datas[:,i] - mean) / std

    datas = datas.reshape(-1, seq, datas.shape[-1])
    return datas


class Helpers:
    def __init__(self, test_folder):
        self.var = RootVariables()
        self.test_folder = test_folder
        Path(self.var.root + 'datasets').mkdir(parents=True, exist_ok=True)
        Path(self.var.root + 'datasets/' + test_folder[5:]).mkdir(parents=True, exist_ok=True)
        # _ = os.system('mkdir -p' + self.var.root + 'datasets')
        # _ = os.system('mkdir -p' + self.var.root + 'datasets/' + test_folder[5:])

        self.train_folders_num, self.test_folders_num = 0, 0
        self.gaze_start_index, self.gaze_end_index = 0, 0
        self.imu_start_index, self.imu_end_index = 0, 0

    def standarization(self, datas):
        datas = np.array(datas)
        seq = datas.shape[1]
        datas = datas.reshape(-1, datas.shape[-1])
        rows, cols = datas.shape
        for i in range(cols):
            mean = np.mean(datas[:,i])
            std = np.std(datas[:,i])
            datas[:,i] = (datas[:,i] - mean) / std

        datas = datas.reshape(-1, seq, datas.shape[-1])
        return datas

    def normalization(self, datas):
        datas = np.array(datas)
        seq = datas.shape[1]
        datas = datas.reshape(-1, datas.shape[-1])
        rows, cols = datas.shape
        for i in range(cols):
            max = np.max(datas[:,i])
            min = np.min(datas[:,i])
            datas[:,i] = (datas[:,i] - min ) / (max - min)

        datas = datas.reshape(-1, seq, datas.shape[-1])
        return datas

    # def load_datasets_folder(self, folder_name):
    #     frames, imu, gaze = None, None, None
    #     toggle = 0
    #     self.gaze_start_index, self.imu_start_index = 0, 0
    #     for index, subDir in enumerate(sorted(os.listdir(self.var.root))):
    #         if 'train_' in subDir:
    #             if toggle != 1:
    #                 toggle = 1
    #                 self.gaze_start_index, self.imu_start_index = 0, 0
    #
    #             subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
    #             os.chdir(self.var.root + subDir)
    #             capture = cv2.VideoCapture('scenevideo.mp4')
    #             frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #             self.gaze_end_index = self.gaze_start_index + frame_count - self.var.trim_frame_size*2
    #             self.imu_end_index = self.imu_start_index + frame_count - self.var.trim_frame_size
    #             sliced_gaze_dataset = self.train_gaze_dataset[self.gaze_start_index: self.gaze_end_index]
    #             print(sliced_gasze_dataset[0])
    #
    #             if folder_name[6:] in subDir:
    #                 print(folder_name)
    #                 sliced_frame_dataset = np.load(str(self.var.frame_size) + '_framesExtracted_data_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
    #                 sliced_imu_dataset = self.train_imu_dataset[self.imu_start_index: self.imu_end_index]
    #                 sliced_gaze_dataset = self.train_gaze_dataset[self.gaze_start_index: self.gaze_end_index]
    #                 data = ALIGN_DATASET(sliced_frame_dataset, sliced_imu_dataset, sliced_gaze_dataset)
    #                 frames, imu, gaze = data.per_file_frame, data.per_file_imu, data.per_file_gaze
    #                 print(sliced_gaze_dataset[0])
    #                 break
    #
    #             self.gaze_start_index = self.gaze_end_index
    #             self.imu_start_index = self.imu_end_index
    #
    #         elif 'test_' in subDir:
    #             if toggle != -1:
    #                 toggle = -1
    #                 self.gaze_start_index, self.imu_start_index = 0, 0
    #
    #             subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
    #             os.chdir(self.var.root + subDir)
    #             capture = cv2.VideoCapture('scenevideo.mp4')
    #             frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #             self.gaze_end_index = self.gaze_start_index + frame_count - self.var.trim_frame_size*2
    #             self.imu_end_index = self.imu_start_index + frame_count - self.var.trim_frame_size
    #
    #             if folder_name[6:] in subDir:
    #                 print(folder_name)
    #                 sliced_frame_dataset = np.load(str(self.var.frame_size) + '_framesExtracted_data_' + str(self.var.trim_frame_size) + '.npy', mmap_mode='r')
    #                 sliced_imu_dataset = self.test_imu_dataset[self.imu_start_index: self.imu_end_index]
    #                 sliced_gaze_dataset = self.test_gaze_dataset[self.gaze_start_index: self.gaze_end_index]
    #                 data = ALIGN_DATASET(sliced_frame_dataset, sliced_imu_dataset, sliced_gaze_dataset)
    #                 frames, imu, gaze = data.per_file_frame, data.per_file_imu, data.per_file_gaze
    #                 print(sliced_gaze_dataset[0])
    #                 break
    #
    #             self.gaze_start_index = self.gaze_end_index
    #             self.imu_start_index = self.imu_end_index
    #
    #
    #     return frames, imu, gaze

    def load_datasets(self, reset_dataset, repeat):
        self.dataset = IMU_GAZE_FRAME_DATASET(self.test_folder, reset_dataset)
        self.train_imu_dataset, self.test_imu_dataset = self.dataset.imu_train_datasets, self.dataset.imu_test_datasets
        self.train_gaze_dataset, self.test_gaze_dataset = self.dataset.gaze_train_datasets, self.dataset.gaze_test_datasets

        test_folder = self.test_folder
        test_folder  = test_folder + '/' if test_folder[-1]!='/' else  test_folder
        toggle = 0
        imu_training_feat, imu_testing_feat = None, None
        training_target, testing_target = None, None

        # check = False if Path(self.var.root + 'datasets/' + test_folder[5:] + str(self.var.frame_size) + '_imu_training_feat_' + test_folder[5:-1] + '.npy').is_file() else False
        if repeat == 1 :
            imu_training_feat = np.load(self.var.root + 'datasets/' + test_folder[5:] + '_imu_training_feat_' + test_folder[5:-1]  + '.npy', allow_pickle=True)
            imu_testing_feat = np.load(self.var.root + 'datasets/' + test_folder[5:] + '_imu_testing_feat_' + test_folder[5:-1]  + '.npy', allow_pickle=True)
            # training_target = np.load(self.var.root + 'datasets/' + test_folder[5:] + '_gaze_training_target_' + test_folder[5:-1]  + '.npy', allow_pickle=True)
            # testing_target = np.load(self.var.root + 'datasets/' + test_folder[5:] + '_gaze_testing_target_' + test_folder[5:-1]  + '.npy', allow_pickle=True)

        else:
            for index, subDir in enumerate(sorted(os.listdir(self.var.root))):
                if 'washands' in subDir:
                    if toggle != 1:
                        toggle = 1
                        self.gaze_start_index, self.imu_start_index = 0, 0
                    print(subDir)
                    self.train_folders_num += 1
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(self.var.root + subDir)
                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.gaze_end_index = self.gaze_start_index + frame_count - self.var.trim_frame_size*2
                    self.imu_end_index = self.imu_start_index + frame_count - self.var.trim_frame_size
                    sliced_imu_dataset = self.train_imu_dataset[self.imu_start_index: self.imu_end_index]
                    sliced_gaze_dataset = self.train_gaze_dataset[self.gaze_start_index: self.gaze_end_index]
                    data = ALIGN_DATASET(sliced_imu_dataset, sliced_gaze_dataset)

                    if self.train_folders_num > 1:
                        imu_training_feat, training_target = np.concatenate((imu_training_feat, data.per_file_imu), axis=0),np.concatenate((training_target, data.per_file_gaze), axis=0)
                    else:
                        imu_training_feat, training_target = data.per_file_imu, data.per_file_gaze

                    self.gaze_start_index = self.gaze_end_index
                    self.imu_start_index = self.imu_end_index

                if 'test_' in subDir:
                    if toggle != -1:
                        toggle = -1
                        self.gaze_start_index, self.imu_start_index = 0, 0

                    self.test_folders_num += 1
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(self.var.root + subDir)
                    capture = cv2.VideoCapture('scenevideo.mp4')
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.gaze_end_index = self.gaze_start_index + frame_count - self.var.trim_frame_size*2
                    self.imu_end_index = self.imu_start_index + frame_count - self.var.trim_frame_size
                    sliced_imu_dataset = self.test_imu_dataset[self.imu_start_index: self.imu_end_index]
                    sliced_gaze_dataset = self.test_gaze_dataset[self.gaze_start_index: self.gaze_end_index]
                    data = ALIGN_DATASET(sliced_imu_dataset, sliced_gaze_dataset)

                    if self.test_folders_num > 1:
                        imu_testing_feat, testing_target = np.concatenate((imu_testing_feat, data.per_file_imu), axis=0),np.concatenate((testing_target, data.per_file_gaze), axis=0)
                    else:
                        imu_testing_feat, testing_target = data.per_file_imu, data.per_file_gaze

                    self.gaze_start_index = self.gaze_end_index
                    self.imu_start_index = self.imu_end_index

            with open(self.var.root + 'datasets/' + test_folder[5:] + '_imu_training_feat_' + test_folder[5:-1] + '.npy', 'wb') as f:
                np.save(f, imu_training_feat)
                f.close()
            with open(self.var.root + 'datasets/' + test_folder[5:] + '_imu_testing_feat_' + test_folder[5:-1] + '.npy', 'wb') as f:
                np.save(f, imu_testing_feat)
                f.close()
            # with open(self.var.root + 'datasets/' + test_folder[5:] + '_gaze_training_target_' + test_folder[5:-1] + '.npy', 'wb') as f:
            #     np.save(f, training_target)
            #     f.close()
            # with open(self.var.root + 'datasets/' + test_folder[5:] + '_gaze_testing_target_' + test_folder[5:-1] + '.npy', 'wb') as f:
            #     np.save(f, testing_target)
            #     f.close()

        return imu_training_feat, imu_testing_feat, training_target, testing_target

if __name__ == "__main__":
    utils = Helpers('test_Lift_S1')
    _, _, t, te = utils.load_datasets()
    print(len(t), len(te))
