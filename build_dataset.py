import os
import sys, math
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
sys.path.append('../')
from loader import JSON_LOADER
from variables import RootVariables
import matplotlib.pyplot as plt
from torchvision import transforms

class BUILDING_DATASETS:
    def __init__(self, test_folder):
        self.var = RootVariables()
        self.dataset = None
        self.imu_arr_acc, self.imu_arr_gyro, self.gaze_arr = None, None, None
        self.train_new, self.test_new = None, None
        temp = None
        self.video_file = 'scenevideo.mp4'
        self.test_folders_num, self.train_folders_num = 0, 0
        self.frame_count = 0
        self.capture = None
        self.ret = None
        self.toggle = 0
        self.test_folder = test_folder
        self.stack_frames = []
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.panda_data = {}

    def populate_gaze_data(self, subDir):

        subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
        os.chdir(self.var.root + subDir)
        capture = cv2.VideoCapture(self.video_file)
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.dataset = JSON_LOADER(subDir)
        self.dataset.POP_GAZE_DATA(self.frame_count)
        self.gaze_arr = np.array(self.dataset.var.gaze_data).transpose()
        _ = os.system('rm gaze_file.csv')
        self.panda_data = {}
        self.create_dataframes(subDir, 'gaze')
        self.gaze_arr = np.array(self.dataset.var.gaze_data).transpose()

        temp = np.zeros((self.frame_count*4-self.var.trim_frame_size*4*2, 2))
        temp[:,0] = self.gaze_arr[tuple([np.arange(self.var.trim_frame_size*4, self.frame_count*4 - self.var.trim_frame_size*4), [0]])]
        temp[:,1] = self.gaze_arr[tuple([np.arange(self.var.trim_frame_size*4, self.frame_count*4 - self.var.trim_frame_size*4), [1]])]
        return temp

    def load_unified_gaze_dataset(self):        ## missing data in imu_lift_s1
        self.test_folders_num, self.train_folders_num = 0, 0
        print('Building gaze dataset ..')
        tqdmloader = tqdm(sorted(os.listdir(self.var.root)))
        for index, subDir in enumerate(tqdmloader):
            if 'train_BookShelf_S1' in subDir:
                tqdmloader.set_description('Train folder: {}'.format(subDir))
                self.temp = self.populate_gaze_data(subDir)
                self.train_folders_num += 1
                if self.train_folders_num > 1:
                    self.train_new = np.concatenate((self.train_new, self.temp), axis=0)
                else:
                    self.train_new = self.temp

            if 'test_XX' in subDir:
                tqdmloader.set_description('Test folder: {}'.format(subDir))
                self.temp = self.populate_gaze_data(subDir)
                self.test_folders_num += 1
                if self.test_folders_num > 1:
                    self.test_new = np.concatenate((self.test_new, self.temp), axis=0)
                else:
                    self.test_new = self.temp
                print(subDir, len(self.test_new.reshape(-1, 4, 2)))

        return self.train_new, self.test_new

    def create_clips(self, cap, index, type):
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter('/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/' + type + '/output_' + str(index) + '.avi', fourcc, 1.0, (224,224))
        chunks = None
        for i in range(5):
            _, frame = cap.read()
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transforms(frame)
            frame = frame.unsqueeze(dim=3)
            chunks = torch.cat((chunks, frame), axis=3) if i > 0 else frame
            out.write(frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES,150+index-4)
        out.release()

    def load_unified_frame_dataset(self, reset_dataset=0):
        ## INCLUDES THE LAST FRAME
        if reset_dataset == 1:
            print('Deleting the old dataset .. ')
            _ = os.system('rm -r ' + self.var.root + 'training_images')
            _ = os.system('rm -r ' + self.var.root + 'testing_images')

            _ = os.system('mkdir ' + self.var.root + 'training_images')
            _ = os.system('mkdir ' + self.var.root + 'testing_images')
            train_frame_index, test_frame_index = 0, 0
            trainpaths, testpaths = [], []
            print("Building Image dataset ..")
            tqdmloader = tqdm(sorted(os.listdir(self.var.root)))
            for index, subDir in enumerate(tqdmloader):
                if 'train_BookShelf_S1' in subDir :
                    tqdmloader.set_description('Train folder: {}'.format(subDir))
                    _ = os.system('mkdir ' + self.var.root + 'training_images/' + subDir)
                    total_frames = 0
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(self.var.root + subDir)
                    self.capture = cv2.VideoCapture(self.video_file)
                    self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    os.chdir(self.var.root + subDir)
                    self.capture = cv2.VideoCapture(self.video_file)
                    self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

                    self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.var.trim_frame_size - 4)
                    for i in range(self.frame_count - 300 + 4):
                        _, frame = self.capture.read()
                        frame = cv2.resize(frame, (398, 224))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        path = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/training_images/' + subDir + 'image_' + str(train_frame_index) + '.jpg'
                        cv2.imwrite(path, frame)
                        # self.create_clips(self.capture, train_frame_index, 'training_images')
                        train_frame_index += 1
                        trainpaths.append(path)

                if 'test_XX' in subDir:
                    tqdmloader.set_description('Test folder: {}'.format(subDir))
                    total_frames = 0
                    subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
                    os.chdir(self.var.root + subDir)
                    self.capture = cv2.VideoCapture(self.video_file)
                    self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
#                       _ = os.system('rm ' + str(self.var.frame_size) + '_framesExtracted_data_' + str(self.var.trim_frame_size) + '.npy')
                    os.chdir(self.var.root + subDir)
                    self.capture = cv2.VideoCapture(self.video_file)
                    self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

                    self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.var.trim_frame_size - 4)
                    for i in range(self.frame_count - 300 + 4):
                        _, frame = self.capture.read()
                        frame = cv2.resize(frame, (398, 224))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        path = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/testing_images/' + subDir + 'image_' + str(test_frame_index) + '.jpg'
                        cv2.imwrite(path, frame)
                        # self.create_clips(self.capture, test_frame_index, 'testing_images')
                        test_frame_index += 1
                        testpaths.append(path)
                    print(test_frame_index)

            dict = {'image_paths': trainpaths}
            df = pd.DataFrame(dict)
            df.to_csv(os.path.dirname(os.path.realpath(__file__)) + '/trainImg.csv')
            dict = {'image_paths':testpaths}
            df = pd.DataFrame(dict)
            df.to_csv(os.path.dirname(os.path.realpath(__file__)) + '/testImg.csv')

    def populate_imu_data(self, subDir):

        subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
        os.chdir(self.var.root + subDir)
        capture = cv2.VideoCapture(self.video_file)
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.dataset = JSON_LOADER(subDir)
        self.dataset.POP_IMU_DATA(self.frame_count, cut_short=True)
        _ = os.system('rm imu_file.csv')
        self.panda_data = {}
        self.create_dataframes(subDir, dframe_type='imu')

        self.imu_arr_acc = np.array(self.dataset.var.imu_data_acc).transpose()
        self.imu_arr_gyro = np.array(self.dataset.var.imu_data_gyro).transpose()
        temp = np.zeros((len(self.imu_arr_acc) , 6))

        temp = np.zeros((self.frame_count*4-self.var.trim_frame_size*4, 6))
        temp[:,0] = self.imu_arr_acc[tuple([np.arange(self.var.trim_frame_size*2, self.frame_count*4 - self.var.trim_frame_size*2), [0]])]
        temp[:,1] = self.imu_arr_acc[tuple([np.arange(self.var.trim_frame_size*2, self.frame_count*4 - self.var.trim_frame_size*2), [1]])]
        temp[:,2] = self.imu_arr_acc[tuple([np.arange(self.var.trim_frame_size*2, self.frame_count*4 - self.var.trim_frame_size*2), [2]])]
        temp[:,3] = self.imu_arr_gyro[tuple([np.arange(self.var.trim_frame_size*2, self.frame_count*4 - self.var.trim_frame_size*2), [0]])]
        temp[:,4] = self.imu_arr_gyro[tuple([np.arange(self.var.trim_frame_size*2, self.frame_count*4 - self.var.trim_frame_size*2), [1]])]
        temp[:,5] = self.imu_arr_gyro[tuple([np.arange(self.var.trim_frame_size*2, self.frame_count*4 - self.var.trim_frame_size*2), [2]])]

        return temp

    def load_unified_imu_dataset(self):     ## missing data in imu_CoffeeVendingMachine_S2
        print('Building IMU dataset ..')
        tqdmloader = tqdm(sorted(os.listdir(self.var.root)))
        for index, subDir in enumerate(tqdmloader):
            if 'train_Book' in subDir :
                tqdmloader.set_description('Train folder: {}'.format(subDir))
                self.temp = self.populate_imu_data(subDir)
                self.train_folders_num += 1
                if self.train_folders_num > 1:
                    self.train_new = np.concatenate((self.train_new, self.temp), axis=0)
                else:
                    self.train_new = self.temp

            if 'test_XX' in subDir:
                tqdmloader.set_description('Test folder: {}'.format(subDir))
                self.temp = self.populate_imu_data(subDir)
                self.test_folders_num += 1
                if self.test_folders_num > 1:
                    self.test_new = np.concatenate((self.test_new, self.temp), axis=0)
                else:
                    self.test_new = self.temp
                print(len(self.test_new.reshape(-1, 4, 6)))

        return self.train_new, self.test_new

    def create_dataframes(self, subDir, dframe_type, start_index=0):
        if dframe_type == 'gaze':
            ## GAZE
            for sec in range(self.frame_count):
                self.panda_data[sec] = list(zip(self.dataset.var.gaze_data[0][start_index:start_index + 4], self.dataset.var.gaze_data[1][start_index:start_index+4]))
                start_index += 4

            self.df_gaze = pd.DataFrame({ key:pd.Series(value) for key, value in self.panda_data.items()}).T
            self.df_gaze.columns =['Gaze_Pt_1', 'Gaze_Pt_2', 'Gaze_Pt_3', 'Gaze_Pt_4']
            self.df_gaze.to_csv('gaze_file.csv')

        elif dframe_type == 'imu':
            ## IMU
            for sec in range(self.frame_count):
                # self.panda_data[sec] = list(tuple((sec, sec+2)))
                self.panda_data[sec] = list(zip(zip(self.dataset.var.imu_data_acc[0][start_index:start_index+4],
                                            self.dataset.var.imu_data_acc[1][start_index:start_index+4],
                                            self.dataset.var.imu_data_acc[2][start_index:start_index+4]),

                                        zip(self.dataset.var.imu_data_gyro[0][start_index:start_index+4],
                                                self.dataset.var.imu_data_gyro[1][start_index:start_index+4],
                                                self.dataset.var.imu_data_gyro[2][start_index:start_index+4])))
                start_index += 4
            self.df_imu = pd.DataFrame({ key:pd.Series(value) for key, value in self.panda_data.items()}).T
            self.df_imu.columns =['IMU_Acc/Gyro_Pt_1', 'IMU_Acc/Gyro_Pt_2', 'IMU_Acc/Gyro_Pt_3', 'IMU_Acc/Gyro_Pt_4']
            self.df_imu.to_csv('imu_file.csv')

if __name__ == "__main__":
    var = RootVariables()
    # dataset_folder = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'
    # os.chdir(dataset_folder)
    dataframes = BUILDING_DATASETS('train_Lift_S1')

    dataframes.load_unified_frame_dataset(reset_dataset=1)
    dataframes.load_unified_gaze_dataset()

    # trainIMU, testIMU = dataframes.load_unified_imu_dataset()
    # imu_datas= dataframes.load_unified_imu_dataset()
    # plt.subplot(221)
    # _ = plt.hist(imu_datas[:,0], bins='auto', label='before N')
    # normal = dataframes.normalization(imu_datas)
    # _ = plt.hist(normal[:,0], bins='auto', label='after N')
    # plt.legend()

    # imu_datas= dataframes.load_unified_imu_dataset()
    # plt.subplot(222)
    # _ = plt.hist(imu_datas[:,0], bins='auto', label='before S')
    # normal = dataframes.standarization(imu_datas)
    # _ = plt.hist(normal[:,0], bins='auto', label='after S')
    # plt.legend()
    # plt.show()
