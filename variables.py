class Variables:
    def __init__(self):
        self.root = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'
        self.root = '/data/Tobii_dataset/'
        # self.root = '/home/sanketthakur/Downloads/IMU-data_processing/'
        # self.root = '/home/sanketthakur/Downloads/Pavis_Social_Interaction_Attention_dataset/'
        self.gaze_dataList = []     ## list to handle data with multiple single element json items.
        self.timestamps_gaze = []   ## timestamp of data
        self.gaze_data = [[], []]   ## 2D co-ordinates of gaze2D; (x,y)
        self.n_gaze_samples = 0     ## number of samples for corresponding sec.
        self.gaze_data_index = 0    ## used to print the timestamp

        self.imu_dataList = []
        self.timestamps_imu = []
        self.imu_data_acc = [[], [], []]
        self.imu_data_gyro = [[], [], []]
        self.n_imu_samples = 0
        self.check_repeat = False   ## to check if repeation

class RootVariables:
    def __init__(self):
        self.root = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'
        self.root = '/data/Tobii_dataset/'
        # self.root = '/home/sanketthakur/Downloads/IMU-data_processing/'
        # self.root = '/home/sanketthakur/Downloads/Pavis_Social_Interaction_Attention_dataset/'
        self.frame_size = 256
        self.imu_input_size = 128
        self.frame_input_channel = 6
        self.batch_size = 8
        self.hidden_size = 128
        self.num_classes = 256
        self.trim_frame_size = 150
        self.num_layers = 2
        self.test_folder = ''

if __name__ =='__main__':
    var = Variables()
