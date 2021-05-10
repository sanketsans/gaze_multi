import os, sys
import numpy as np
import pandas as pd
sys.path.append('../')
from variables import RootVariables

if __name__ == '__main__':
    var = RootVariables()
    folder = ''
    folderloc = '/data/Tobii_dataset/heatmap_testing_images/test_sanket_drawing_S1'# vr.root + folder
    a = []
    for files in enumerate(sorted(os.listdir(folderloc))):
        # print(files[1])
        a.append(folderloc + '/' + files[1])

    a = np.array(a)

    df = pd.DataFrame(a, columns=['image_paths'])
    df.to_csv('/data/Tobii_dataset//sample.csv', index=True)
