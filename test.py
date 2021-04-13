import os, cv2
from tqdm import tqdm
import torch, argparse
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    a = np.arange(101)
    c = 0
    for i in range(len(a) - 40):
        print([a[20  - 4 + i+j] for j in range(5)], c)
        c+= 1

    print(len(a), len(a) - 40)
