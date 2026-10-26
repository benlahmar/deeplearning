import math, time, copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
torch.set_num_threads(4)
print('PyTorch :', torch.__version__)
