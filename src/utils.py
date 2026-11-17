import torch
import numpy as np
import random

def load_config():
    pass

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
