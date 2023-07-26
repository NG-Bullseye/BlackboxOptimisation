import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from RS_BO.Benchmark import Benchmark, Optimization, GridSearch, RandomSearch
from RS_BO.Utility import load_real_data


class DataClass:

    def __init__(self,INPUT_DATAPATH):
        self.yaw_acc_mapping = load_real_data.main(INPUT_DATAPATH)