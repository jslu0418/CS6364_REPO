# Functions for load datasets for HW4.
import numpy as np
import pandas as pd

def load_boston():
    '''
    Load Boston housing dataset.
    '''
    # boston_dataset = load_boston()
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    return (data, target)
