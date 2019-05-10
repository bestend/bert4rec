import argparse
import json
import os
import pickle


def read_data(data_path):
    with open(os.path.join(data_path, "data.pickle"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(data_path, "dic.pickle"), "rb") as f:
        dic = pickle.load(f)
    with open(os.path.join(data_path, "params.json"), "r") as f:
        params = json.load(f)
    return data, dic, params


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
