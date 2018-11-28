from data_reader_general import *
import argparse
import yaml
import numpy as np

#Set default parameters of preprocessing data
parser = argparse.ArgumentParser(description='TSA')
parser.add_argument('--config', default='cfgs/config_rnn_gcnn_glove_laptop.yaml')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--e', '--evaluate', action='store_true')

args = parser.parse_args()


def train():
    #Load configuration file
    with open(args.config) as f:
        config = yaml.load(f)
        
    for k, v in config['common'].items():
        setattr(args, k, v)
    
    #############Load and process raw files########
    path1 = "data/laptop/Laptop_Train_v2.xml"
    path2 = "data/laptop/Laptops_Test_Gold.xml"
    path_list = [path1, path2]
    #First time, need to preprocess and save the data
    #Read XML file
    dr = data_reader(args)
    dr.read_train_test_data(path_list)
    print('Data Preprocessed!')
    
    
#     ###############Load preprocessed files, split training and dev parts if necessary#########
    dr = data_reader(args)
    data = dr.load_data('data/laptop/Laptop_Train_v2.xml.pkl')
    dr.split_save_data(args.train_path, args.valid_path)
    print('Splitting finished')

    



if __name__ == "__main__":
    train()