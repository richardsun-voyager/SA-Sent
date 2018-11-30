from data_reader_general import *
import argparse
import yaml
import numpy as np

#Set default parameters of preprocessing data
parser = argparse.ArgumentParser(description='TSA')
parser.add_argument('--config', default='cfgs/config_crf_glove_mitchell.yaml')
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
    path_list = []
    for i in range(10):
        path = 'data/mitchell/train.'+str(i+1)+'.csv'
        path_list.append(path)
        path = 'data/mitchell/test.'+str(i+1)+'.csv'
        path_list.append(path)
    
#     path1 = "data/laptop/Laptops_Train_V2.xml"
#     path2 = "data/laptop/Laptops_Test_Gold.xml"
#     path_list = [path1, path2]
    #####First time, need to preprocess and save the data
    #######Read XML file
    dr = data_reader(args)
    dr.read_train_test_data(path_list, 'csv')
    print('Data Preprocessed!')
    
    
    ###############Load preprocessed files, split training and dev parts if necessary#########
    dr = data_reader(args)
    data = dr.load_data('data/mitchell/train.1.csv.pkl')
    dr.split_save_data(args.train_path, args.valid_path)
    print('Splitting finished')

    



if __name__ == "__main__":
    train()