from data_reader_mask_target import *
#from data_reader_general import *
import argparse
import yaml
import numpy as np

#Set default parameters of preprocessing data
parser = argparse.ArgumentParser(description='TSA')
parser.add_argument('--config', default='cfgs/tweets_mask_target/config_gcnn_glove_tweets_mask_target.yaml')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--e', '--evaluate', action='store_true')

args = parser.parse_args()


def train():
    #Load configuration file
    with open(args.config) as f:
        config = yaml.load(f)
        
    for k, v in config['common'].items():
        setattr(args, k, v)
    
    #############Load and process raw mitchell files########
#     path_list = []
#     for i in range(10):
#         path = 'data/mitchell/train.'+str(i+1)+'.csv'
#         path_list.append(path)
#         path = 'data/mitchell/test.'+str(i+1)+'.csv'
#         path_list.append(path)


#     ###########Load and process laptop data###########
    
#     path1 = "data/laptop/Laptop_Train_v2.xml"
#     path2 = "data/laptop/Laptops_Test_Gold.xml"
#     path_list = [path1, path2]
#     #####First time, need to preprocess and save the data
#     #######Read XML file


    ###########Load and process lidong tweet data###########
    
    path1 = "data/tweets_mask_target/train.csv"
    path2 = "data/tweets_mask_target/test.csv"
    path_list = [path1, path2]
    #####First time, need to preprocess and save the data
    #######Read XML file
    
    
    #############Load and process restaurant data###########
    dr = data_reader(args)
#     path1 = "data/restaurant_parse/Restaurants_Train_v2.xml"
#     path2 = "data/restaurant_parse/Restaurants_Test_Gold.xml"
#     path_list = [path1, path2]


#     ########Load Indonesian data###############
#     path1 = "data/indo/train.xml"
#     path2 = "data/indo/dev.xml"
#     path3 = "data/indo/test.xml"
#     path_list = [path1, path2, path3]

#     dr.read_train_test_data(path_list, 'csv')
#     print('Data Preprocessed!')
    
    
    ###############Load preprocessed files, split training and dev parts if necessary#########
    dr = data_reader(args)
    data = dr.load_data('data/tweets_mask_target/train.csv.pkl')
    dr.split_save_data(args.train_path, args.valid_path)
    print('Splitting finished')

    



if __name__ == "__main__":
    train()