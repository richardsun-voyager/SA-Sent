from model_gcnn_glove import *
from data_reader_general import *
from configs.config import config
from torch import optim
import pickle

def train():
#     path1 = "data/generated_data/Restaurants_Train_v2.xml"
#     path2 = "data/generated_data/Restaurants_Test_Gold.xml"
#     path3 = "data/generated_data/filename.xml"
#     path4 = "data/generated_data/samples.xml"
#     path_list = [path1, path2, path3, path4]
#     #First time, need to preprocess and save the data
#     #Read XML file
#     dr = data_reader(config)
#     dr.read_train_test_data(path_list)
#     print('Data Preprocessed!')
    
    dr = data_reader(config)
    dr.load_data('data/generated_data/Restaurants_Train_v2.xml.pkl')
    dr.split_save_data(config.train_path, config.valid_path)
    print('Splitting finished')


if __name__ == "__main__":
    train()