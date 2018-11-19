from model_gcnn_glove import *
from data_reader_general import *
from configs.config import config
from torch import optim
import pickle

def train():
    TRAIN_DATA_PATH = "data/2014/Laptop_Train_v2.xml"
    TEST_DATA_PATH = "data/2014/Laptops_Test_Gold.xml"
    path_list = [TRAIN_DATA_PATH, TEST_DATA_PATH]
    #First time, need to preprocess and save the data
    #Read XML file
    dr = data_reader(config)
    dr.read_train_test_data(path_list)
    print('Data Preprocessed!')
    
    dr = data_reader(config)
    dr.load_data('data/laptop/Laptop_Train_v2.xml.pkl')
    dr.split_save_data(config.train_path, config.valid_path)
    print('Splitting finished')


if __name__ == "__main__":
    train()