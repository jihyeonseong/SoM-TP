import os
import random
import argparse
import datetime
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

import torch
from utils.columns import uni_data_name, mul_data_name
from utils.data_load import TimeSeriesWithLabels, load_uea_dataset, load_ucr_dataset

from model.ConvPool import ConvPool
from utils.lrp_convpool import LRP_ConvPool

from model.SoMTP import SoMTP
from utils.lrp_somtp import LRP_SoMTP

# seed fix #
random_seed=10
torch.manual_seed(random_seed) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 
np.random.seed(random_seed) 
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 


def main(args, train_dataset, valid_dataset, test_dataset, num, data_type, model_folder, result_folder, name):
    if args.model == 'ConvPool':
        model = ConvPool(input_size= train_dataset.input_size, 
                          time_length = train_dataset.timelength,
                          classes= train_dataset.num_classes, 
                          data_type= data_type,
                          args= args
                          )
        LRP_ConvPool(args, train_dataset, test_dataset, num, data_type, model, model_folder, result_folder, name)
      
    elif args.model == 'SoMTP':
        model = SoMTP(input_size= train_dataset.input_size, 
                          time_length = train_dataset.timelength,
                          classes= train_dataset.num_classes, 
                          data_type= data_type,
                          args= args
                          )
        LRP_SoMTP(args, train_dataset, test_dataset, num, data_type, model, model_folder, result_folder, name)
    
    else:
        raise Exception("No model!")
        
        
if __name__=='__main__':    
    # Arguments parsing
    parser = argparse.ArgumentParser()
    
    ### uni-var ###           
    for i, dataset in enumerate(uni_data_name):
        train_dataset = TimeSeriesWithLabels(dataset, 'univar', 'TRAIN', 'train') 
        valid_dataset = TimeSeriesWithLabels(dataset, 'univar', 'TRAIN', 'valid') 
        test_dataset = TimeSeriesWithLabels(dataset, 'univar', 'TEST', 'test') 

        parser = argparse.ArgumentParser()        
        parser.add_argument('--gpuidx', default=1, type=int, help='gpu index')
        parser.add_argument('--model', default='ConvPool', type=str, help='ConvPool | ConvSwitch')
        parser.add_argument('--pool', default='STP', type=str, help='GTP | STP | DTP | Switch')
        parser.add_argument('--pool_op', default='MAX', type=str, help='MAX | AVG')
        parser.add_argument('--deep_extract', default='FCN', type=str, help='only FCN available')
        parser.add_argument('--proto_num', default=5, type=int, help = '3 | 4 | 5 | 6 | 7')
        parser.add_argument('--cost_type', default='euclidean', type=str, help='cosine | dotprod | euclidean')

        parser.add_argument('--batch_size', default=8, type=int, help='batch size')
        parser.add_argument('--num_epoch', default=300, type=int, help='# of training epochs')
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--decay', default=1e-1, type=float, help='lambda decay')

        parser.add_argument('--gtp', default='MAX', type=str, help='GTP pool type')
        parser.add_argument('--stp', default='MAX', type=str, help='STP pool type')
        parser.add_argument('--dtp', default='MAX', type=str, help='DTP pool type')

        parser.add_argument('--batch', default=0, type=int, help='LRP batch')
        parser.add_argument('--sample', default=0, type=int, help='LRP sample')

        args = parser.parse_args()
        print(i, dataset)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)
        
        if args.deep_extract != 'FCN':
            raise Exception("Not supported!")

        if args.model == 'ConvPool':
            model_folder = f'./checkpoint/{args.model}_FCN_{args.pool}_{args.pool_op}' 
            result_folder = f'./checkpoint/{args.model}_FCN_{args.pool}_{args.pool_op}_LRP'
            os.makedirs(result_folder, exist_ok=True)
        else:
            model_folder = f'./checkpoint/{args.model}_FCN_{args.pool_op}' 
            result_folder = f'./checkpoint/{args.model}_FCN_{args.pool_op}_LRP'
            os.makedirs(result_folder, exist_ok=True)

        main(args=args, train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, num=i, data_type='uni', model_folder=model_folder, result_folder=result_folder, name = dataset) 
    
    
    
    
    