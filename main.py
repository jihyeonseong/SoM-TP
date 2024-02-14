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
from utils.train_convpool import train_ConvPool

from model.SoMTP import SoMTP
from utils.train_somtp import train_SoMTP

def main(args, train_dataset, valid_dataset, test_dataset, num, data_type):
    performance = []
    if args.model == 'ConvPool':
        model = ConvPool(input_size= train_dataset.input_size, 
                          time_length = train_dataset.timelength,
                          classes= train_dataset.num_classes, 
                          data_type= data_type,
                          args= args
                          )
        performance = train_ConvPool(args, train_dataset, valid_dataset, test_dataset, num, data_type, model, result_folder)
        
    elif args.model == 'SoMTP':
        model = SoMTP(input_size= train_dataset.input_size, 
                       time_length = train_dataset.timelength,
                       classes= train_dataset.num_classes, 
                       data_type= data_type,
                       args= args
                       )
        performance = train_SoMTP(args, train_dataset, valid_dataset, test_dataset, num, data_type, model, result_folder)
    
    else:
        raise Exception("No model!")
               
    return performance

if __name__=='__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser()
    
    ### uni-var ###    
    report_perform = pd.DataFrame()
    for i, dataset in enumerate(uni_data_name):
        train_dataset = TimeSeriesWithLabels(dataset, 'univar', 'TRAIN', 'train') 
        valid_dataset = TimeSeriesWithLabels(dataset, 'univar', 'TRAIN', 'valid') 
        test_dataset = TimeSeriesWithLabels(dataset, 'univar', 'TEST', 'test') 

        parser = argparse.ArgumentParser()    
        parser.add_argument('--result_folder', default='./checkpoint', type=str, help='result folder path')
        parser.add_argument('--gpuidx', default=1, type=int, help='gpu index')
        parser.add_argument('--model', default='ConvPool', type=str, help='ConvPool | SoM-TP')
        parser.add_argument('--deep_extract', default='FCN', type=str, help='FCN | ResNet')
        parser.add_argument('--pool', default='STP', type=str, help='GTP | STP | DTP | Switch')
        parser.add_argument('--pool_op', default='MAX', type=str, help='MAX | AVG')
        parser.add_argument('--proto_num', default=5, type=int, help = '3 | 4 | 5 | 6 | 7')
        parser.add_argument('--cost_type', default='euclidean', type=str, help='DTP cost type: cosine | dotprod | euclidean')

        parser.add_argument('--seed', default=10, type=int, help='seed test')
        parser.add_argument('--batch_size', default=8, type=int, help='batch size')
        parser.add_argument('--num_epoch', default=300, type=int, help='# of training epochs')
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--decay', default=1e-1, type=float, help='SoM-TP lambda decay')

        parser.add_argument('--gtp', default='MAX', type=str, help='GTP pool type')
        parser.add_argument('--stp', default='MAX', type=str, help='STP pool type')
        parser.add_argument('--dtp', default='MAX', type=str, help='DTP pool type')

        args = parser.parse_args()
        print(i, dataset)

        random_seed=args.seed
        torch.manual_seed(random_seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed) 
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)

        if i == 0:
            result_folder = os.path.join(args.result_folder, f'{datetime.datetime.now()}_{args.seed}')
            os.makedirs(result_folder, exist_ok=True)

        perform = main(args=args, train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, num=i, data_type='uni')

        perform_df = pd.DataFrame(perform, index=['loss', 'acc', 'f1macro', 'f1micro', 'f1weight', 'auroc', 'prauc'], columns = [dataset]).T
        report_perform = pd.concat([report_perform, perform_df], axis=0)
        report_perform.to_csv(os.path.join(result_folder, f'{args.model}_{args.deep_extract}_{args.pool}_{args.pool_op}_uni_performance.csv'))

        pd.DataFrame(vars(args), index=np.arange(1)).to_csv(os.path.join(result_folder, 'log.csv'))

    ### multi-var ###
    report_perform = pd.DataFrame()    
    for i, dataset in enumerate(mul_data_name):
        train_dataset = TimeSeriesWithLabels(dataset, 'multivar', 'TRAIN', 'train') 
        valid_dataset = TimeSeriesWithLabels(dataset, 'multivar', 'TRAIN', 'valid') 
        test_dataset = TimeSeriesWithLabels(dataset, 'multivar', 'TEST', 'test') 

        parser = argparse.ArgumentParser()     
        parser.add_argument('--seed', default=10, type=int, help='seed test')
        parser.add_argument('--result_folder', default='./check', type=str, help='result folder path')
        parser.add_argument('--gpuidx', default=1, type=int, help='gpu index')
        parser.add_argument('--model', default='ConvPool', type=str, help='ConvPool | ConvSwitch')
        parser.add_argument('--pool', default='STP', type=str, help='GTP | STP | DTP | Switch')
        parser.add_argument('--pool_op', default='MAX', type=str, help='MAX | AVG')
        parser.add_argument('--switch_op', default='batch', type=str, help='batch')
        parser.add_argument('--deep_extract', default='FCN', type=str, help='FCN | ResNet')
        parser.add_argument('--proto_num', default=5, type=int, help = '3 | 4 | 5 | 6 | 7')
        parser.add_argument('--cost_type', default='euclidean', type=str, help='cosine | dotprod | euclidean')

        parser.add_argument('--batch_size', default=8, type=int, help='batch size')
        parser.add_argument('--num_epoch', default=300, type=int, help='# of training epochs')
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--decay', default=1e-1, type=float, help='lambda decay')

        parser.add_argument('--gtp', default='MAX', type=str, help='GTP pool type')
        parser.add_argument('--stp', default='MAX', type=str, help='STP pool type')
        parser.add_argument('--dtp', default='MAX', type=str, help='DTP pool type')

        args = parser.parse_args()
        print(i, dataset)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)

        random_seed=args.seed
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        np.random.seed(random_seed) 
        random.seed(random_seed) 
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) 

        perform = main(args=args, train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, num=i, data_type='mul')

        perform_df = pd.DataFrame(perform, index=['loss', 'acc', 'f1macro', 'f1micro', 'f1weight', 'auroc', 'prauc'], columns = [dataset]).T
        report_perform = pd.concat([report_perform, perform_df], axis=0)
        report_perform.to_csv(os.path.join(result_folder, f'{args.model}_{args.deep_extract}_{args.pool}_{args.pool_op}_mul_performance.csv'))

        pd.DataFrame(vars(args), index=np.arange(1)).to_csv(os.path.join(result_folder, 'log.csv'))

                         
        