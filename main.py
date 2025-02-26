import os
import time
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd
from utils.utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='PeMS07')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--window_size', type=int, default=12) 
parser.add_argument('--horizon', type=int, default=3)  
parser.add_argument('--train_length', type=float, default=7)  
parser.add_argument('--valid_length', type=float, default=2)  
parser.add_argument('--test_length', type=float, default=1)  

parser.add_argument('--optimizer', type=str, default='RMSProp') 
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--early_stop_step', type=int, default=10)
parser.add_argument('--exponential_decay_step', type=int, default=15) 
parser.add_argument('--validate_freq', type=int, default=1)

parser.add_argument('--epoch', type=int, default=50)  
parser.add_argument('--multi_layer', type=int, default=5) 
parser.add_argument('--batch_size', type=int, default=50)  
parser.add_argument('--norm_method', type=str, default='z_score')  
parser.add_argument('--dropout_rate', type=float, default=0.4)  
parser.add_argument('--leakyrelu_rate', type=float, default=0.02)  
parser.add_argument('--Fouropt', type=str, default='FB')  
parser.add_argument('--attention_set', type=str, default='linear')  
parser.add_argument('--non_linear', type=str, default='linear') 
parser.add_argument('--modes', type=int, default=5, help='fourier component sampling')
parser.add_argument('--activation', type=str, default='softmax')  
parser.add_argument('--gconv', type=str, default='gegen')  
parser.add_argument('--layers', type=int, default=2, help='model layers')
parser.add_argument('--order', type=int, default=4, help='order')
parser.add_argument('--coe_a', type=float, default=1.2) 
parser.add_argument('--coe_b', type=float, default=1)  

args = parser.parse_args()
print(f'Training configs: {args}')
set_random_seed(args.seed)
data_file = os.path.join('dataset', args.dataset + '.csv')
log_time = str(time.time())
result_train_file = os.path.join('output', args.dataset, log_time, 'train')
result_test_file = os.path.join('output', args.dataset, log_time, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
data = pd.read_csv(data_file).values

train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]

if __name__ == '__main__':
    Mae = 0
    Rmse = 0
    for r in range(args.run):
        set_random_seed(r)
        if args.train:
            try:
                before_train = datetime.now().timestamp()
                _, normalize_statistic = train(train_data, valid_data, r, args, result_train_file)
                after_train = datetime.now().timestamp()
                print(f'Training took {(after_train - before_train) / 60} minutes')
            except KeyboardInterrupt:
                print('-' * 99)
                print('Exiting from training early')
        if args.evaluate:
            before_evaluation = datetime.now().timestamp()
            mae, rmse = test(test_data, args, result_train_file, result_test_file)
            Mae += mae
            Rmse += rmse
            after_evaluation = datetime.now().timestamp()
            print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
            print('done')

    Mae_M = Mae / args.run
    Rmse_M = Rmse / args.run
    print('Performance on test set: MAE_MEAN: {:5.4f} | RMSE_MEAN: {:5.4f} '.format(Mae_M, Rmse_M))
