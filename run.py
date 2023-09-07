import argparse
import torch
from exp.exp_main import Exp_Main
import random
import pandas as pd
import numpy as np
import pickle
import feather

# df = feather.read_dataframe('./factors.feather')
# df['ret_next'] = df.groupby('id',group_keys =False)['ret_c2c'].shift(-1)
# factors = [i for i in df.columns if i.endswith('_factor')]
# factors.remove('industry_factor')
# for column in factors:
#     mask = ((df[column] < 1e-8)&(df[column]>0))  # 创建布尔条件掩码
#     df[column] = np.where(mask, 0.00001, df[column])  # 将小于1e-8的数字替换为0.00001

#     mask = ((df[column] > -1e-8)&(df[column]<0))  # 创建布尔条件掩码
#     df[column] = np.where(mask, -0.00001, df[column])  # 将大于-1e-8的数字替换为-0.00001
#     mask = (df[column].abs() > 1e40)
#     df[column] = np.where(mask, 0.00001, df[column])
# macros = pd.read_csv('./Macro_factor.csv')
# df['date'] = pd.to_datetime(df['date'])
# macros['date'] = pd.to_datetime(macros['date'])
# df = pd.merge(df,macros,on = 'date',how = 'left')
# df = df[df['date']<='2023-03-31']
# df.replace([np.inf, -np.inf], 0.00001, inplace=True)
# df.fillna(0, inplace=True)
# grouped = df.groupby('id')
# stock_dict = {}
# id_mapping = {}
# i = 1
# for stock, group in grouped:
#     id_mapping[i] = stock
#     stock_dict[i] = group
#     del stock_dict[i]['id']
#     del stock_dict[i]['industry_factor']
#     i+=1
# with open('id_mapping.pkl', 'wb') as f:
#     pickle.dump(id_mapping, f)

df = feather.read_dataframe('../daily.feather')
df = df[df['date']>='2000-01-01']
df.replace([np.inf, -np.inf], 0.00001, inplace=True)
df.fillna(0, inplace=True)

df['date'] = pd.to_datetime(df['date'])
grouped = df.groupby('id')
stock_dict = {}

for stock, group in grouped:
    stock_dict[stock] = group
    del stock_dict[stock]['id']
    del stock_dict[stock]['Markettype']

parser = argparse.ArgumentParser(description='Non-stationary Transformers for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Transformer',
                    help='model name, options: [ns_Transformer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='ret', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='fixed',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2023, help='random seed')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        torch.cuda.set_device(args.gpu)

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        for stock, data in stock_dict.items():
            if stock>=1980 and stock<=5000:
                if data['date'].min() <= pd.to_datetime('2018-10-31') and data['date'].max()>=pd.to_datetime('2021-05-31'):
                    try:
                        print("现在是stock",stock)
                        exp = Exp(args,stock,data)
                        print('>>>>>>>training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.train(setting)
                        del exp
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error message: {str(e)}")
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    pred = []
    for stock, data in stock_dict.items():
        if stock>=2000 and stock<=2300:
            if data['date'].min() <= pd.to_datetime('2018-10-31') and data['date'].max()>=pd.to_datetime('2021-05-31'):
                try:
                    print("现在是stock",stock)
                    exp = Exp(args,stock,data)
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    tempt = exp.test(setting, test=1)
                    pred.append(tempt)
                    if args.do_predict:
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.predict(setting, True)
                    del exp
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error message: {str(e)}")

    try:
        pred = np.concatenate(pred, axis=0)
        pred = pred.reshape(-1, pred.shape[-1])
        pred = pd.DataFrame(pred, columns=['pred', 'year', 'month', 'day','id'])
        feather.write_dataframe(pred, f'./pred/pred_2000_to_2300.feather')
    except Exception as e:
        print(f"Error message: {str(e)}")
