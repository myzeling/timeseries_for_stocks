from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, Informer, Autoformer
from ns_models import ns_Transformer, ns_Informer, ns_Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

import feather
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args, stock ,data):
        super(Exp_Main, self).__init__(args)
        self.data = data
        self.stock = stock
    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'ns_Transformer': ns_Transformer,
            'ns_Informer': ns_Informer,
            'ns_Autoformer': ns_Autoformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set = data_provider(self.args, flag, self.data, self.stock)
        
        return data_set

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        print("我们开始验证")
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                if torch.isnan(loss):
                    print("这是这个狗东西的batch出问题了",batch_y_mark)
                else:
                    total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join('./checkpoints/', str(self.stock), setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        all_data = []
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join('./checkpoints/', str(self.stock), setting)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = f'./test_results/{str(self.stock)}/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                date = batch_y_mark[:, -self.args.pred_len:, :].detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                combined = np.concatenate([pred, date], axis=-1)
                stock_column = np.full(combined.shape[:-1] + (1,), str(self.stock), dtype=object)
                combined = np.concatenate([combined,stock_column],axis= -1)
                all_data.append(combined)

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pdf = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pdf, os.path.join(folder_path, str(i) + '.pdf'))
        all_data = np.concatenate(all_data, axis=0)
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = f'./results/{str(self.stock)}/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return all_data

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        path = os.path.join(self.args.checkpoints, setting)
        if load:
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

       

        return

# class Exp_train(Exp_Basic):
#     def __init__(self, args, data_dict_real):
#         super(Exp_train, self).__init__(args)
#         self.data_dict = data_dict_real
#         self.stock_left = None
#     def _build_model(self):
#         model_dict = {
#             'Transformer': Transformer,
#             'Informer': Informer,
#             'Autoformer': Autoformer,
#             'ns_Transformer': ns_Transformer,
#             'ns_Informer': ns_Informer,
#             'ns_Autoformer': ns_Autoformer,
#         }
#         model = model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag, stock):
#         data_set = data_provider(self.args, flag, self.data_dict, stock)
        
#         return data_set

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion

#     def vali(self, vali_data, vali_loader, criterion):
#         print("我们开始验证")
#         total_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

#                 pred = outputs.detach().cpu()
#                 true = batch_y.detach().cpu()
#                 loss = criterion(pred, true)
#                 if torch.isnan(loss):
#                     print("这是这个狗东西的batch出问题了",batch_y_mark)
#                 else:
#                     total_loss.append(loss)
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss

#     def train(self, setting):
#         train_data = []
#         test_data = []
#         stock_left = []
#         for stock in self.data_dict.keys():
#             print("stock",stock)
#             if self.data_dict[stock]['date'].max() < pd.to_datetime('2020-01-31'):
#                 try:
#                     train_data_set = self._get_data('train', stock)
#                     train_data.append(train_data_set)
#                 except Exception as e:
#                     print(f"E{stock}在小于2020时候进入训练集失败，该循环回合跳过")
#                     print(f"Error message: {str(e)}")
#                     stock_left.append(stock)
#                     continue
#             elif self.data_dict[stock]['date'].min() >= pd.to_datetime('2020-01-31'):
#                 try:
#                     test_data_set = self._get_data('test', stock)
#                 except Exception as e:
#                     print(f"E{stock}在大于2020时候进入测试集失败")
#                     print(f"Error message: {str(e)}")
#                     stock_left.append(stock)
#                     continue
#             else:
#                 try:
#                     train_data_set = self._get_data('train', stock)
#                     train_data.append(train_data_set)
#                 except Exception as e:
#                     print(f"E{stock}是跨度股票，但是进入训练集失败了，可能是数量不够")
#                     print(f"Error message: {str(e)}")
#                     try:
#                         test_data_set = self._get_data('test', stock)
#                         test_data.append(test_data_set)
#                     except Exception as e:
#                         print(f"E{stock}是跨度股票，进入训练集失败了，而且进入测试集合也失败了，是彻底的失败股票")
#                         print(f"Error message: {str(e)}")
#                         stock_left.append(stock)
#                         continue
#         train_data = ConcatDataset(train_data)
#         test_data = ConcatDataset(test_data)
#         train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, drop_last=True，num_workers = 10)
#         test_loader = DataLoader(test_data, batch_size=512, shuffle=False, drop_last=False, num_workers = 10)

#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()

#         for epoch in range(self.args.train_epochs):
#             print("开始训练")
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)

#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                     f_dim = -1 if self.args.features == 'MS' else 0
#                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                     loss = criterion(outputs, batch_y)
#                     train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()
            
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             test_loss = self.vali(test_data, test_loader, criterion)
#             print("损失好像计算成功了")
#             print("Epoch: {0}, Steps: {1}|Train loss:{2:.7f} | Test Loss: {3:.7f}".format(
#                 epoch + 1, train_steps, train_loss, test_loss))
#             early_stopping(train_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#             adjust_learning_rate(model_optim, epoch + 1, self.args)

#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model

#     def test(self, setting, test=0):
#         test_data = []
#         all_data = []
#         for stock in self.data_dict.keys():
#             print("stock",stock)
#             if self.data_dict[stock]['date'].min() >= pd.to_datetime('2020-01-31'):
#                 try:
#                     test_data_set = self._get_data('test', stock)
#                 except Exception as e:
#                     print(f"E{stock}在大于2020时候进入测试集失败")
#                     print(f"Error message: {str(e)}")
#                     continue
#             else:
#                 try:
#                     test_data_set = self._get_data('test', stock)
#                     test_data.append(test_data_set)
#                 except Exception as e:
#                     print(f"E{stock}是跨度股票，测试集合也失败了，非常失败")
#                     print(f"Error message: {str(e)}")
#                     continue
            
#         test_data = ConcatDataset(test_data)
#         test_loader = DataLoader(test_data, batch_size = 256, shuffle = False, drop_last = False,num_workers = 10)
        
#         path = os.path.join('./checkpoints/', setting)
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

#         preds = []
#         trues = []
#         folder_path = f'./test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         dataframe_path = f'./{self.args.model}_dataframe/'
#         if not os.path.exists(dataframe_path):
#             os.makedirs(dataframe_path)
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc="Testing")):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()
#                 date = batch_y_mark[:, -self.args.pred_len:, :].detach().cpu().numpy()
#                 pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
#                 true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

#                 combined = np.concatenate([pred, date], axis=-1)
#                 all_data.append(combined)

#                 preds.append(pred)
#                 trues.append(true)
#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pde = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pde, os.path.join(folder_path, str(i) + '.pdf'))
#                 print(f"循环{i+1}已经完成了")
#         print("所有的循环都完成了，现在我们进入到完成dataframe环节")
#         all_data = np.concatenate(all_data, axis=0)
#         all_data = all_data.reshape(-1, all_data.shape[-1])
#         all_data = pd.DataFrame(all_data, columns=['pred', 'stock', 'year', 'month', 'day'])
#         print("进入到重置date")
#         # all_data['date'] = all_data[['year', 'month', 'day']].astype(int).astype(str).agg('-'.join, axis=1)
#         # all_data.drop(['year', 'month', 'day'], axis=1, inplace=True)
#         feather.write_dataframe(all_data,dataframe_path+'all_data.feather')
#         print("data保存为feather已经成功了")
#         preds = np.array(preds)
#         trues = np.array(trues)
#         print('test shape:', preds[0].shape, trues[0].shape)
#         preds = preds[0].reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
#         trues = trues[0].reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
#         print('test shape:', preds.shape, trues.shape)
#         # result save
#         folder_path = f'./results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('stock_left:{}'.format(self.stock_left))
#         f.write('\n')
#         f.write('\n')
#         f.close()
        
#         np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + 'pred.npy', preds)
#         np.save(folder_path + 'true.npy', trues)

#         return

#     def predict(self, setting, load=False):
#         pred_data, pred_loader = self._get_data(flag='pred')
#         path = os.path.join(self.args.checkpoints, setting)
#         if load:
#             best_model_path = os.path.join(path, 'checkpoint.pth')
#             self.model.load_state_dict(torch.load(best_model_path))

#         preds = []

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 pred = outputs.detach().cpu().numpy()  # .squeeze()
#                 preds.append(pred)

#         preds = np.array(preds)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         np.save(folder_path + 'real_prediction.npy', preds)

#         return
