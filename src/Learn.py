import os
import sys
import tqdm
import glob
import torch
import Models
import datetime
import numpy as np
sys.path.append("./src")
import DataSets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter



class Learn_LSTM:
    def __init__(self, points_num, frames_num,  pred_future_frame, batch_size, seq_len, modelSaveSpan):
        #諸パラメータの設定
        self.points_num = points_num
        self.frames_num = frames_num
        self.batch_size = batch_size
        self.pred_future_fame = pred_future_frame
        self.seq_len = seq_len
        self.modelSaveSpan = modelSaveSpan
        self.useAugmentation = True
        self.useCustomLossFunction = True
        self.cogWeight = 1.0
        self.relPosWeight = 1.0
        self.d = self.present_time()
        self.ModelFolderPath = "Models/Models" + "_" + str(pred_future_frame) + "_" + str(self.d)
        self.writerPath = 'runs/' + ("Aug_" if self.useAugmentation else "") + ("CF_" if self.useCustomLossFunction else "") + str(pred_future_frame) + "_" + self.d
        print("tensorboard --logdir=" + self.writerPath)
        #解析用のやつ
        self.writer = SummaryWriter(self.writerPath)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def present_time(self):
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        d = now.strftime('%Y%m%d%H%M%S')   
        return d
    
    # def data_loader_multiple(self, folder_paths, DataSetsClass):
    #     #データローダの生成
    #     if DataSetsClass == DataSets.SelectedPointsHumanPoseDataFolder:
    #         train_datasets = []
    #         for folder_path in folder_paths:
    #             train_datasets += [DataSets.SelectedPointsHumanPoseDataFolder(f, frames_num=self.frames_num, points_num=self.points_num) for f in glob.glob(f"{folder_path}/Train")]
    #         train_dataset = ConcatDataset(train_datasets)
    #         self.trainLoader = DataLoader(train_dataset,batch_size=10,shuffle=True)
    #         self.trainLoader.dataset.use_augmentation = self.useAugmentation

    #         eval_datasets = []
    #         for folder_path in folder_paths:
    #             eval_datasets += [DataSets.SelectedPointsHumanPoseDataFolder(f, frames_num=self.frames_num, points_num=self.points_num) for f in glob.glob(f"{folder_path}/Eval/")]
    #         eval_dataset = ConcatDataset(eval_datasets)
    #         self.evalLoader = DataLoader(eval_dataset, batch_size=10, shuffle=True)
    #         self.evalLoader.dataset.use_augmentation = self.useAugmentation
        
    #     # テストデータのロード
    #         test_datasets = []
    #         for folder_path in folder_paths:
    #             test_datasets += [DataSets.SelectedPointsHumanPoseDataFolder(f, frames_num=self.frames_num, points_num=self.points_num) for f in glob.glob(f"{folder_path}/Test/")]
    #         test_dataset = ConcatDataset(test_datasets)
    #         self.testLoader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    #         #サンプルを出力
    #         X,Y = self.trainLoader.dataset.__getitem__(0)
    #         print(X.size(),Y.size())
    #         return (X, Y)
    #     else:
    #         print('the class imported from DataSets.py can not be compatible to this loader.')
        
    
    def data_loader_forLSTM(self, folder_path, DataSetsClass):
        #データローダの生成
        if DataSetsClass == DataSets.StockPriceDataSet_forLSTM:
            self.trainLoader = DataLoader(DataSets.StockPriceDataSet_forLSTM(f"{folder_path}/Train", frames_num=self.frames_num, points_num=self.points_num),batch_size=self.batch_size, shuffle=True)
            self.trainLoader.dataset.use_augmentation = self.useAugmentation
            self.evalLoader = DataLoader(DataSets.StockPriceDataSet_forLSTM(f"{folder_path}/Eval", frames_num=self.frames_num, points_num=self.points_num),batch_size=self.batch_size, shuffle=True)
            self.evalLoader.dataset.use_augmentation = self.useAugmentation
            self.testLoader = DataLoader(DataSets.StockPriceDataSet_forLSTM(f"{folder_path}/Test", frames_num=self.frames_num, points_num=self.points_num),batch_size=self.batch_size, shuffle=True)
            #サンプルを出力
            X,Y = self.trainLoader.dataset.__getitem__(0)
            print(X.size(),Y.size())
            return (X, Y)
        else:
            print('the class imported from DataSets.py can not be compatible to this loader.')
            print(f'the DataSetsClass is {DataSetsClass}')

    
    def network_forLSTM(self, learning_Model, lr, weight_decay):
        # ネットワークの設定
        input_size = self.points_num
        output_size = self.points_num
        self.model = learning_Model.to(self.device)
        self.loadEpoch = 0
        if self.loadEpoch != 0:

            #print("Load Model" + str(self.loadEpoch) + "Epoch")
            self.model = torch.load(os.path.join(self.ModelFolderPath,str(self.loadEpoch) + ".pth"))

        # モデルのグラフをTensorBoardに追加
        example_input = torch.randn(self.batch_size, self.seq_len, input_size).to(self.device)  # ダミーの入力データ
        self.writer.add_graph(self.model, example_input)
        
        # 損失関数と最適化手法の定義
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)

        if(os.path.isdir(self.ModelFolderPath) is False):
            os.makedirs(self.ModelFolderPath)

        return criterion, optimizer
    
    def predict(self, criterion):
        # テストデータの予測
        self.model.eval()
        for x_test,y_test in self.testLoader:
                y_pred = self.model(x_test)
                loss = criterion(y_pred, y_test)
                print("Loss:",loss.item())


    def learn_forLSTM(self, criterion, optimizer, n_epochs):
        # 学習の実行
        n_epochs = n_epochs
        for epoch in tqdm.tqdm(range(self.loadEpoch,n_epochs)):
            self.model.train()
            trainLossSum = 0
            for x_train, y_train in self.trainLoader:
                if x_train.numel() != self.batch_size * self.seq_len * self.points_num:
                    continue
                x_train = x_train.view(self.batch_size, self.seq_len, self.points_num)
                y_pred = self.model(x_train)
                loss = criterion(y_pred, y_train)
                trainLossSum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.writer.add_scalar('training loss',
                            trainLossSum,
                                epoch)
            if (epoch + 1) % self.modelSaveSpan == 0:
                #print("Eval with EvalLoader:")
                self.model.eval()
                lossEachPart = torch.zeros(self.points_num).to(self.device)
                evalLossSum = 0
                evalCount = 0
                for x_test,y_test in self.testLoader:
                    if x_test.numel() != self.batch_size * self.seq_len * self.points_num:
                        continue
                    x_test = x_test.view(self.batch_size, self.seq_len, self.points_num)
                    y_pred = self.model(x_test)
                    loss = criterion(y_pred, y_test)
                    diff = torch.abs(y_pred - y_test)
                    diffSum = torch.sum(diff,0)
                    lossEachPart += diffSum
                    evalLossSum += loss.item()
                    evalCount += 1
                #print("Avg Eval Loss:" + str(evalLossSum/evalCount))
                modelPath = os.path.join(self.ModelFolderPath,str(epoch+1) + ".pth")
                torch.save(self.model,modelPath)
                #print("saving model at " + modelPath)
                self.writer.add_scalar('evalLoss',
                                evalLossSum,
                                    epoch)
                # self.writer.add_scalars("Diff_chest",{"X":lossEachPart[0],"Z":lossEachPart[1]},epoch)
                # self.writer.add_scalars("Diff_AnkleLeft",{"X":lossEachPart[2],"Z":lossEachPart[3]},epoch)
                # self.writer.add_scalars("Diff_AnkleRight",{"X":lossEachPart[4],"Z":lossEachPart[5]},epoch)

                #試しにこの上の二つを消してみました
                for i in range(6,len(lossEachPart)):
                    self.writer.add_scalar("Diff_" + str(i),lossEachPart[i],epoch)
        self.writer.close()
    

class Learn_transformer:
    def __init__(self, points_num, frames_num, pred_future_frame, batch_size, seq_len, modelSaveSpan):
        #諸パラメータの設定
        self.points_num = points_num
        self.frames_num = frames_num
        self.batch_size = batch_size
        self.pred_future_fame = pred_future_frame
        self.seq_len = seq_len
        self.modelSaveSpan = modelSaveSpan
        self.useAugmentation = True
        self.useCustomLossFunction = True
        self.cogWeight = 1.0
        self.relPosWeight = 1.0
        self.d = self.present_time()
        self.ModelFolderPath = "Models/Models" + "_" + str(pred_future_frame) + "_" + str(self.d)
        self.writerPath = 'runs/' + ("Aug_" if self.useAugmentation else "") + ("CF_" if self.useCustomLossFunction else "") + str(pred_future_frame) + "_" + self.d
        print("tensorboard --logdir=" + self.writerPath)
        #解析用のやつ
        self.writer = SummaryWriter(self.writerPath)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def present_time(self):
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        d = now.strftime('%Y%m%d%H%M%S')   
        return d

    def data_loader_fortransformer(self, folder_path, DataSetsClass):
        #データローダの生成
        if DataSetsClass == DataSets.StockPriceDataSet_fortransformer:
            self.trainLoader = DataLoader(DataSets.StockPriceDataSet_fortransformer(f"{folder_path}/Train", frames_num=self.frames_num, points_num=self.points_num),batch_size=self.batch_size, shuffle=True)
            self.trainLoader.dataset.use_augmentation = self.useAugmentation
            self.evalLoader = DataLoader(DataSets.StockPriceDataSet_fortransformer(f"{folder_path}/Eval", frames_num=self.frames_num, points_num=self.points_num),batch_size=self.batch_size, shuffle=True)
            self.evalLoader.dataset.use_augmentation = self.useAugmentation
            self.testLoader = DataLoader(DataSets.StockPriceDataSet_fortransformer(f"{folder_path}/Test", frames_num=self.frames_num, points_num=self.points_num),batch_size=self.batch_size, shuffle=True)
            #サンプルを出力
            X,Y = self.trainLoader.dataset.__getitem__(0)
            print(X.size(),Y.size())
            return (X, Y)
        else:
            print('the class imported from DataSets.py can not be compatible to this loader.')
            print(f'the DataSetsClass is {DataSetsClass}')
    
    def network_fortraansformer(self, learning_Model, lr, weight_decay):
        # ネットワークの設定
        input_size = self.points_num
        output_size = self.points_num
        self.model = learning_Model.to(self.device)
        self.loadEpoch = 0
        if self.loadEpoch != 0:

            #print("Load Model" + str(self.loadEpoch) + "Epoch")
            self.model = torch.load(os.path.join(self.ModelFolderPath,str(self.loadEpoch) + ".pth"))

        # モデルのグラフをTensorBoardに追加
        example_input = torch.randn(self.batch_size, self.seq_len, input_size).to(self.device)  # ダミーの入力データ
        self.writer.add_graph(self.model, example_input)
        
        # 損失関数と最適化手法の定義
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)

        if(os.path.isdir(self.ModelFolderPath) is False):
            os.makedirs(self.ModelFolderPath)

        return criterion, optimizer
