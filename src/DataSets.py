import os
import random
import numpy
import tqdm
import torch
from torch.utils.data import Dataset


# データ(torch.tensor形式)が含まれるフォルダを指定すると、それを読み込んでデータセットを作ってくれる自作Datasetクラス
# データフォルダの構成は
# (datadir)→(X)→（入力群）
#          →(Y)→(正解データ群)
# となっている必要がある
class HumanPoseDataFolder(Dataset):
    def __init__(self,dataDir):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataDir = dataDir
        self.dirX = os.path.join(dataDir, "X")
        self.dirY = os.path.join(dataDir, "Y")
        filesX = os.listdir(self.dirX)
        filesY = os.listdir(self.dirY)
        self.dataPathListX = []
        self.dataPathListY = []
        self.dataX = []
        self.dataY = []
        self.dirX_in = torch.zeros(6 * 15)
        self.dirY_in = torch.zeros(6 * 15)
        self.dirZ_in = torch.zeros(6 * 15)
        self.dirX_out = torch.zeros(6)
        self.dirY_out = torch.zeros(6)
        self.dirZ_out = torch.zeros(6)
        for i in range(15):
            self.dirX_out[0] = 1
            self.dirY_out[1] = 1
            self.dirZ_out[2] = 1
            self.dirX_in[i * 6 + 0] = 1
            self.dirY_in[i * 6 + 2] = 1
            self.dirZ_in[i * 6 + 4] = 1
        self.dirX_in = self.dirX_in.to(self.device)
        self.dirY_in = self.dirY_in.to(self.device)
        self.dirZ_in = self.dirZ_in.to(self.device)
        self.dirX_out = self.dirX_out.to(self.device)
        self.dirY_out = self.dirY_out.to(self.device)
        self.dirZ_out = self.dirZ_out.to(self.device)
        self.use_augmentation = False
        self.rng = numpy.random.default_rng()
        print(self.dirX_in)
        print(self.dirY_in)
        print(self.dirZ_in)
        print(self.dirX_out)
        print(self.dirY_out)
        print(self.dirZ_out)
        # process X
        for file in tqdm.tqdm(filesX):
            dataPath = os.path.join(self.dirX, file)
            self.dataX.append(torch.load(dataPath).to(self.device))
            self.dataPathListX.append(dataPath)
        # process Y
        for file in tqdm.tqdm(filesY):
            dataPath = os.path.join(self.dirY, file)
            self.dataY.append(torch.load(dataPath).to(self.device))
            self.dataPathListY.append(dataPath)
    def __getitem__(self, index):
        tensorX = self.dataX[index]
        tensorY = self.dataY[index]
        if self.use_augmentation:
            offsetX = self.rng.uniform(-3, 3)
            offsetY = self.rng.uniform(-0.5, 0.5)
            offsetZ = self.rng.uniform(-3, 3)
            offsetTensorX = torch.zeros(6 * 15)
            offsetTensorX = self.dirX_in * offsetX + self.dirY_in * offsetY + self.dirZ_in * offsetZ
            offsetTensorX = offsetTensorX.to(self.device)
            offsetTensorY = torch.zeros(6)
            offsetTensorY = self.dirX_out * offsetX + self.dirY_out * offsetY + self.dirZ_out * offsetZ
            offsetTensorY = offsetTensorY.to(self.device)
            tensorX = tensorX + offsetTensorX
            tensorY = tensorY + offsetTensorY
        return tensorX, tensorY
    def __len__(self):
        return len(self.dataPathListX)



#x, z座標のみを学習させるときに使うもの
class SelectedPointsHumanPoseDataFolder(Dataset):
    def __init__(self,dataDir, points_num, frames_num):
        self.points_num = points_num
        self.frames_num = frames_num
        self.dataDir = dataDir
        self.dirX = os.path.join(dataDir, "X")
        self.dirY = os.path.join(dataDir, "Y")
        filesX = os.listdir(self.dirX)
        filesY = os.listdir(self.dirY)
        self.dataPathListX = []
        self.dataPathListY = []
        self.dataX = []
        self.dataY = []

        #offset用の軸操作？
        self.dirX_in = torch.zeros(self.points_num* self.frames_num)
        self.dirZ_in = torch.zeros(self.points_num* self.frames_num)
        self.dirX_out = torch.zeros(self.points_num)
        self.dirZ_out = torch.zeros(self.points_num)


        #各軸に対してフラグのようなもの立ててる
        for i in range(self.frames_num):
            self.dirX_in[i * self.points_num + 0] = 1
            self.dirZ_in[i * self.points_num + 1] = 1
            
        #to cuda
        self.dirX_in = self.dirX_in.to(self.device)
        self.dirZ_in = self.dirZ_in.to(self.device)
        self.dirX_out = self.dirX_out.to(self.device)
        self.dirZ_out = self.dirZ_out.to(self.device)

        self.rng = numpy.random.default_rng()

        #process X
        for file in tqdm.tqdm(filesX):
            dataPath = os.path.join(self.dirX, file)
            self.dataX.append(torch.load(dataPath).to(self.device))
            self.dataPathListX.append(dataPath)
        # process Y
        for file in tqdm.tqdm(filesY):
            dataPath = os.path.join(self.dirY, file)
            self.dataY.append(torch.load(dataPath).to(self.device))
            self.dataPathListY.append(dataPath)

    def __getitem__(self, index):
        tensorX = self.dataX[index]
        tensorY = self.dataY[index]

        #オフセット用ランダムな数を生成
        offsetX = self.rng.uniform(-5, 5)
        offsetZ = self.rng.uniform(-5, 5)

        #オフセット生成
        offsetTensorX = torch.zeros(self.points_num * self.frames_num)
        offsetTensorY = torch.zeros(self.points_num)
        offsetTensorX = self.dirX_in * offsetX+ self.dirZ_in * offsetZ
        offsetTensorY = self.dirX_out * offsetX + self.dirZ_out * offsetZ
        
        #オフセット追加
        tensorX = tensorX + offsetTensorX
        tensorY = tensorY + offsetTensorY
        return tensorX, tensorY

    def __len__(self):
        return len(self.dataPathListX)
    


class SelectedPointsHumanPoseDataFolder_forRNN(Dataset):
    def __init__(self, dataDir, points_num, frames_num):
        self.points_num = points_num
        self.frames_num = frames_num
        self.dataDir = dataDir
        self.dirX = os.path.join(dataDir, "X")
        self.dirY = os.path.join(dataDir, "Y")
        filesX = os.listdir(self.dirX)
        filesY = os.listdir(self.dirY)
        self.dataPathListX = []
        self.dataPathListY = []
        self.dataX = []
        self.dataY = []

        # 各軸に対してフラグを設定
        self.dirX_in = torch.zeros(self.frames_num, self.points_num)
        self.dirZ_in = torch.zeros(self.frames_num, self.points_num)
        self.dirX_out = torch.zeros(self.points_num)
        self.dirZ_out = torch.zeros(self.points_num)

        for i in range(self.frames_num):
            self.dirX_in[i, 0] = 1
            self.dirZ_in[i, 1] = 1

        #to cuda
        self.dirX_in = self.dirX_in.to(self.device)
        self.dirZ_in = self.dirZ_in.to(self.device)
        self.dirX_out = self.dirX_out.to(self.device)
        self.dirZ_out = self.dirZ_out.to(self.device)

        self.rng = numpy.random.default_rng()

        # process X
        for file in tqdm.tqdm(filesX):
            dataPath = os.path.join(self.dirX, file)
            self.dataX.append(torch.load(dataPath).to(self.device))
            self.dataPathListX.append(dataPath)
        # process Y
        for file in tqdm.tqdm(filesY):
            dataPath = os.path.join(self.dirY, file)
            self.dataY.append(torch.load(dataPath).to(self.device))
            self.dataPathListY.append(dataPath)

    def __getitem__(self, index):
        tensorX = self.dataX[index]
        tensorY = self.dataY[index]

        # オフセット用ランダムな数を生成
        offsetX = self.rng.uniform(-5, 5)
        offsetZ = self.rng.uniform(-5, 5)

        # オフセット生成
        offsetTensorX = torch.zeros(self.frames_num, self.points_num).to(self.device)
        offsetTensorY = torch.zeros(self.points_num).to(self.device)
        offsetTensorX = self.dirX_in * offsetX + self.dirZ_in * offsetZ
        offsetTensorY = self.dirX_out * offsetX + self.dirZ_out * offsetZ
        
        # オフセット追加
        tensorX = tensorX + offsetTensorX
        tensorY = tensorY + offsetTensorY
        
        # tensorXを(frames_num, points_num)の形状に変形 (必要ないので削除)
        # tensorX = tensorX.view(self.frames_num, self.points_num)
        
        return tensorX, tensorY

    def __len__(self):
        return len(self.dataPathListX)
    

class StockPriceDataSet_forLSTM(Dataset):
    def __init__(self, dataDir, points_num, frames_num):
        self.points_num = points_num
        self.frames_num = frames_num
        self.dataDir = dataDir
        self.dirX = os.path.join(dataDir, "X")
        self.dirY = os.path.join(dataDir, "Y")
        filesX = os.listdir(self.dirX)
        filesY = os.listdir(self.dirY)
        self.dataPathListX = []
        self.dataPathListY = []
        self.dataX = []
        self.dataY = []

        # 各軸に対してフラグを設定
        self.dirX_in = torch.zeros(self.frames_num, self.points_num)
        self.dirZ_in = torch.zeros(self.frames_num, self.points_num)
        self.dirX_out = torch.zeros(self.points_num)
        self.dirZ_out = torch.zeros(self.points_num)

        for i in range(self.frames_num):
            self.dirX_in[i, 0] = 1
            self.dirZ_in[i, 1] = 1

        # to cuda
        self.dirX_in = self.dirX_in.to(self.device)
        self.dirZ_in = self.dirZ_in.to(self.device)
        self.dirX_out = self.dirX_out.to(self.device)
        self.dirZ_out = self.dirZ_out.to(self.device)

        self.rng = numpy.random.default_rng()

        # process X
        for file in tqdm.tqdm(filesX):
            dataPath = os.path.join(self.dirX, file)
            self.dataX.append(torch.load(dataPath).to(self.device))
            self.dataPathListX.append(dataPath)
        # process Y
        for file in tqdm.tqdm(filesY):
            dataPath = os.path.join(self.dirY, file)
            self.dataY.append(torch.load(dataPath).to(self.device))
            self.dataPathListY.append(dataPath)

    def __getitem__(self, index):
        tensorX = self.dataX[index]
        tensorY = self.dataY[index]

        # オフセット用ランダムな数を生成
        offsetX = self.rng.uniform(-5, 5)
        offsetZ = self.rng.uniform(-5, 5)

        # オフセット生成
        offsetTensorX = torch.zeros(self.frames_num, self.points_num).to(self.device)
        offsetTensorY = torch.zeros(self.points_num).to(self.device)
        offsetTensorX = self.dirX_in * offsetX + self.dirZ_in * offsetZ
        offsetTensorY = self.dirX_out * offsetX + self.dirZ_out * offsetZ
        
        # オフセット追加
        tensorX = tensorX + offsetTensorX
        tensorY = tensorY + offsetTensorY
        
        return tensorX, tensorY

    def __len__(self):
        return len(self.dataPathListX)