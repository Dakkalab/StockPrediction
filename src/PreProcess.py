import os
import csv
import torch.utils.data
import tqdm
import pandas as pd
import random
import math
import numpy as np

class Preprocess:
    def __init__(self, train_rate, eval_rate, test_rate):
        self.train_rate = train_rate
        self.eval_rate = eval_rate
        self.test_rate = test_rate

        self.dataCount = {"Train":0,"Eval":0,"Test":0}

        self.dataCount["Train"] = 0
        self.dataCount["Eval"] = 0
        self.dataCount["Test"] = 0

    # ターゲットカラムのリスト
    def get_column_indices(self, header, target_cols):
        """
        ヘッダーとターゲットカラムのリストを基に、ターゲットカラムのインデックスを取得する
        """
        return [header.index(col) for col in target_cols]
    
    def confirm_length(self, output_path):
        file_path = f'{output_path}/Train/X/1.pt'
        data = torch.load(file_path)
        return len(data)


    def preprocess_points(self, folder_path, output_path, predictionFutureFrame, target_cols):
        # フォルダ内のすべてのファイルとディレクトリを取得

        files = os.listdir(folder_path)
        target_files = []
        header = None


        for fileIndex in tqdm.tqdm(range(len(files))):
            # ファイルの絶対パスを取得
            file_path = os.path.join(folder_path, files[fileIndex])
            with open(file_path, 'r') as csvfile:
                # CSVファイルを読み込み
                reader = csv.reader(csvfile)
                if header is None:
                    header = next(reader)
                target_files.append(file_path)

        #カラム指定がある場合
        if target_cols:
            column_indices = self.get_column_indices(header, target_cols)
        
        if os.path.isdir(output_path) is False:
            for subPathA in ["Train","Eval","Test"]:
                for subPathB in ["X","Y"]:
                    os.makedirs(os.path.join(output_path,subPathA,subPathB))

        # ファイルのリストを作成
        fileCount = len(target_files)
        trainCount = int(fileCount * self.train_rate)
        evalCount = int(fileCount * self.eval_rate)
        for fileIndex in tqdm.tqdm(range(len(target_files))):
            # ファイルの絶対パスを取得
            file_path = target_files[fileIndex]
            #print(file_path)
            #データのモード(Train/Eval/Test)を取得
            #確率的に決定しないと偏る
            with open(file_path, 'r') as csvfile:
                # CSVファイルを読み込み
                reader = csv.reader(csvfile)
                next(reader) 

                # CSVファイルの内容を二次元配列に変換
                data = [row for row in reader]

                #カラム指定がある場合
                if target_cols:
                    data = [[row[idx] for idx in column_indices] for row in data]

                
                for frameIndex in range(16,len(data)):
                    if frameIndex + predictionFutureFrame < len(data):
                        #過去15フレームをとってきてデータとする
                        strX = data[frameIndex - 15 : frameIndex]
                        #print(strX[0])
                        #predictionFutureFrameフレーム後の未来を持ってきて正解データとする
                        strY = data[frameIndex + predictionFutureFrame]

                        X = [list(map(float, row)) for row in strX]
                        flattenX = [elem for sublist in X for elem in sublist]
                        r = random.random()
                        if r < self.train_rate:
                            mode = "Train"
                        elif r < self.train_rate + self.eval_rate:
                            mode = "Eval"
                        else:
                            mode = "Test"
                        Y = list(map(float, strY))
                        torch.save(torch.tensor(flattenX),os.path.join(output_path,mode,"X",str(self.dataCount[mode])+".pt"))
                        torch.save(torch.tensor(Y),os.path.join(output_path,mode,"Y",str(self.dataCount[mode])+".pt"))
                        self.dataCount[mode] += 1

