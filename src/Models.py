import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

# 5層の全結合層から構成されるDNN
# hidden layerの入出力数は論文に依拠
class FiveLayerNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(FiveLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_len, output_size):
        super(StockPriceLSTM, self).__init__() #nn.Moduleからプロパティメソッドを持ってくるのに必要
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        central_neuron = (hidden_size*seq_len + output_size /2) #真ん中のニューロン数
        self.fc1 = nn.Linear(hidden_size*seq_len, central_neuron)
        self.fc2 = nn.Linear(central_neuron, output_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        x, _ = self.lstm(x)
        x = x.reshape(batch_size, seq_len * self.hidden_size)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return x

#PsitionalEncoding用のクラスも作成

class StockPriceTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layer, num_decoder_layer, dropout, outputsize):
        super(StockPriceTransformer, self).__init__()