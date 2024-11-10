import math
import torch
import numpy as np
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
        central_neuron = int((hidden_size*seq_len + output_size)/2) #真ん中のニューロン数
        self.fc1 = nn.Linear(hidden_size*seq_len, central_neuron)
        self.fc2 = nn.Linear(central_neuron, output_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        x, _ = self.lstm(x)
        x = x.reshape(batch_size, seq_len * self.hidden_size)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return x


# Functions for positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len, d_model)
        position=torch.arange(0, max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0).transpose(0,1)
        self.register_buffer("pe",pe)
    
    def forward(self,x):
        return self.dropout(x+self.pe[:np.shape(x)[0],:])



class StockPriceTransformer(nn.Module):
    def __init__(self, feature_size, num_layers, dropout):
        super(StockPriceTransformer, self).__init__()
        self.model_type='Transformer'
        self.src_mask=None
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_encoder=PositionalEncoding(d_model=feature_size)
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=feature_size,nhead=10,dropout=dropout)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=num_layers)
        self.decoder=nn.Linear(feature_size,1)
    
    def init_weights(self):
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform(-0.1,0.1)

    def _generate_square_subsequent_mask(self,sz):
        mask=(torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
        mask=mask.float().masked_fill(mask==0,float('-inf')).masked_fill(mask==1,float(0.0))
        return mask

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0)!=len(src):
            device=self.device
            mask=self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask=mask
        src=self.pos_encoder(src)
        output=self.transformer_encoder(src,self.src_mask)
        output=self.decoder(output)
        return output