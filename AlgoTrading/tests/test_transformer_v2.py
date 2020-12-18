import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import plotly.graph_objs as go


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.empty_cache()

# This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all or just the predicted values.
calculate_loss_over_all_values = False

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)
#
# print(out)

input_window  = 100
output_window = 5
batch_size    = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class ModelTrainingAndEvaluation():

    def __init__(self):
        self.df = self.get_data()

    def get_data(self):
        # time_        = np.arange(start=0, stop=400, step=0.1)
        # amplitude   = np.sin(time_) + np.sin(time_*0.05) + np.sin(time_*0.12) * np.random.normal(-0.2, 0.2, len(time_))

        # Work on historical data from the csv file
        historical_data_file = f'historical_data/BTC/2h/ETHBTC_2h'
        dataframe_ = pd.read_csv(historical_data_file, sep='\t')

        # Remove duplicated lines in the historical data if present
        self.df = dataframe_.loc[~dataframe_.index.duplicated(keep='first')]

        # # Compute the log returns
        # self.df.loc[:,'open']  = np.log(self.df.loc[:,'open'].pct_change()+1)
        # self.df.loc[:,'high']  = np.log(self.df.loc[:,'high'].pct_change()+1)
        # self.df.loc[:,'low']   = np.log(self.df.loc[:,'low'].pct_change()+1)
        # self.df.loc[:,'close'] = np.log(self.df.loc[:,'close'].pct_change()+1)

        # Make the triggers values binary : -1/1
        # Doc : df.loc[<row selection>, <column selection>]
        self.df.loc[self.df.buys.isna(),  'buys']  = 0
        self.df.loc[self.df.buys != 0,    'buys']  = -1
        self.df.loc[self.df.sells.isna(), 'sells'] = 0
        self.df.loc[self.df.sells != 0,   'sells'] = +1

        if 'index' in self.df.columns:
            del self.df['index']

        # Set index
        self.df.set_index('time', inplace=True)

        # Reformat datetime index, Binance's data is messy
        self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d %H:%M:%S.%f')

        return self.df

    def scale_data(self):
        # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # amplitude = scaler.fit_transform(X=amplitude.reshape(-1, 1)).reshape(-1)        # Fit to data, then transform it. amplitude est un vecteur ligne
        amplitude = scaler.fit_transform(X=self.df.close.to_numpy().reshape(-1, 1)).reshape(-1)

        samples        = int(len(amplitude)*0.7)
        train_data_raw = amplitude[:samples]
        test_data_raw  = amplitude[samples:]

        return train_data_raw, test_data_raw

    def transform_data(self):

        train_data_raw, test_data_raw = self.scale_data()

        def create_inout_sequences(input_data, tw):
            inout_seq = []
            for i in range(len(input_data)-tw):
                train_seq   = np.append(input_data[i:i+tw][:-output_window], output_window*[0])     # Replace the last values of the chunk by zeros
                train_label = input_data[i:i+tw]
                inout_seq.append((train_seq, train_label))  # inout_seq = list of tuples of chunks of the train data

            # inout_seq = [
            #               ([1, 2, 3, 0, 0], [1, 2, 3, 4, 5]),
            #               ([2, 3, 4, 0, 0], [2, 3, 4, 5, 6]),
            #             ]

            return torch.FloatTensor(inout_seq)     # list to tensor

        # convert our train data into Pytorch tensors
        train_sequence = create_inout_sequences(input_data=train_data_raw, tw=input_window)
        train_sequence = train_sequence[:-output_window]
        test_sequence = create_inout_sequences(input_data=test_data_raw, tw=input_window)
        test_sequence = test_sequence[:-output_window]

        print("Data is ready.")

        return train_sequence.to(device), test_sequence.to(device)

    @staticmethod
    def get_batch(source, i, batch_size_):

        seq_len = min(batch_size_, len(source)-1-i)
        seq     = source[i:i+seq_len]
        # Concatenate a sequence of tensors along a new dimension. dim=0 by default. All tensors need to be of the same size.
        input_values  = torch.stack(tensors=torch.stack(tensors=[item[0] for item in seq]).chunk(input_window,1)) # 1 is feature size
        target_values = torch.stack(tensors=torch.stack(tensors=[item[1] for item in seq]).chunk(input_window,1))
        return input_values, target_values

    def train_model(self):

        model.train()               # Turn on the train mode
        total_loss = 0.
        start_time = time.time()

        for batch, i in enumerate(range(0, len(train_data)-1, batch_size)):
            input_values, target_values = self.get_batch(train_data, i, batch_size)

            optimizer.zero_grad()       # Sets gradients of all model parameters to zero before each backward pass
            output = model(input_values)

            if calculate_loss_over_all_values:
                loss = criterion(output, target_values)
            else:
                loss = criterion(output[-output_window:], target_values[-output_window:])

            loss.backward()     # Back Propagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()    # makes the optimizer iterate over all parameters (tensors) it is supposed to update and use their internally stored grad to update their values.

            # Print to the console, 5 times per epoch
            total_loss += loss.item()
            log_interval = int(len(train_data) / batch_size / 5)
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed  = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.6f} | {:5.2f} ms | '
                      'loss {:5.5f} | ppl {:8.2f}'.format(epoch,
                                                          batch,
                                                          len(train_data) // batch_size,
                                                          scheduler.get_last_lr()[0],
                                                          elapsed * 1000 / log_interval,
                                                          cur_loss,
                                                          math.exp(cur_loss)
                                                          ))
                total_loss = 0
                start_time = time.time()

    def predict_future(self, eval_model, data_source, steps):
        eval_model.eval()

        fig  = go.Figure()


        # Make the predictions
        _ , target_values = self.get_batch(source=data_source, i=0, batch_size_=1)
        with torch.no_grad():   # turn off gradients computation, works in conjunction with .eval()
            for i in range(0, steps):
                input_data = target_values[:input_window]
                predictions = eval_model(input_data)
                target_values = torch.cat((target_values, predictions[-1:]))
        target_values = target_values.cpu().view(-1)

        train_data_, test_data_ = self.scale_data()

        fig.add_trace(go.Scatter(x    = self.df.index[:len(train_data_)],
                                 y    = train_data_,
                                 mode = 'lines',
                                 line = dict(color='black'),
                                 name = 'Train data',
                                 ))
        fig.update_yaxes(title_text  = "Scaled <b> ETH </b> price in BTC")
        fig.update_layout({"yaxis" : {"zeroline" : True},
                           "title" : 'Transformer model'})

        fig.add_trace(go.Scatter(x    = self.df.index[len(train_data_):len(train_data_)+len(test_data_)],
                                 y    = test_data_,
                                 mode = 'lines',
                                 line = dict(color='blue'),
                                 name = 'Test data',
                                 ))

        fig.add_trace(go.Scatter(x    = self.df.index[len(train_data_)+input_window:len(train_data_)+steps],
                                 y    = target_values,
                                 mode = 'lines',
                                 line = dict(color='red'),
                                 name = 'Predictions',
                                 ))
        # Layout for the main graph
        fig.update_layout({
            'margin': {'t': 100, 'b': 20},
            'height': 800,
            'hovermode': 'x',
            'legend_orientation':'h',

            'xaxis'  : {
                'showline'      : True,
                'zeroline'      : False,
                'showgrid'      : False,
                'showticklabels': True,
                'rangeslider'   : {'visible': False},
                'showspikes'    : True,
                'spikemode'     : 'across+toaxis',
                'spikesnap'     : 'cursor',
                'spikethickness': 0.5,
                'color'         : '#a3a7b0',
            },
            'yaxis'  : {
                # 'autorange'      : True,
                # 'rangemode'     : 'normal',
                # 'fixedrange'    : False,
                'showline'      : False,
                'showgrid'      : False,
                'showticklabels': True,
                'ticks'         : '',
                'showspikes'    : True,
                'spikemode'     : 'across+toaxis',
                'spikesnap'     : 'cursor',
                'spikethickness': 0.5,
                'spikecolor'    : '#a3a7b8',
                'color'         : '#a3a7b0',
            },
            'yaxis2' : {
                # "fixedrange"    : True,
                'showline'      : False,
                'zeroline'      : False,
                'showgrid'      : False,
                'showticklabels': True,
                'ticks'         : '',
                # 'color'        : "#a3a7b0",
            },
            'legend' : {
                'font'          : dict(size=15, color='#a3a7b0'),
            },
            'plot_bgcolor'  : '#23272c',
            'paper_bgcolor' : '#23272c',
        })

        fig.show()


        # plt.figure(figsize=(14,10))
        # plt.plot(self.df.index[:len(train_data_)], train_data_, color="black", label='train_data')
        # plt.plot(self.df.index[len(train_data_):len(train_data_)+len(test_data_)], test_data_,  color="blue",  label='test_data')
        # plt.plot(self.df.index[len(train_data_):len(train_data_)+input_window], target_values[:input_window],  color="red",  label='predictions')
        # plt.grid(True, which='both')
        # plt.title('predict_future')
        # plt.legend(loc='upper left')
        # plt.show()


if __name__ == "__main__":

    model     = TransAm().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)              # Choose the optimizer with a learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
    epochs    = 5                                                            # Number of times the data is passed in the model (=number of trainings)

    model_train_eval = ModelTrainingAndEvaluation()
    train_data, test_data = model_train_eval.transform_data()

    # Train the model multiple times
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model_train_eval.train_model()

        model_train_eval.predict_future(eval_model=model, data_source=test_data, steps=200)

        # if epoch % 5 == 0:
        #     val_loss = plot_and_loss(eval_model=model, data_source=test_data, epoch=epoch)
        #     predict_future(eval_model=model, data_source=test_data, steps=200)
        # else:
        #     val_loss = evaluate(eval_model=model, data_source=test_data)

        print('-' * 89)
        print('| end of epoch{:3d} out of{:3d} | Duration: {:5.2f}s'.format(epoch, epochs, (time.time() - epoch_start_time)))
        print('-' * 89)

        scheduler.step()

    # predict_future(eval_model=model, data_source=val_data, steps=200)