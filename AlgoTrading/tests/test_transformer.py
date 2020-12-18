import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


torch.manual_seed(0)
np.random.seed(0)

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
batch_size    = 10 # batch size
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

def get_data():
    time        = np.arange(0, 400, 0.1)
    amplitude   = np.sin(time) + np.sin(time*0.05) + np.sin(time*0.12) * np.random.normal(-0.2, 0.2, len(time))

    # # Work on historical data from the csv file
    # historical_data_file = f'historical_data/BTC/1h/ETHBTC_1h'
    # dataframe_ = pd.read_csv(historical_data_file, sep='\t')
    #
    # # Remove duplicated lines in the historical data if present
    # dataframe = dataframe_.loc[~dataframe_.index.duplicated(keep='first')]
    #
    # # # Compute the log returns
    # # dataframe.loc[:,'open']  = np.log(dataframe.loc[:,'open'].pct_change()+1)
    # # dataframe.loc[:,'high']  = np.log(dataframe.loc[:,'high'].pct_change()+1)
    # # dataframe.loc[:,'low']   = np.log(dataframe.loc[:,'low'].pct_change()+1)
    # # dataframe.loc[:,'close'] = np.log(dataframe.loc[:,'close'].pct_change()+1)
    #
    # # Make the triggers values binary : -1/1
    # # Doc : df.loc[<row selection>, <column selection>]
    # dataframe.loc[dataframe.buys.isna(),  'buys']  = 0
    # dataframe.loc[dataframe.buys != 0,    'buys']  = -1
    # dataframe.loc[dataframe.sells.isna(), 'sells'] = 0
    # dataframe.loc[dataframe.sells != 0,   'sells'] = +1
    #
    # if 'index' in dataframe.columns:
    #     del dataframe['index']
    #
    # # Set index
    # dataframe.set_index('time', inplace=True)
    #
    # # Reformat datetime index, Binance's data is messy
    # dataframe.index = pd.to_datetime(dataframe.index, format='%Y-%m-%d %H:%M:%S.%f')


    scaler = MinMaxScaler(feature_range=(-1, 1))
    # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
    amplitude = scaler.fit_transform(X=amplitude.reshape(-1, 1)).reshape(-1)        # Fit to data, then transform it. amplitude est un vecteur ligne
    # amplitude = scaler.fit_transform(dataframe.close.to_numpy().reshape(-1, 1)).reshape(-1)

    samples    = int(len(amplitude)*0.7)
    train_data = amplitude[:samples]
    test_data  = amplitude[samples:]

    # train_data = [0.01401334 0.05531073 0.09731888 ... 0.32418373 0.17159021 0.26516966], len=2800

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    train_sequence = create_inout_sequences(input_data=train_data, tw=input_window)
    train_sequence = train_sequence[:-output_window]

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(input_data=test_data, tw=input_window)
    test_data = test_data[:-output_window]

    print("Data is ready.")

    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data    = source[i:i+seq_len]
    # Concatenates a sequence of tensors along a new dimension. dim=0 by default. All tensors need to be of the same size.
    input   = torch.stack(tensors=torch.stack(tensors=[item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target  = torch.stack(tensors=torch.stack(tensors=[item[1] for item in data]).chunk(input_window,1))
    return input, target

def train(train_data):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data)-1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)

        optimizer.zero_grad()       # Sets gradients of all model parameters to zero before each backward pass
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

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

def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source)-1):
            data, target = get_batch(data_source, i, 1)
            # look like the model returns static values for the output window
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    print(test_result)
    print(len(test_result))

    plt.plot(truth[:500], color="blue", label='truth[:500]')
    plt.plot(test_result, color="red", label='test_result')
    plt.plot(test_result-truth, color="green", label='test_result-truth')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.legend(loc='upper left')
    plt.title('plot_and_loss')
    # plt.savefig('graph/transformer-epoch%d.png'%epoch)
    plt.show()
    # plt.close()

    return total_loss / i

def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    # _ , target = get_batch(source=data_source, i=0, batch_size=1)
    # with torch.no_grad():   # turn off gradients computation, works in conjunction with .eval()
    #     for i in range(0, steps):
    #         input_data = target[-input_window:]
    #         predictions = eval_model(input_data)
    #         print(predictions)
    #         target = torch.cat((target, predictions[-1:]))
    #
    # target = target.cpu().view(-1)


    train_data_, test_data_ = get_data()

    # _ , target_ = get_batch(source=train_data_, i=0, batch_size=1)  # single batch of all the train data
    train_data_ = train_data_.cpu().view(-1)
    print(len(train_data_))
    plt.plot(train_data_, color="black", label='train_data_')
    # plt.plot(test_data_,  color="blue", label='test_data_')
    plt.legend(loc='upper left')
    plt.show()

    _ , target = get_batch(source=data_source, i=0, batch_size=1)
    with torch.no_grad():   # turn off gradients computation, works in conjunction with .eval()
        for i in range(0, steps):
            input_data = target[-input_window:]
            predictions = eval_model(input_data)
            target = torch.cat((target, predictions[-1:]))

    target = target.cpu().view(-1)

    plt.plot(target, color="red", label='data')
    plt.plot(target[:input_window], color="blue", label='data[:input_window]')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.title('predict_future')
    plt.legend(loc='upper left')
    plt.show()

# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich
# auch zu denen der predict_future
# soit il y a une erreur dans la perte ou dans la méthode de train, mais les résultats sont différents de ceux de prediction_future
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)



if __name__ == "__main__":

    train_data, val_data = get_data()
    model = TransAm().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)     # Choose the optimizer with a leanring rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
    epochs = 5      # Number of times the data is passed in the model (=number of trainings)

    # Train the model multiple times
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data=train_data)

        predict_future(eval_model=model, data_source=val_data, steps=200)

        # if epoch % 5 == 0:
        #     val_loss = plot_and_loss(eval_model=model, data_source=val_data, epoch=epoch)
        #     predict_future(eval_model=model, data_source=val_data, steps=200)
        # else:
        #     val_loss = evaluate(eval_model=model, data_source=val_data)

        print('-' * 89)
        print('| end of epoch {:3d} out of {:3d} | Duration: {:5.2f}s'.format(epoch, epochs, (time.time() - epoch_start_time)))
        print('-' * 89)

        scheduler.step()

    # predict_future(eval_model=model, data_source=val_data, steps=200)