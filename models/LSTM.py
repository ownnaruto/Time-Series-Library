import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers

        self.lstm1 = nn.LSTM(input_size=self.channels, hidden_size=self.hidden_size,
                             num_layers=self.num_layers, batch_first=True)

        self.linear1 = nn.Linear(self.hidden_size, 1)

        self.seq2pred = nn.Linear(self.seq_len, self.pred_len)

        self.linear2 = nn.Linear(1, self.channels)

        self.lstm2 = nn.LSTM(input_size=self.channels, hidden_size=self.channels, num_layers=self.num_layers,
                             batch_first=True)

    def forecast(self, x):
        # x: B, seq_len, enc_in
        x, _ = self.lstm1(x)
        x = self.linear1(x).squeeze(-1)  # B, seq_len
        x = self.seq2pred(x).unsqueeze(-1)  # B, pred_len, 1
        x = self.linear2(x)  # B, pred_len, enc_in
        x, _ = self.lstm2(x)  # B, pred_len, enc_in
        return x

    def imputation(self, x):
        raise NotImplemented

    def anomaly_detection(self, x_enc):
        raise NotImplemented

    def classification(self, x_enc):
        raise NotImplemented

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x)
        else:
            raise NotImplemented
