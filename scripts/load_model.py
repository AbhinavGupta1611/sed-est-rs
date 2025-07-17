import torch
from models.mTAN_LSTM import mTANLSTMModel
from models.lstm_model import LSTMModel

def load_mTANLSTM(filename, input_dim, hidden_dim_lstm, n_lstm_layers, hidden_dim_mtan, embed_dim_mtan, num_heads, Lseq, output_dim, non_linear_transform, RS_inds):
    lstm = mTANLSTMModel(input_dim, hidden_dim_lstm, n_lstm_layers, hidden_dim_mtan, embed_dim_mtan, num_heads, Lseq, output_dim, non_linear_transform, RS_inds)
    lstm.cuda()
    state = torch.load(open(filename, 'rb'))
    return lstm.load_state_dict(state, strict = True)

def load_LSTM(filename, input_dim, hidden_dim, n_layers, output_dim):
    lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
    lstm.cuda()
    state = torch.load(open(filename, 'rb'))
    return lstm.load_state_dict(state, strict = True)