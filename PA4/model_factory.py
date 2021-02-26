################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
import lstm
# Build and return the model here based on the configuration.
def get_model(config_data, vocab_size):
    hidden_size = int(config_data['model']['hidden_size'])
    embedding_size = int(config_data['model']['embedding_size'])
    model_type = str(config_data['model']['model_type'])
    dropout = float(config_data['model']['dropout'])
    # You may add more parameters if you want
    if model_type == 'LSTM':
        model = lstm.ResLSTM(hidden_size, embedding_size, dropout, vocab_size)
    else:
        raise NotImplementedError("Model Factory Not Implemented")
