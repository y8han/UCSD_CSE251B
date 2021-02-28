#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:

import torch
import torch.nn as nn
from torchvision import models
import torch.utils.data as data
from constants import *


class ResNetEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.fc_in_feature = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.model.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(in_features=self.fc_in_feature, out_features=hidden_size, bias=True)
        self.batch_normalize = nn.BatchNorm1d(hidden_size, momentum=0.01)
    def forward(self, images):
        features = self.model(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.batch_normalize(features)
        return features
    
class LSTMDecoder(nn.Module):
    def __init__(self, word_embedding_size, hidden_size, dropout, vocab_size, temperature, max_length):
        super().__init__()
        self.input_size = word_embedding_size
        self.hidden_size= hidden_size
        self.dropout = dropout
        if temperature == 0:
            self.temperature = 0.001
        else:
            self.temperature = temperature
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.wordEmbedded = nn.Embedding(vocab_size, word_embedding_size)
        self.linear_Embed2Word = nn.Linear(in_features = self.hidden_size, out_features = self.vocab_size, bias = True)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, dropout=dropout)
        self.caption_softmax = nn.Softmax(dim=1)
    def forward(self, captions, features):
        # captions are of shape of BatchSize * sequenceLength * 1 => 0 <= n < vacab_size
        features = torch.unsqueeze(features, 0)
        hc_init = (features, features)
        captions_compact = self.wordEmbedded(captions)
        lstm_out, _ = self.lstm(captions_compact, hc_init)
        final = self.linear_Embed2Word(lstm_out)
        return final
    def generateCaption(self, feature, stochastic=False):
        initial_input = torch.ones((feature.shape[0], 1)).long().to('cuda')
        #torch.tensor(1).to('cuda') # this is the '<start>'
        lstm_input = self.wordEmbedded(initial_input)
        feature = torch.unsqueeze(feature, 0)
        hc_states = (feature, feature)
        res = []
        for i in range(self.max_length):
            #print(lstm_input.shape)
            lstm_output, hc_states = self.lstm(lstm_input, hc_states)
            lstm_final_word = self.linear_Embed2Word(lstm_output)
            #print(lstm_final_word.shape)
            lstm_final_word = lstm_final_word.squeeze()
            #print(lstm_final_word, "This is lstm output for generation 1")            
            if stochastic:
                #print("Stochastic", self.temperature)
                lstm_final_word = self.caption_softmax(lstm_final_word/self.temperature)
                predicted = data.WeightedRandomSampler(weights=lstm_final_word, num_samples = 1, replacement=False)
                predicted = torch.tensor(list(predicted)).long().to('cuda')
                predicted = torch.squeeze(predicted)
                #print(predicted, predicted.shape)
            else:
                _, predicted = lstm_final_word.max(1)
                #predicted = torch.unsqueeze(predicted, 1)
                #print(predicted, "This is lstm output for generation 1")
            #print(predicted.shape, predicted)
            res.append(predicted)
            lstm_input = self.wordEmbedded(predicted)
            lstm_input = torch.unsqueeze(lstm_input, 1)
        res = torch.stack(res, 1)
        #print(res.shape)
        return res
        
class ResLSTM(nn.Module):
    def __init__(self, hidden_size, word_embedding_size, dropout, vocab_size, temperature = 1, max_length = 25):
        super().__init__()
        self.encoder = ResNetEncoder(hidden_size)
        self.decoder = LSTMDecoder(word_embedding_size, hidden_size, dropout, vocab_size, temperature, max_length)        
        
    def forward(self, images, captions):
        features = self.encoder(images)
        final = self.decoder(captions, features)
        return final
    def generateCaption(self, images, stochastic):
        features = self.encoder(images)
        final = self.decoder.generateCaption(features, stochastic)
        return final
# In[ ]:





# In[ ]:




