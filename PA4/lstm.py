#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
from torchvision import models
from constants import *


class ResNetEncoder(nn.Module):
    def __init__(self, hidden_size):
        self.model = models.resnet50(pretrained=True)
        self.fc_in_feature = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.model.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(in_features=self.fc_in_feature, out_features=hidden_size, bias=True)
        self.batch_normalize = nn.BatchNorm1d(hidden_size, momentum=0.01)
    def foward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.batch_normalize(features)
        return features
    
class LSTMDecoder(nn.Module):
    def __init__(self, word_embedding_size, hidden_size, dropout, vocab_size):
        self.input_size = word_embedding_size
        self.hidden_size= hidden_size
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.wordEmbedded = nn.Embedding(vocab_size, embedding_size)
        self.linear_Embed2Word = nn.linear(in_features = self.hidden_size, out_features = self.vocab_size, bias = True)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, dropout=dropout)
        #self.softmax = nn.functional.softmax()
    def forward(self, captions, features):
        # captions are of shape of BatchSize * sequenceLength * 1 => 0 <= n < vacab_size
        hc_init = (features, features)
        #captions_compact = self.linear_word2Embed(captions)
        captions_compact = self.embedding(captions)
        lstm_out, hc_out = self.lstm(captions_compact, hc_init)
        final = self.linear_Embed2Word(lstm_out)
        return final
        
        
class ResLSTM(nn.Module):
    def __init__(self, hidden_size, word_embedding_size, dropout, vocab_size):
        self.encoder = ResNetEncoder(hidden_size)
        self.decoder = LSTMDecoder(word_embedding_size, hidden_size, dropout, vocab_size)        
        
    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        final = self.decoder(captions, features)
        return final
    


# In[ ]:




