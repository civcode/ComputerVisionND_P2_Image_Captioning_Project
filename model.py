import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-34 for a lighter model
        #resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        # Freeze all layers
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Unfreeze only the last convolutional block
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.batch_norm = nn.BatchNorm1d(resnet.fc.in_features)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.drop = nn.Dropout(p=0.3)
        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch_norm(features)
        features = self.embed(features)
        features = self.drop(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        #self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        #self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5 if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        embeddings = self.drop1(embeddings)
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1) 
        hiddens, _ = self.rnn(inputs)
        hiddens = self.drop2(hiddens)
        outputs = self.linear(hiddens) 
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        predicted_sentence = []
        for i in range(max_len):
            hiddens, states = self.rnn(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_sentence.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence
