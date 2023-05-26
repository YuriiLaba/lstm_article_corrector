import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)

        # Concatenate the hidden states from both directions
        concatenated_output = torch.cat((output[:, -1, :self.hidden_size], output[:, 0, self.hidden_size:]), dim=1)

        output = self.fc(concatenated_output)
        return output