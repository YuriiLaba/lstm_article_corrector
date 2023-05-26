import json

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

from sklearn.preprocessing import LabelEncoder

from bi_lstm_classifier import BiLSTMClassifier


class ArticlesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class ArticleCorrector:
    def __init__(self, path_to_train_data, path_to_train_labels, path_to_test_data, path_to_test_labels):
        with open(path_to_train_data, 'r') as f:
            self.train_data = json.load(f)

        with open(path_to_train_labels, 'r') as f:
            self.train_labels = json.load(f)

        with open(path_to_test_data, 'r') as f:
            self.test_data = json.load(f)

        with open(path_to_test_labels, 'r') as f:
            self.test_labels = json.load(f)

        self.articles_list = ["a", "an", "the"]
        self.label_encoder = LabelEncoder()
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_training_data(self):
        prepared_training_set = []
        prepared_training_labels = []

        for train_sent, train_labels in zip(self.train_data, self.train_labels):
            for idx, train_token in enumerate(train_sent):
                if train_token in self.articles_list:

                    prepared_training_set_ = []

                    prepared_training_set_.extend(train_sent[idx-2: idx])

                    if train_labels[idx] is None:
                        prepared_training_set_.extend([train_sent[idx]])
                        prepared_training_labels.append(train_sent[idx])
                    else:
                        prepared_training_set_.extend([train_sent[idx]])
                        prepared_training_labels.append(train_labels[idx])

                    prepared_training_set_.extend(train_sent[idx+1: idx+3])
                    prepared_training_set.append(prepared_training_set_)

        encoded_labels = self.label_encoder.fit_transform(prepared_training_labels)
        self.vocab = build_vocab_from_iterator(prepared_training_set, specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        numerical_data = [[self.vocab[token] for token in example] for example in prepared_training_set]
        padded_data = pad_sequence([torch.tensor(example) for example in numerical_data], batch_first=True)
        return padded_data, encoded_labels

    def train(self):
        padded_train_data, encoded__train_labels = self.prepare_training_data()

        input_size = len(self.vocab)
        hidden_size = 256
        num_classes = 3

        learning_rate = 0.001
        num_epochs = 5
        batch_size = 32

        model = BiLSTMClassifier(input_size, hidden_size, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = ArticlesDataset(padded_train_data, encoded__train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.to(self.device)

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_predictions

            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy*100:.2f}%')

        return model

    def make_predictions(self, model, data):
        model.eval()

        predictions = []

        for sample in data:
            predictions_ = []

            for idx, t in enumerate(sample):
                if t not in self.articles_list:
                    predictions_.append(None)
                else:

                    infer_data = []
                    infer_data.extend(sample[idx-2: idx])
                    infer_data.extend([sample[idx]])
                    infer_data.extend(sample[idx+1: idx+3])

                    infer_data = [self.vocab[example] for example in infer_data]

                    with torch.no_grad():
                        outputs = model(torch.tensor([infer_data]).to(self.device))
                        _, predicted = torch.max(outputs.data, 1)
                        predicted_prob = torch.max(torch.softmax(outputs, dim=-1)).item()
                        decoded_class = self.label_encoder.inverse_transform([predicted.item()])[0]

                    predictions_.append([decoded_class, predicted_prob])

            predictions.append(predictions_)
            return predictions





