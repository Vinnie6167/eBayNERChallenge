import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torchmetrics
import bcolz
import pickle
from torch.nn.utils.rnn import pad_sequence
import random

DATASET_ADDRESS = "dataset/Train_Tagged_Titles.tsv.gz"


class AspectDataset(Dataset):
    def __init__(self, titles, word2idx, aspect2idx):
        self.titles = titles
        self.word2idx = word2idx
        self.aspect2idx = aspect2idx

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = [self.word2idx[w[0].lower()] for w in self.titles[idx]]
        labels = [self.aspect2idx[w[1]] for w in self.titles[idx]]
        return torch.tensor(title, dtype=torch.int), torch.tensor(labels, dtype=torch.int)

    def getNumAspects(self):
        return len(self.aspect2idx)


def pad_batch(batch):
    (xx, yy) = zip(*batch)
    x_lens = [x.shape[-1] for x in xx]
    y_lens = [len(y) for y in yy]
    
    max_x = max(x_lens)
    
    xx_pad = torch.zeros((len(x_lens), max_x))
    for i, x in enumerate(xx):
        xx_pad[i, 0:x_lens[i]] = x

    yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1)

    return xx_pad, yy_pad, x_lens, y_lens


class NERModel(nn.Module):
    def __init__(self, glove, rnn_dim, rnn_layers, vocab, word2idx, num_aspects):

        super(NERModel, self).__init__()

        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.num_aspects = num_aspects

        # Build glove embedding matrix
        emb_dim = len(glove[list(glove.keys())[0]])
        vocab_size = len(vocab)
        emb_matrix = torch.zeros((vocab_size, emb_dim))

        non = 0
        for i, word in enumerate(vocab):
            try: 
                emb_matrix[i] = glove[word]
            except KeyError:
                non += 1
                emb_matrix[i] = torch.rand(emb_dim)
        print(f"Using random embedding for {non}/{len(vocab)} words")

        # Initializing Embedding Layer
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.embedding_layer.load_state_dict({'weight': emb_matrix})

        # Initializing RNN Layer
        dropout = 0.5 if rnn_layers > 1 else 0
        # self.rnn = nn.RNN(emb_dim, rnn_dim, rnn_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=dropout)

        # Initializing Linear Layer
        self.linear_layer = nn.Linear(rnn_dim * 2, num_aspects)

    def forward(self, words):

        # Apply Embedding Layer
        emb_out = self.embedding_layer(words)

        # Apply RNN Layer
        rnn_out, _ = self.rnn(emb_out)

        # Apply Linear Layer
        batch_size, max_len, hidden_dim = rnn_out.shape
        rnn_out = rnn_out.reshape((-1, hidden_dim))
        logits = self.linear_layer(rnn_out)
        logits = logits.reshape((batch_size, max_len, self.num_aspects))

        return logits
    
    def predict(self, words):

        logits = self.forward(words)
        preds = logits.argmax(1)
        
        return preds


def train(model, num_epochs, train_dataloader, test_dataloader, loss_fn, optimizer):
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        
        print(f"=== Epoch {epoch} ===")

        # Train
        for X, y, _, _ in train_dataloader:
            X = X.int()

            # Compute prediction and loss
            logits = model(X)
            loss = loss_fn(logits.reshape((-1, model.num_aspects)), y.reshape((-1)).long())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Train Loss and Accuracy
        train_loss, correct, count = 0, 0, 0
        with torch.no_grad():
            for X, y, _, _ in train_dataloader:
                X = X.int()
                
                logits = model(X)
                train_loss += loss_fn(logits.reshape((-1, model.num_aspects)), y.reshape((-1)).long()).item()
                correct += (logits.argmax(2) == y).type(torch.float).sum().item()
                count += logits.shape[0] * logits.shape[1]

        train_loss /= len(train_dataloader)
        correct /= count
        train_losses.append(train_loss)
        train_accs.append(correct)

        print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        
        # Train Loss and Accuracy
        test_loss, correct, count = 0, 0, 0
        with torch.no_grad():
            for X, y, _, _ in test_dataloader:
                X = X.int()
                
                logits = model(X)
                test_loss += loss_fn(logits.reshape((-1, model.num_aspects)), y.reshape((-1)).long()).item()
                correct += (logits.argmax(2) == y).type(torch.float).sum().item()
                count += logits.shape[0] * logits.shape[1]

        test_loss /= len(test_dataloader)
        correct /= count
        test_losses.append(test_loss)
        test_accs.append(correct)

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return train_loss, train_accs, test_loss, test_accs


def f1score(model, val_dataloader):

    f1scorer = torchmetrics.classification.MulticlassF1Score(num_classes = len(model.getNumAspects()), average="weighted")

    score = 0

    with torch.no_grad():
        for X, y in val_dataloader:
            logits = model(X)
            score += f1scorer(logits, y).item()
    
    score /= len(val_dataloader)

    return score

def main():

    # Load data
    df = pd.read_csv(DATASET_ADDRESS, compression='gzip', header=0, sep = '\t', on_bad_lines='skip')
    df['Tag'].fillna(method='ffill', inplace=True)
    titles = []
    cur_title = None
    cur_record = None
    for _, row in df.iterrows():
        if cur_record is not None and row['Record Number'] == cur_record:
            cur_title.append((row['Token'], row['Tag']))
        else:
            if cur_title is not None:
                titles.append(cur_title)
            cur_record = row['Record Number']
            cur_title = []

    # Build word2idx and aspect2idx
    vocab = np.unique(np.array([word.lower() for word in df["Token"].tolist()]))
    aspects = np.unique(np.array(df["Tag"].tolist()))
    num_aspects = aspects.shape[0]
    word2idx = {k : v for v, k in enumerate(vocab)}
    # idx2word = {k : v for k, v in enumerate(vocab)}
    aspect2idx = {k : v for v, k in enumerate(aspects)}
    # idx2aspect = {k : v for k, v in enumerate(aspects)}

    # Divide into test and train dataset
    random.shuffle(titles)
    num_titles = len(titles)
    train_titles = titles[:int(num_titles*0.7)]
    test_titles = titles[int(num_titles*0.7):]
    train_dataset = AspectDataset(train_titles, word2idx, aspect2idx)
    test_dataset = AspectDataset(test_titles, word2idx, aspect2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=pad_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=pad_batch)

    # Load pre-trained word embeddings (glove)
    vectors = bcolz.open('glove_data/6B.50.dat')[:]
    words = pickle.load(open('glove_data/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open('glove_data/6B.50_idx.pkl', 'rb'))
    glove = {w: torch.from_numpy(vectors[word2idx[w]]) for w in words}

    # Model, Optimizer, Loss Function
    model = NERModel(glove, 25, 2, vocab, word2idx, num_aspects)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # Train Loop
    train_loss, train_accs, test_loss, test_accs = train(model, 50, train_dataloader, test_dataloader, loss_fn, optimizer)


    # plt.plot(np.array(xlabel), np.array(trainloss), label='Train Loss')
    # plt.plot(np.array(xlabel), np.array(val_loss), label='Test Loss')
    # plt.title('Loss')
    # plt.legend()
    # plt.show()

    # plt.plot(np.array(xlabel), np.array(trainaccuracy), label='Train Accuracy')
    # plt.plot(np.array(xlabel), np.array(val_accuracy), label='Test Accuracy')
    # plt.title('Accuracy')
    # plt.legend()
    # plt.show()

    # plt.plot(np.array(xlabel), np.array(f1score))
    # plt.title('F1 Score')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()