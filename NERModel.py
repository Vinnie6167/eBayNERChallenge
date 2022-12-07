import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torchmetrics
import bcolz
import pickle

DATASET_ADDRESS = "dataset/Train_Tagged_Titles.tsv.gz"


class AspectDataset(Dataset):
    def __init__(self, data, dict1, dict2):
        self.data = data
        self.dict1 = dict1
        self.dict2 = dict2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.dict1[(self.data.loc[idx,"Token"]).lower()], self.dict2[(self.data.loc[idx,"Tag"]).lower()]


class NERModel(nn.Module):
    def __init__(self, glove, rnn_dim, rnn_layers, vocab, num_aspects):

        super(NERModel, self).__init__()

        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers

        # Build glove embedding matrix
        emb_dim = len(glove[glove.keys()[0]])
        vocab_size = len(vocab)
        emb_matrix = torch.zeros((vocab_size, emb_dim))

        for i, word in enumerate(vocab):
            try: 
                emb_matrix[i] = glove[word]
            except KeyError:
                print(f"Warning: No Glove embedding for {word} - using random embedding")
                emb_matrix[i] = torch.rand(emb_dim)

        # Initializing Embedding Layer
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.embedding_layer.load_state_dict({'weight': emb_matrix})
        self.emb_layer.weight.requires_grad = False

        # Initializing RNN Layer
        dropout = 0.5 if rnn_layers > 1 else 0
        self.rnn = nn.RNN(emb_dim, rnn_dim, rnn_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Initializing Linear Layer
        self.linear_layer = nn.Linear(rnn_dim * 2, num_aspects)

    def forward(self, words):

        # Apply Embedding Layer
        emb_out = self.embedding_layer(words)

        # Apply RNN Layer
        hidden_0 = torch.zeros(self.n_layers * 2, self.hidden_dim)
        rnn_out, _ = self.rnn(emb_out, hidden_0)

        # Apply Linear Layer
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim * 2)
        logits = self.linear_layer(rnn_out)

        return logits
    
    def predict(self, words):

        logits = self.forward(words)
        preds = logits.argmax(1)
        
        return preds


def train(train_dataloader, val_dataloader, model, loss_fn, optimizer, metric, num_epochs, trainloss, val_loss_l, trainaccuracy, val_accuracy_l, score_l):
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch} ===")
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            logits = model(X)
            loss = loss_fn(logits, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute loss and accuracy on entire dataset
        train_loss, correct = 0, 0

        with torch.no_grad():
            count = 0
            for X, y in train_dataloader:
                count += 1
                logits = model(X)
                train_loss += loss_fn(logits, y).item()
                correct += (logits.argmax(1) == y).type(torch.float).sum().item()

        train_loss /= len(train_dataloader)
        correct /= len(train_dataloader.dataset)
        trainloss.append(train_loss)
        trainaccuracy.append(correct)
        print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        
        val_loss, correct, score = 0, 0, 0

        with torch.no_grad():
            count = 0
            for X, y in val_dataloader:
                count += 1
                logits = model(X)
                score += metric(logits, y).item()
                val_loss += loss_fn(logits, y).item()
                correct += (logits.argmax(1) == y).type(torch.float).sum().item()

        val_loss /= len(val_dataloader)
        correct /= len(val_dataloader.dataset)
        score /= len(val_dataloader)
        val_loss_l.append(val_loss)
        val_accuracy_l.append(correct)
        score_l.append(score)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        print("F1 Score: ", score)


def main():

    # Load pre-trained word embeddings (glove)
    vectors = bcolz.open('glove_data/6B.50.dat')[:]
    words = pickle.load(open('glove_data/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open('glove_data/6B.50_idx.pkl', 'rb'))
    glove = {w: torch.from_numpy(vectors[word2idx[w]]) for w in words}

    # Create data loaders.
    data = pd.read_csv(DATASET_ADDRESS, compression='gzip', header=0, sep = '\t', on_bad_lines='skip')

    # Build string 2 int and int 2 string mappings
    data['Tag'].fillna(method='ffill', inplace=True)
    d =  data["Token"].tolist()

    tags = data["Tag"].tolist()
    d1 = np.array(d)

    d1 = np.delete(d1, np.where(d1 == 'The'))
    d1 = np.delete(d1, np.where(d1 == 'the'))
    d1 = np.delete(d1, np.where(d1 == 'a'))
    d1 = np.delete(d1, np.where(d1 == 'an'))

    d2 = np.array(tags)
    d1 = np.char.lower(d1)
    d2 = np.char.lower(d2)
    unique_d1 = np.unique(d1)
    unique_d2 = np.unique(d2)
    d1 = unique_d1.tolist()
    d2 = unique_d2.tolist()
    dict1 = {k: v for v, k in enumerate(d1)}
    dict2 = {k: v for v, k in enumerate(d2)}

    msk = np.random.rand(len(data)) < 0.7
    train_data = data[msk]
    val_data = data[~msk]
    train_data.reset_index(inplace=True, drop=True)
    val_data.reset_index(inplace=True, drop=True)
    train_data = AspectDataset(train_data, dict1, dict2)
    val_data = AspectDataset(val_data, dict1, dict2)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
    
    # Get vocabulary size

    target_vocab = list(d1)

    model = Model(vocab_size=len(dict1), embedding_dim=50, num_aspects=len(dict2), hidden_dim=64, n_layers=2, glove=glove, target_vocab=target_vocab)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    metric = torchmetrics.classification.MulticlassF1Score(num_classes = len(dict2), average="weighted")
    epochs = 100
    trainloss = []
    val_loss = []
    trainaccuracy = []
    val_accuracy = []
    f1score = []
    xlabel = list(range(0,epochs))
    train(train_dataloader, val_dataloader, model, loss_fn, optimizer, metric, epochs, trainloss, val_loss, trainaccuracy, val_accuracy, f1score)

    plt.plot(np.array(xlabel), np.array(trainloss), label='Train Loss')
    plt.plot(np.array(xlabel), np.array(val_loss), label='Test Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.plot(np.array(xlabel), np.array(trainaccuracy), label='Train Accuracy')
    plt.plot(np.array(xlabel), np.array(val_accuracy), label='Test Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(np.array(xlabel), np.array(f1score))
    plt.title('F1 Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()