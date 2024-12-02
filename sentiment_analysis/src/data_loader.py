import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.idx = 2

    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence.lower().split()]

class SentimentDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        encoded_sentence = torch.tensor(self.vocab.encode(sentence), dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)
        return encoded_sentence, label

def collate_fn(batch):
    sentences, labels = zip(*batch)
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_sentences, labels