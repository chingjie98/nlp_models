import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam
from dataset import Vocabulary, SentimentDataset, collate_fn
from model import SentimentLSTM
from predict import predict_sentiment

def main():
    data = [
        ("I love this product", 1),
        ("This is the worst experience ever", 0),
        ("Absolutely fantastic", 1),
        ("Not good at all", 0),
    ]

    vocab = Vocabulary()
    for sentence, _ in data:
        vocab.add_sentence(sentence)

    dataset = SentimentDataset(data, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(vocab.word2idx)
    embedding_dim = 16
    hidden_dim = 32
    output_dim = 1

    model = SentimentLSTM(vocab_size, embedding​⬤