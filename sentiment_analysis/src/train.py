import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam
from dataset import Vocabulary, SentimentDataset, collate_fn
from model import SentimentLSTM

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

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for padded_sentences, labels in dataloader:
        optimizer.zero_grad()
        predictions = model(padded_sentences).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")