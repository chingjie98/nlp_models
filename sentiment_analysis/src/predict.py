import torch
from dataset import Vocabulary
from model import SentimentLSTM

def predict_sentiment(model, vocab, sentence):
    model.eval()
    with torch.no_grad():
        encoded_sentence = torch.tensor(vocab.encode(sentence), dtype=torch.long).unsqueeze(0)
        prediction = model(encoded_sentence).item()
        return "Positive" if prediction >= 0.5 else "Negative"

data = [
    ("I love this product", 1),
    ("This is the worst experience ever", 0),
    ("Absolutely fantastic", 1),
    ("Not good at all", 0),
]

vocab = Vocabulary()
for sentence, _ in data:
    vocab.add_sentence(sentence)

vocab_size = len(vocab.word2idx)
embedding_dim = 16
hidden_dim = 32
output_dim = 1

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)

test_sentence = "This product is amazing"
print(f"Sentence: '{test_sentence}', Sentiment: {predict_sentiment(model, vocab, test_sentence)}")