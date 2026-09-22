"""The 2015-style VQA model: CNN image vector + LSTM question vector -> fuse -> 2,000-way answer classifier."""
import torch, torch.nn as nn, torchvision

class ClassicVQA(nn.Module):
    def __init__(self, vocab=5000, n_answers=2000, d=512):
        super().__init__()
        cnn = torchvision.models.resnet18(weights=None); cnn.fc = nn.Identity()   # ImageNet backbone minus its head
        self.cnn = cnn
        self.q_embed, self.q_lstm = nn.Embedding(vocab, 300), nn.LSTM(300, d, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(d, 1024), nn.ReLU(), nn.Linear(1024, n_answers))  # W2 σ(W1 h)

    def forward(self, image, question_ids):
        v = self.cnn(image)                                  # (B, 512) image vector
        _, (h, _) = self.q_lstm(self.q_embed(question_ids))  # (1, B, 512) question vector
        fused = v * h[0]                                     # element-wise product: the classic fusion
        return self.classifier(fused)                        # logits over the top-2,000 answers

torch.manual_seed(0)                                             # reproducible random weights
answers = ["yes", "no", "2", "1", "white", "red", "walking"] + [f"ans_{i}" for i in range(1993)]
model = ClassicVQA()
logits = model(torch.randn(1, 3, 224, 224), torch.randint(0, 5000, (1, 8)))
top = logits.softmax(-1)[0].topk(3)
print("logits", tuple(logits.shape), "-> top-3 (untrained):", [(answers[i], round(p, 5)) for p, i in zip(top.values.tolist(), top.indices.tolist())])
print("Can it ever answer 'a satellite dish'?", "a satellite dish" in answers)
