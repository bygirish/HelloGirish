"""Show & Tell (2015) in 25 lines: a CNN image vector becomes the LSTM's initial state; the LSTM writes words."""
import torch, torch.nn as nn
import torchvision

class ShowAndTell(nn.Module):
    def __init__(self, vocab=1000, d=256):
        super().__init__()
        cnn = torchvision.models.resnet18(weights=None)          # use weights="DEFAULT" for ImageNet features
        cnn.fc = nn.Identity()                                   # drop the 1000-class head -> 512-d vector
        self.cnn, self.img2h = cnn, nn.Linear(512, d)
        self.embed, self.lstm, self.out = nn.Embedding(vocab, d), nn.LSTM(d, d, batch_first=True), nn.Linear(d, vocab)

    def forward(self, image, caption_ids):
        h0 = self.img2h(self.cnn(image)).unsqueeze(0)           # image -> h0   (the whole "fusion"!)
        c0 = torch.zeros_like(h0)
        x, _ = self.lstm(self.embed(caption_ids), (h0, c0))
        return self.out(x)                                        # next-word logits at every position

torch.manual_seed(0)                                             # reproducible random weights
model = ShowAndTell()
img = torch.randn(2, 3, 224, 224)
caption = torch.randint(0, 1000, (2, 12))                        # <start> a tabby cat is leaning ...
logits = model(img, caption[:, :-1])
loss = nn.functional.cross_entropy(logits.reshape(-1, 1000), caption[:, 1:].reshape(-1))
print("logits", tuple(logits.shape), "| teacher-forced next-word loss", round(loss.item(), 2),
      "| ln(1000) =", round(torch.log(torch.tensor(1000.)).item(), 2))
