import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

feats = 128
epochs = 2000
limit = 1000
groupSize = 128
bitLength = 4
redundancy = 0.75

class Encoder(nn.Module):
    def __init__(self, input_bits):
        super().__init__()
        self.seq_len = int(input_bits * redundancy)
        self.net = nn.Sequential(
            nn.Linear(input_bits, feats),
            nn.ReLU(),
            nn.Linear(feats, self.seq_len * 4)
        )

    def forward(self, x):
        # x shape: (batch_size, input_bits)
        out = self.net(x)  # (batch_size, seq_len*4)
        out = out.view(-1, self.seq_len, 4)  # (batch_size, seq_len, 4)
        probs = F.gumbel_softmax(out, tau = 1, hard=True, dim=-1)
        return probs

class Decoder(nn.Module):
    def __init__(self, output_bits, seq_len):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(nn.Flatten(),
            nn.Linear(seq_len * 4, feats),
            nn.ReLU(),
            nn.Linear(feats, output_bits), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class SimpleDNAAutoencoder(nn.Module):
    def __init__(self, input_bits):
        super().__init__()
        self.seq_len = int(input_bits * redundancy)  # 2 bits per base
        self.encoder = Encoder(input_bits)
        self.decoder = Decoder(input_bits, self.seq_len)

    def forward(self, x):
        dna = self.encoder(x)
        out = self.decoder(dna)
        return out


# ---- Training ----
# tiny example
input_bits = bitLength
model = SimpleDNAAutoencoder(input_bits)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

best_loss = float("inf")
count = 0
max = limit

xpoints = np.array([])
ypoints = np.array([])

# training on random tiny data
for epoch in range(epochs):
    # batch of 16 random binary sequences
    x = torch.randint(0, 2, (groupSize, input_bits)).float()
    output = model(x)
    loss = loss_fn(output, x)

    val_loss = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if val_loss < best_loss:
        count = 0
        best_loss = val_loss
    else:
        count += 1

    if count >= max:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        break

    if epoch % 10 == 0:
        xpoints = np.concatenate((xpoints, np.array([epoch])))
        ypoints = np.concatenate((ypoints, np.array([loss.item()])))
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), f"dna_model_{bitLength}.pth")

plt.plot(xpoints, ypoints)
plt.ylim(0, 1)
plt.show()

# ---- Test ----
x_test = torch.randint(0, 2, (1, input_bits)).float()
dna = model.encoder(x_test)
print("Input bits: ", x_test)
print("DNA one-hot: ", dna)
recovered_bits = model.decoder(dna)
print("Recovered bits (probabilities): ", recovered_bits)
print("Recovered bits (0/1): ", (recovered_bits > 0.5).float())






