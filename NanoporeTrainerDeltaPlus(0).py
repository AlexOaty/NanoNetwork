import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.ndimage import maximum
from torch.nn.functional import gumbel_softmax

feats = 256
steps = 5000
limit = 1000
groupSize = 512
bitLength = 20

delta = 2
window_length = 4

redundancies = np.linspace(1.8, delta * 2, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_bits, redundancy):
        super().__init__()
        self.seq_len = int(input_bits * 0.5 * redundancy)
        self.net = nn.Sequential(
            nn.Linear(input_bits, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feats),
            nn.ReLU(),
            nn.Linear(feats, self.seq_len * 4)
        )

    def forward(self, x, epoch):
        out = self.net(x)
        out = out.view(-1, self.seq_len, 4)
        tau = max(0.1, 2.0 * (0.99 ** epoch))
        probs = F.gumbel_softmax(out, tau=1, hard=True, dim=-1)
        return probs


class NanoporeChannel(nn.Module):
    def __init__(self, L, Delta):
        super().__init__()
        self.L = L
        self.Delta = Delta

    def forward(self, dna):
        # dna: (batch, seq_len, 4)

        dna = dna.permute(0, 2, 1)  # → (batch, 4, seq_len)

        kernel = torch.ones(4, 1, self.L, device=dna.device)

        out = F.conv1d(
            dna,
            kernel,
            stride=self.Delta,
            padding=self.L - 1,
            groups=4
        )

        return out.permute(0, 2, 1)  # → (batch, windows, 4)


class Decoder(nn.Module):
    def __init__(self, output_bits, seq_len):
        super(Decoder, self).__init__()
        # Calculate windows
        dummy_dna = torch.zeros(1, seq_len, 4)
        nanopore = NanoporeChannel(window_length, delta)
        num_windows = nanopore(dummy_dna).shape[1]

        self.net = nn.Sequential(
            # First layer: Extract local "base-pair" features
            nn.Conv1d(4, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Second layer: Extract "window-overlap" features
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Third layer: Bring it down to a bit-like representation
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(output_bits), # Align signal length to bit length
            nn.Conv1d(64, 1, kernel_size=1),    # 1 channel = 1 bit probability
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, 4, W)
        return self.net(x)


class NanoporeNetwork(nn.Module):
    def __init__(self, input_bits, redundancy):
        super().__init__()
        self.seq_len = int(input_bits * (redundancy/2))
        self.encoder = Encoder(input_bits, redundancy)
        self.decoder = Decoder(input_bits, self.seq_len)
        self.nanopore = NanoporeChannel(window_length, delta)

    def forward(self, x, epoch):
        dna = self.encoder(x, epoch)
        nano = self.nanopore(dna) / window_length
        out = self.decoder(nano)
        return out

input_bits = bitLength
loss_fn = nn.BCELoss()

for redundancy in redundancies:
    # ---- Training ----
    model = NanoporeNetwork(input_bits, redundancy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

    best_loss = float("inf")
    count = 0
    maxL = limit

    xpoints = np.array([])
    ypoints = np.array([])

    for step in range(steps):
        x = torch.randint(0, 2, (groupSize, input_bits)).float().to(device)
        output = model(x, step)

        loss = loss_fn(output, x)

        if(delta == 1):
            target_idx = (x[:, 0::2] * 2 + x[:, 1::2]).long()

            B, L = target_idx.shape
            S = model.seq_len


            # Create positions in original sequence
            pos = torch.linspace(0, (L - 1), S, device=target_idx.device)

            # Convert to indices
            idx = pos.long() # or floor()

            target_idx = target_idx.gather(1, idx.unsqueeze(0).expand(B, -1))

            logits = model.encoder.net(x).view(-1, model.seq_len, 4)

            dna_loss = F.cross_entropy(
                logits.view(-1, 4),
                target_idx.view(-1)
            )

            loss += dna_loss


        val_loss = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(loss)

        if val_loss < best_loss:
            count = 0
            best_loss = val_loss
        else:
            count += 1

        if count >= maxL:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            break

        if step % 10 == 0:
            xpoints = np.concatenate((xpoints, np.array([step])))
            ypoints = np.concatenate((ypoints, np.array([loss.item()])))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {current_lr}")


    torch.save(model.state_dict(), f"nano_model_{bitLength}-{delta}-{window_length}.pth")

    plt.plot(xpoints, ypoints, data=f"Redundancy: {redundancy}")

plt.ylim(0, ypoints[0])
plt.title(f"Delta={delta}, Window Length={window_length}")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend(redundancies, title="Redundancy")
plt.show()


# ---- Test ----
x_test = torch.randint(0, 2, (1, input_bits)).float().to(device)
dna = model.encoder(x_test, 4000)

print("Input bits: ", x_test)
print("DNA one-hot: ", dna)

nanopore = model.nanopore(dna) / window_length
print("Nanopore read vector: ", nanopore)

recovered_bits = model.decoder(nanopore)
print("Recovered bits (probabilities): ", recovered_bits)
print("Recovered bits (0/1): ", (recovered_bits > 0.5).float())