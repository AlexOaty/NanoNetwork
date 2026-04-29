import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib.pyplot import title
from scipy.ndimage import maximum
from torch.nn.functional import gumbel_softmax

feats = 256
steps = 3000
limit = 3000
groupSize = 512
bitLength = 50
bar = False

delta = 2
window_lengths = [4, 5]

redundancy = delta
redundancies = np.linspace(delta, delta, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_bits, redundancy):
        super().__init__()
        self.seq_len = int(input_bits * 0.5 * redundancy)
        self.net = nn.Sequential(
            nn.Linear(input_bits, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feats),
            nn.ReLU(),
            nn.Linear(feats, self.seq_len * 4)
        )

    def forward(self, x, epoch):
        out = self.net(x)
        out = out.view(-1, self.seq_len, 4)
        tau = max(0.1, 2.0 * (0.99 ** epoch))
        probs = F.gumbel_softmax(out, tau=tau, hard=True, dim=-1)
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
            padding=self.L - delta,
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

        # Replace the end of your Decoder Sequential with this:
        self.net = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * num_windows, 256),  # num_windows is already calculated
            nn.ReLU(),
            nn.Linear(256, output_bits),
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

result = []
for window_length in window_lengths:
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

        if(True):
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
        if step == steps-1:
            result.append(loss.item())
        if step % 10 == 0:
            xpoints = np.concatenate((xpoints, np.array([step])))
            ypoints = np.concatenate((ypoints, np.array([loss.item()])))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {current_lr}")


    torch.save(model.state_dict(), f"nano_model_{bitLength}-{delta}-{window_length}.pth")
    plt.plot(xpoints, ypoints)
    #if bar:
        #plt.bar(redundancy, result[-1], data=f"Redundancy: {redundancy}", width=0.03)
        #plt.text(redundancy, result[-1], f"{result[-1]:.3f}", ha='center')

if bar:
    plt.ylim(0, 0.2)
    plt.xticks(redundancies.tolist())
else:
    plt.ylim(0, 3)
    plt.legend(window_lengths, title="Window Length")
plt.title(f"Delta={delta}, Window Length=4-5, Redundancy=2")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()


# ---- Test ----
x_test = torch.randint(0, 2, (3, input_bits)).float().to(device)
dna = model.encoder(x_test, 4000)

print("Input bits: ", x_test)
print("DNA one-hot: ", dna)

nanopore = model.nanopore(dna) / window_length
print("Nanopore read vector: ", nanopore)

recovered_bits = model.decoder(nanopore)
print("Recovered bits (probabilities): ", recovered_bits)
print("Recovered bits (0/1): ", (recovered_bits > 0.5).float())