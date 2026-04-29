import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
BIT_LENGTH = 50
DELTA = 2
WINDOW_LENGTH = 6
Redundancies = np.linspace(1.2, 1.6, 11)
BATCH_SIZE = 512
STEPS = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, in_bits, out_len):
        super().__init__()
        # Break the bits into a "grid" or "latent sequence" first
        self.latent_dim = 128
        self.initial = nn.Linear(in_bits, out_len * self.latent_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(self.latent_dim, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 4, kernel_size=3, padding=1)
        )

    def forward(self, x, tau, hard):
        x = self.initial(x).view(-1, self.latent_dim, SEQ_LEN)
        logits = self.conv(x).permute(0, 2, 1) # [B, SEQ_LEN, 4]
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

class Decoder(nn.Module):
    def __init__(self, bit_len, num_windows):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * num_windows, 512),
            nn.SiLU(),
            nn.Linear(512, bit_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1))


class NanoporeChannel(nn.Module):
    def forward(self, dna):
        dna = dna.permute(0, 2, 1)
        kernel = torch.ones(4, 1, WINDOW_LENGTH).to(dna.device)
        # Using padding=0 to avoid "zero-inflation" at the edges
        out = F.conv1d(dna, kernel, stride=DELTA, padding=WINDOW_LENGTH-DELTA, groups=4)
        return out.permute(0, 2, 1)

class Model(nn.Module):
    def __init__(self, BIT_LENGTH, SEQ_LEN):
        super().__init__()
        self.enc = Encoder(BIT_LENGTH, SEQ_LEN).to(device)
        self.chan = NanoporeChannel().to(device)
        dummy_dna = torch.zeros(1, SEQ_LEN, 4).to(device)
        num_windows = self.chan(dummy_dna).shape[1]
        self.dec = Decoder(BIT_LENGTH, num_windows).to(device)

history = []
result = []
for REDUNDANCY in Redundancies:
    SEQ_LEN = int((BIT_LENGTH / 2) * REDUNDANCY)
    model = Model(BIT_LENGTH, SEQ_LEN).to(device)
    enc = Encoder(BIT_LENGTH, SEQ_LEN).to(device)
    chan = NanoporeChannel().to(device)
    dummy_dna = torch.zeros(1, SEQ_LEN, 4).to(device)
    num_windows = chan(dummy_dna).shape[1]
    dec = Decoder(BIT_LENGTH, num_windows).to(device)

    optimizer = torch.optim.AdamW(list(model.enc.parameters()) + list(model.dec.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    loss_fn = nn.BCELoss()

    for step in range(STEPS):
        # Higher starting Tau (5.0) for more exploration
        tau = max(0.1, 5.0 * (0.9992 ** step))
        # Switch to hard mode much earlier (step 500)
        hard = step > 500

        bits = torch.randint(0, 2, (BATCH_SIZE, BIT_LENGTH)).float().to(device)
        dna = model.enc(bits, tau, hard)
        signal = model.chan(dna)

        # Increase noise significantly during the "hard" transition
        noise_level = 0.02 if step > 500 else 0.005
        signal = signal + torch.randn_like(signal) * noise_level

        preds = model.dec(signal)
        loss = loss_fn(preds, bits)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to handle the "hard" switch spikes
        torch.nn.utils.clip_grad_norm_(list(model.enc.parameters()) + list(model.dec.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        if step == STEPS - 1:
            result.append(loss.item())

        history.append(loss.item())
        if step % 250 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.5f} | Tau: {tau:.2f} | Hard: {hard}")

    # Save Model to File
    torch.save(model.state_dict(), f"nano_model_{BIT_LENGTH}-{DELTA}-{WINDOW_LENGTH}-{REDUNDANCY}.pth")
    plt.bar(REDUNDANCY, result[-1], data=f"Redundancy: {REDUNDANCY}", width=0.03)
    plt.text(REDUNDANCY, result[-1], f"{result[-1]:.3f}", ha='center')

plt.ylim(0, 0.2)
plt.xticks(Redundancies.tolist())
plt.title(f"Delta={DELTA}, Window Length={WINDOW_LENGTH}")
plt.xlabel("Redundancy")
plt.ylabel("Final Loss")
plt.show()

test_bits = torch.randint(0, 2, (1, BIT_LENGTH)).float().to(device)
with torch.no_grad():
    print(test_bits)
    dna = model.enc(test_bits, 0.1, True)
    print(dna)
    nano = model.chan(dna)
    print(nano)
    pred = (model.dec(nano) > 0.5).float()
    print(pred)
    print(f"Final Accuracy: {(test_bits == pred).float().mean().item() * 100:.2f}%")
    hamming_distances = torch.sum(pred != test_bits, dim=1)
    print(f"Hamming Distance Avg: {torch.mean(hamming_distances.float()).item():.2f}")