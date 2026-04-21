import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Config ---
BIT_LENGTH = 50
DELTA = 2
WINDOW_LENGTH = 6
REDUNDANCY = 1.29  # Slightly higher to help convergence first
BATCH_SIZE = 512
STEPS = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Theoretical minimum: bits/2 * redundancy + padding for the window ends
SEQ_LEN = int((BIT_LENGTH / 2) * REDUNDANCY)


class Encoder(nn.Module):
    def __init__(self, in_bits, out_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_bits, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, out_len * 4)
        )

    def forward(self, x, tau, hard):
        logits = self.net(x).view(-1, SEQ_LEN, 4)
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


class Decoder(nn.Module):
    def __init__(self, bit_len, num_windows):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * num_windows, 512),
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
        return out.permute(0, 2, 1) / WINDOW_LENGTH


# --- Initialization ---
enc = Encoder(BIT_LENGTH, SEQ_LEN).to(device)
chan = NanoporeChannel().to(device)
dummy_dna = torch.zeros(1, SEQ_LEN, 4).to(device)
num_windows = chan(dummy_dna).shape[1]
dec = Decoder(BIT_LENGTH, num_windows).to(device)

optimizer = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
loss_fn = nn.BCELoss()

history = []
for step in range(STEPS):
    # Higher starting Tau (5.0) for more exploration
    tau = max(0.1, 5.0 * (0.9992 ** step))
    # Switch to hard mode much earlier (step 500)
    hard = step > 500

    bits = torch.randint(0, 2, (BATCH_SIZE, BIT_LENGTH)).float().to(device)
    dna = enc(bits, tau, hard)
    signal = chan(dna)

    # Increase noise significantly during the "hard" transition
    noise_level = 0.02 if step > 500 else 0.005
    signal = signal + torch.randn_like(signal) * noise_level

    preds = dec(signal)
    loss = loss_fn(preds, bits)

    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to handle the "hard" switch spikes
    torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
    optimizer.step()
    scheduler.step()

    history.append(loss.item())
    if step % 250 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.5f} | Tau: {tau:.2f} | Hard: {hard}")

# --- Plot and Final Test ---
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.yscale('log')
plt.title(f"Corrected Convergence (R={REDUNDANCY})")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

test_bits = torch.randint(0, 2, (1, BIT_LENGTH)).float().to(device)
with torch.no_grad():
    print(test_bits)
    dna = enc(test_bits, 0.1, True)
    print(dna)
    nano = chan(dna)
    print(nano)
    pred = (dec(nano) > 0.5).float()
    print(pred)
    print(f"Final Accuracy: {(test_bits == pred).float().mean().item() * 100:.2f}%")