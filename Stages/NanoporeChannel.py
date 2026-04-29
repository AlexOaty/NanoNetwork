import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Config ---
BIT_LENGTH = 50
DELTA = 3
WINDOW_LENGTH = 6
REDUNDANCY = 1.44  # Slightly higher to help convergence first
BATCH_SIZE = 512
STEPS = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Theoretical minimum: bits/2 * redundancy + padding for the window ends
SEQ_LEN = int((BIT_LENGTH / 2) * REDUNDANCY)


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

# --- Initialization ---
model = Model(BIT_LENGTH, SEQ_LEN)
model.load_state_dict(torch.load("nano_model_50-3-6-1.44.pth", map_location=device))
model.eval()

test_bits = torch.randint(0, 2, (3000, BIT_LENGTH)).float().to(device)
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