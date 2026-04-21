import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- Updated Config ---
BIT_LENGTH = 50
DELTA = 2
WINDOW_LENGTH = 4
REDUNDANCY = 1.25  # Slightly higher to ensure SEQ_LEN is 33 or 34
BATCH_SIZE, STEPS = 512, 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = int((BIT_LENGTH / 2) * REDUNDANCY)
P = WINDOW_LENGTH - 1
NUM_WINDOWS = ((SEQ_LEN + 2 * P) - WINDOW_LENGTH) // DELTA + 1


class FinalNanoporeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Deeper Encoder to find "Hard" distinct DNA sequences
        self.encoder = nn.Sequential(
            nn.Linear(BIT_LENGTH, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, SEQ_LEN * 4)
        )

        # Powerful Decoder to resolve overlapping "Mean" signals
        self.decoder = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * NUM_WINDOWS, 1024), nn.SiLU(),
            nn.Linear(1024, BIT_LENGTH)
        )
        self.register_buffer('kernel', torch.ones(4, 1, WINDOW_LENGTH) / WINDOW_LENGTH)

    def channel(self, dna):
        x = dna.permute(0, 2, 1)
        x_padded = F.pad(x, (P, P), value=0)
        return F.conv1d(x_padded, self.kernel, stride=DELTA, groups=4).permute(0, 2, 1)

    def forward(self, bits, tau=1.0, hard=False):
        dna_logits = self.encoder(bits).view(-1, SEQ_LEN, 4)
        dna = F.gumbel_softmax(dna_logits, tau=tau, hard=hard, dim=-1)
        signal = self.channel(dna)
        # Permute for Conv1d: [B, 4, N]
        return self.decoder(signal.permute(0, 2, 1)), dna, signal


# --- Setup Training ---
model = FinalNanoporeNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# OneCycleLR is great for jumping out of that 95% local minimum
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=STEPS, pct_start=0.3)
loss_fn = nn.BCEWithLogitsLoss()

# --- Training Loop ---
for s in range(STEPS):
    tau = max(0.1, 3.0 * (0.9997 ** s))  # Slower decay
    hard = (s > 2500)  # Switch even later! Let it get to 99% accuracy "softly" first

    bits = torch.randint(0, 2, (BATCH_SIZE, BIT_LENGTH)).float().to(device)
    logits, _, _ = model(bits, tau, hard)
    loss = loss_fn(logits, bits)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if s % 500 == 0:
        acc = ((torch.sigmoid(logits) > 0.5) == bits).float().mean().item()
        print(f"Step {s:4d} | Loss: {loss.float()} | Acc: {acc * 100:.1f}% | Tau: {tau:.2f} | Hard: {hard}")


# --- Final Check ---
model.eval()
with torch.no_grad():
    test_bits = torch.randint(0, 2, (1, BIT_LENGTH)).float().to(device)
    logits, dna, signal = model(test_bits, tau=0.1, hard=True)
    preds = torch.sigmoid(logits)
    final_acc = ((preds[0] > 0.5).float() == test_bits[0]).float().mean().item()
    print(test_bits)
    print(dna)
    print(signal)
    print((preds[0] > 0.5).float())
    print(f"FINAL TEST ACCURACY: {final_acc * 100:.2f}%")