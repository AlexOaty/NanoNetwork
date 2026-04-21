import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_bits):
        super().__init__()
        self.seq_len = int(input_bits *0.75)
        self.net = nn.Sequential(
            nn.Linear(input_bits, 128),
            nn.ReLU(),
            nn.Linear(128, self.seq_len * 4)
        )

    def forward(self, x):
        # x shape: (batch_size, input_bits)
        out = self.net(x)  # (batch_size, seq_len*4)
        out = out.view(-1, self.seq_len, 4)  # (batch_size, seq_len, 4)
        probs = F.gumbel_softmax(out, tau=1.0, hard=True, dim=-1)
        return probs

class Decoder(nn.Module):
    def __init__(self, output_bits, seq_len):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * 4, 128), nn.ReLU(), nn.Linear(128, output_bits), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class SimpleDNAAutoencoder(nn.Module):
    def __init__(self, input_bits):
        super().__init__()
        self.seq_len = int(input_bits *0.75)  # 2 bits per base
        self.encoder = Encoder(input_bits)
        self.decoder = Decoder(input_bits, self.seq_len)

    def forward(self, x):
        dna = self.encoder(x)
        out = self.decoder(dna)
        return out


# ---- Training ----
# tiny example
input_bits = 4
model = SimpleDNAAutoencoder(input_bits)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

model = SimpleDNAAutoencoder(input_bits)
model.load_state_dict(torch.load("dna_model_4.pth"))
model.eval()
# ---- Test ----
x_test = torch.randint(0, 2, (1, input_bits)).float()
dna = model.encoder(x_test)
print("Input bits: ", x_test)
print("DNA one-hot: ", dna)
recovered_bits = model.decoder(dna)
print("Recovered bits (probabilities): ", recovered_bits)
print("Recovered bits (0/1): ", (recovered_bits > 0.5).float())







