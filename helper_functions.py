import numpy as np
import pennylane as qml
import torch
from torch import nn
from torch.nn import functional as F

# Discriminator
class Discriminator(nn.Module):

    def __init__(self, n_qubits, d_hidden=(64, 64), dropout=0.0):
        super().__init__()

        self.n_qubits = n_qubits
        self.d_hidden = d_hidden

        # Per-sample encoder φ : R^{n_qubits} → R^{d_hidden[1]}
        self.encoder = nn.Sequential(
            nn.Linear(n_qubits, d_hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden[0], d_hidden[1]),
            nn.GELU(),
        )

        # Head ρ : R^{d_hidden[1]} → R
        self.head = nn.Sequential(
            nn.Linear(d_hidden[1], d_hidden[0] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden[0] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (M, n_qubits) or (B, M, n_qubits)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, M, n_qubits)

        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (M, n_qubits) or (B, M, n_qubits), got {x.shape}")

        B, M, D = x.shape
        if D != self.n_qubits:
            raise ValueError(f"Expected last dim = n_qubits = {self.n_qubits}, got {D}")

        # Encode per sample: (B, M, n_qubits) → (B, M, d_hidden[1])
        z = self.encoder(x)

        # Pool over samples (permutation invariant): (B, M, d_hidden[1]) → (B, d_hidden[1])
        z = z.mean(dim=1)

        # Classify: (B, d_hidden[1]) → (B,)
        logits = self.head(z).squeeze(-1)
        return logits
# Generator
class QuantumBronMachine(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.dev = qml.device("default.qubit", wires = n_qubits, shots = None)

        random_init = torch.randn(n_layers, n_qubits, 3) / 100
        self.weights = nn.Parameter(random_init)
        
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights, wires=np.arange(n_qubits))
            return qml.probs(wires=np.arange(n_qubits))
        
        self.circuit = circuit
    
    def forward(self):
        return self.circuit(self.weights)

# Loss fuction for the generator    
def RBF_kernel(X,  sigmas):
    X = X.view(-1, 1)                
    dist2 = (X - X.T) ** 2            

    sig2 = (sigmas ** 2).view(-1, 1, 1)         
    K = torch.exp(-dist2.unsqueeze(0) / (2.0 * sig2)).mean(dim=0) 
    return K

def MMD_probs(p, q, K):
    return torch.matmul(p, torch.matmul(K, p)) + torch.matmul(q, torch.matmul(K, q)) - 2 * torch.matmul(p, torch.matmul(K, q))

def all_bitstrings(n_qubits: int, device=None, dtype=torch.float32):
    # (2^n, n) table with rows = binary representation of 0..2^n-1
    K = 2 ** n_qubits
    idx = torch.arange(K, device=device)
    bits = ((idx[:, None] >> torch.arange(n_qubits, device=device)) & 1).to(dtype)
    # bits currently least-significant-bit first; reverse if you prefer MSB-first
    bits = torch.flip(bits, dims=[1])
    return bits  # (K, n_qubits)

def sample_bitstrings_gumbel(
    q_or_logits: torch.Tensor,
    n_qubits: int,
    M: int,
    tau: float = 1.0,
    hard: bool = True,
):
    """
    q_or_logits: (K,) with K=2^n, either probs or logits
    returns: x in (M, n_qubits) (hard -> {0,1} forward, soft grads backward)
    """
    device = q_or_logits.device
    dtype = q_or_logits.dtype
    K = 2 ** n_qubits
    assert q_or_logits.shape[-1] == K, f"Expected shape ({K},), got {q_or_logits.shape}"

    # If it's probabilities, convert to logits (stable)
    if torch.all(q_or_logits >= 0) and torch.all(q_or_logits <= 1.0 + 1e-6):
        q = q_or_logits.clamp_min(1e-12)
        logits = torch.log(q)
    else:
        logits = q_or_logits

    logits = logits.unsqueeze(0).expand(M, K)  # (M, K)

    # Differentiable categorical samples (straight-through if hard=True)
    y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1).to(dtype=dtype)  # (M, K)

    # Map one-hot states to bitstrings
    B = all_bitstrings(n_qubits, device=device, dtype=dtype)  # (K, n_qubits)
    x = y @ B  # (M, n_qubits)
    return x