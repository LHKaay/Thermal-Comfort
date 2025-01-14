import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import AutoNCP
from ncps.torch import LTC

class Expert(nn.Module):
    def __init__(self, backbone, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        assert backbone in ["LSTM","LNN", "CNN", "combine"]
        
        if backbone=="LSTM":
            self.backbone = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        elif backbone=="LNN":
            self.wiring = AutoNCP(units=132, output_size=hidden_dim)
            self.backbone = LTC(input_dim, self.wiring, batch_first=True)
        elif backbone=="CNN":
            pass
        elif backbone=="combine":
            if random.randint(0,1)==0:
                self.backbone = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
                print("LSTM")
            else:
                self.wiring = AutoNCP(units=132, output_size=hidden_dim)
                self.backbone = LTC(input_dim, self.wiring, batch_first=True)
                print("LNN")

        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, x):
        x, _ = self.backbone.forward(x)
        x = self.linear(x)

        return x
    
class Gating(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.gate(x)
        x = self.softmax(x)

        return x
    
class MixtureOfExperts(nn.Module):
    def __init__(self, backbone, input_dim, hidden_dim, output_dim, num_experts, k):
        super(MixtureOfExperts, self).__init__()
        self.k = k
        self.gate = Gating(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(backbone, input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        # self.output_layer = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        gating_scores = self.gate(x)

        _, topk_indices = gating_scores.topk(self.k, dim=1, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(1, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = (gating_scores * mask).unsqueeze(2)
        # Normalize the gating scores to sum to 1 across the selected top experts
        # gating_scores = F.normalize(gating_scores, p=1, dim=1).unsqueeze(2)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weighted_expert_outputs = torch.sum(expert_outputs * gating_scores, dim=1)
        # output = self.output_layer(weighted_expert_outputs)

        return weighted_expert_outputs