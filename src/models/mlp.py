"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3, prior: float = 0.5):
        super().__init__()
        
        layers = []
        current_dim = in_dim
        
        # 1. Build hidden layers (Linear + Nonlinearity + Dropout)
        for h in hidden:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())  # Nonlinearity
            layers.append(nn.Dropout(dropout))
            current_dim = h
            
        self.features = nn.Sequential(*layers)
        
        # 2. Output layer (1 neuron)
        self.head = nn.Linear(current_dim, 1)
        
        # 3. Initialize the output layer based on the prior.
        # For the sigmoid output to match the prior initially, the raw pre-sigmoid 
        # output must equal the logit of the prior: logit = ln(prior / (1 - prior))
        prior_clamped = max(min(prior, 0.999), 0.001)  # Prevent log(0) or division by zero
        init_bias = math.log(prior_clamped / (1.0 - prior_clamped))
        
        # Zero the weights and set the bias so the initial prediction is exactly the prior
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, init_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through hidden layers
        x = self.features(x)
        
        # Calculate raw (logit) output
        x = self.head(x)
        
        # Constrain to [0, 1] range
        x = torch.sigmoid(x)
        
        # 4. Convert (batch, 1) shape to (batch,)
        return x.squeeze(-1)