import torch
import torch.nn as nn

class PianoTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_classes):
        super(PianoTransformer, self).__init__()
        self.transformer1 = nn.Transformer(
            d_model,
            nhead,
            num_layers,
            dim_feedforward=dim_feedforward 
        )
        self.transformer2 = nn.Transformer(
            d_model,
            nhead,
            num_layers,
            dim_feedforward=dim_feedforward 
        )
        self.linear_pitch = nn.Linear(d_model*112,88)
        self.linear_velocity = nn.Linear(d_model*112, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.transformer1(x,x)
        x1 = x1.view(x.shape[0], -1)
        
        x2 = self.transformer2(x,x)
        x2 = x2.view(x2.shape[0], -1)
        
        pitch = self.softmax(self.linear_pitch(x1))
        velocity = self.softmax(self.linear_velocity(x2))
        return pitch, velocity
