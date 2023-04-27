import torch
import torch.nn as nn

class PianoCNN(nn.Module):
    def __init__(self, num_classes):
        super(PianoCNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_velocity = nn.Linear(128 * 16 * 16, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, pitch):
        pitch = pitch.view(-1, 1, 1, 1).float() / 88.0
        pitch = pitch.repeat(1, 1, x.shape[1], x.shape[2])
        x = torch.cat([x.unsqueeze(1), pitch], dim=1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        
        velocity = self.softmax(self.fc_velocity(x))
        return velocity

