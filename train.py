import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from models.piano_transformer import PianoTransformer
from models.piano_cnn import PianoCNN

from utils.data import PianoAudioDataset

# Hyperparameters
learning_rate = 1e-5
epochs = 20
batch_size = 4

# Dataset and DataLoader
data_dir = 'data/train'
dataset = PianoAudioDataset(data_dir)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = PianoTransformer(d_model=128, nhead=8, num_layers=6, dim_feedforward=512, num_classes=128).to(device)
model = PianoCNN(num_classes=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



import datetime

# Training loop
model.train()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file_name = f"log/training_{current_time}.txt"

with open(log_file_name, "w") as log_file:
    for epoch in range(epochs):
        for i, (spectrograms, pitches, velocities) in enumerate(dataloader):
            spectrograms = spectrograms.to(device)
            pitches = pitches.to(device)
            velocities = velocities.to(device)

            velocity_pred = model(spectrograms, pitches)
            loss = criterion(velocity_pred, velocities)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % 50 == 0:
                write_string = f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}'
                print(write_string)
                log_file.write(write_string + "\n")
            

