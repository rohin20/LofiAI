import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import load_midi_files, create_training_data
from model import LSTMModel

sequence_length = 100
batch_size = 64
num_epochs = 100
learning_rate = 0.001
input_size = 128
hidden_size = 256
output_size = 128

sequences = load_midi_files('LofiData')
data, targets = create_training_data(sequences, sequence_length)
data = torch.tensor(data, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)


model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


torch.save(model.state_dict(), 'lstm_model.pth')