import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from sklearn import neighbors


class RNN(nn.Module):
    def __init__(self, input_length):
        super(MyModel, self).__init__()
        self.rnn_nodes = 1024
        self.rnn_layers = 12
        
        # Define the GRU layers
        self.gru_layers = nn.ModuleList()
        for _ in range(self.rnn_layers):
            self.gru_layers.append(nn.GRUCell(self.rnn_nodes, self.rnn_nodes))
        
        # Output layer
        self.dense = nn.Linear(self.rnn_nodes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_length, features = x.size()
        h = torch.zeros(batch_size, self.rnn_nodes).to(x.device)

        # Iterate through time steps
        for t in range(seq_length):
            for gru_layer in self.gru_layers:
                h = gru_layer(x[:, t, :], h)

        # Final output
        out = self.dense(h)
        out = self.sigmoid(out)
        return out

def running_scatter(x, y, N):
    rn = np.zeros(N)
    xs = np.linspace(min(x), max(x), num=N)

    for i in range(1, N - 1):
        check = np.where((x >= xs[i - 1]) & (x <= xs[i + 1]))[0]
        if len(check) > 1:
            q75, q25 = np.percentile(y[check], [75, 25])
            rn[i] = abs(q75 - q25)
    
    return rn

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def gen_chan(mag, phase, knn, N):
    asort = np.argsort(phase)
    mag = mag[asort]
    phase = phase[asort]

    # Remove NaN values
    mask = ~np.isnan(mag) & ~np.isnan(phase)
    mag = mag[mask]
    phase = phase[mask]
    knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(phase.min(), phase.max(), num=N)[:, np.newaxis])
    rn = running_scatter(phase, mag, N)
    delta_phase = np.diff(phase)
    delta_phase = np.insert(delta_phase, 0, phase[0])  # Insert the first phase value
    return mag, knn_m, rn, phase, smooth(knn_m, int(N/20)), smooth(knn_m, int(N/5)), delta_phase



class AstronomyDataset(Dataset):
    def __init__(self, root_dir, input_length):
        self.root_dir = root_dir
        self.input_length = input_length
        self.knn_N = int(self.input_length / 20)
        self.knn = neighbors.KNeighborsRegressor(knn_N, weights='distance')
        self.file_list = []
        
        # Iterate through subdirectories (0 and 1) to collect file paths and labels
        for label in [0,1]:
            label_dir = root_dir + '/' + str(label) + '/')
            if os.path.isdir(label_dir):
                label_files = os.listdir(label_dir)
                self.file_list.extend([(os.path.join(label_dir, file), int(label)) for file in label_files])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        
        # Load the CSV file as a Pandas DataFrame
        data_frame = pd.read_csv(file_path)

        # Extract 'mag' and 'phase' columns as NumPy arrays
        mag = data_frame['mag'].values
        phase = data_frame['phase'].values

        # Extract the results as needed
        mag, knn_m, rn, phase, knn_m_smooth20, knn_m_smooth5, delta_phase = gen_chan(mag, phase, knn, N)
        x = np.vstack((mag, knn_m, rn, phase, knn_m_smooth20, knn_m_smooth5, delta_phase)).T
        x = torch.tensor(x, dtype=torch.float32)
        
        return x, label



def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Adjust the learning rate
    scheduler.step(running_loss)
    
    print(f'Train Epoch: {epoch}, Loss: {running_loss / len(train_loader)}')

def validate(model, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
    
    print(f'Validation Epoch: {epoch}, Loss: {val_loss / len(val_loader)}')



# Instantiate the model
input_length = 200# You should specify the input length here

model = RNN(input_length)

# Specify the path to your dataset folder
dataset_root = '/path/to/your/dataset'

# Create an instance of your custom dataset for the entire dataset
full_dataset = AstronomyDataset(dataset_root, input_length)

# Split the dataset into training and validation sets
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoader instances for training and validation
batch_size = 32  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 50  # You can adjust the number of epochs
for epoch in range(1, num_epochs + 1):
    train(model, train_loader, optimizer, criterion, epoch)
    validate(model, val_loader, criterion, epoch)

