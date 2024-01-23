
import os
import json
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from src.pointNetDataset import ModelNet, RandomRotateTransform, RandomJitterTransform, ScaleTransform
from src.models.pointnet_10 import PointNet

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
config = {
    "lr": 1e-4,
    "batch_size": 64,
    "model": {
        "conv1a_out": 64,
        "conv2a_out": 128,
        "conv3a_out": 1024,
    },
}

# Directory for saving logs and checkpoints
run_name = datetime.today().strftime('%Y-%m-%d')
run_dir = os.path.join(os.getcwd(), 'debug_run', run_name)
os.makedirs(run_dir, exist_ok=True)

# Initialize datasets
metadata_path = '/dataNfs/modelnet10/metadata.parquet'


def get_model_net_10(datadir, batch_size, num_points):
    transform = transforms.Compose([
        RandomRotateTransform(),
        RandomJitterTransform(),
        ScaleTransform(),
    ])

    train_data = ModelNet(datadir, split='train', num_points=num_points, transform=transform)
    test_data = ModelNet(datadir, split='test', num_points=num_points, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


train_loader, test_loader = get_model_net_10(metadata_path, batch_size=config['batch_size'], num_points=1024)

# Initialize PointNet model
net = PointNet(use_dropout = False).to(device)

# Loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=config['lr'])

# Training loop
n_epochs = 20
training_losses = []
training_accuracies = []
test_losses = []

for epoch in range(n_epochs):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), torch.tensor(labels, dtype=torch.long).to(device)  # Convert labels to torch.long

        optimizer.zero_grad()

        outputs = net(inputs)
        outputs = outputs.float()  # Ensure outputs have the correct data type
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    training_loss = running_loss / len(train_loader)
    training_accuracy = 100 * correct / total

    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)

    # Evaluate on the test set
    net.eval()
    correct = 0
    total = 0

      # Inside the evaluation loop
    with torch.no_grad():
        total_test_loss = 0.0  # Initialize total test loss to zero

        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), torch.tensor(labels, dtype=torch.long).to(device)  # Convert labels to torch.long
            outputs = net(inputs)
            outputs = outputs.float()  # Ensure outputs have the correct data type

            # Calculate the test loss for this batch
            loss = loss_func(outputs, labels)

            # Accumulate the test loss
            total_test_loss += loss.item()

        # Calculate the average test loss over all batches
        average_test_loss = total_test_loss / len(test_loader)

        # Append the average test loss to the list
        test_losses.append(average_test_loss)


     # Print training and test metrics at the end of each epoch
    print(f'Epoch [{epoch + 1}/{n_epochs}] '
          f'Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%, '
          f'Test Loss: {average_test_loss:.4f}')

# Save training metrics

np.savetxt(os.path.join(run_dir, 'training_losses.txt'), np.array(training_losses))
np.savetxt(os.path.join(run_dir, 'training_accuracies.txt'), np.array(training_accuracies))
np.savetxt(os.path.join(run_dir, 'test_losses.txt'), np.array(test_losses))

print("Training complete.")


# Save configuration
with open(os.path.join(run_dir, 'config_pointnet.json'), 'w') as f:
    json.dump(config, f)




def plot_and_save_metrics(run_dir):
    # Load training and test losses and accuracies from text files
    training_losses = np.loadtxt(os.path.join(run_dir, 'training_losses.txt'))
    test_losses = np.loadtxt(os.path.join(run_dir, 'test_losses.txt'))
    training_accuracies = np.loadtxt(os.path.join(run_dir, 'training_accuracies.txt'))
  
    # Plot and save the combined losses
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    losses_plot_path = os.path.join(run_dir, 'combined_losses_plot.png')
    plt.savefig(losses_plot_path)
    plt.close()

    # Plot and save the combined accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(training_accuracies, label='Training Accuracy', color='green')
   
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    accuracies_plot_path = os.path.join(run_dir, 'accuracies_plot.png')
    plt.savefig(accuracies_plot_path)
    plt.close()

    return losses_plot_path, accuracies_plot_path

plot_and_save_metrics(run_dir)
