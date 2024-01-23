import os
import json
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from src.pointNetDataset import ModelNet, RandomRotateTransform, RandomJitterTransform, ScaleTransform
from src.models.pointnet import PointNet

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

def get_model_net_10(datadir, batch_size, num_points, validation_split=0.15):
    transform = transforms.Compose([
        RandomRotateTransform(),
        RandomJitterTransform(),
        ScaleTransform(),
    ])

    # Create ModelNet dataset for train and test
    full_data = ModelNet(datadir, split='train', num_points=num_points, transform=transform)
    test_data = ModelNet(datadir, split='test', num_points=num_points, transform=transform)

    # Calculate sizes for train, validation, and test sets
    total_size = len(full_data)
    validation_size = int(validation_split * total_size)
    train_size = total_size - validation_size

    # Split indices for train and validation sets
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_indices, validation_indices = indices[:train_size], indices[train_size:]

    # Create Subset datasets
    train_subset = Subset(full_data, train_indices)
    validation_subset = Subset(full_data, validation_indices)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, validation_loader, test_loader

loss_func = nn.CrossEntropyLoss()

# Training loop
n_epochs = 50
training_losses = []
training_accuracies = []
validation_losses = []
validation_accuracies = []
best_validation_accuracy = 0.0
best_hyperparameters = {}

for lr in [config['lr'], config['lr']*0.5, config['lr']*0.25]:
    for batch_size in [config['batch_size'], config['batch_size']*2]:
        # Initialize the model with current hyperparameters for each iteration
        net = PointNet(use_dropout=False).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        train_loader, validation_loader, test_loader = get_model_net_10(
            metadata_path, batch_size=batch_size, num_points=1024
        )
        for epoch in range(n_epochs):
            net.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.long().to(device)


                optimizer.zero_grad()

                outputs = net(inputs)
                outputs = outputs.float()
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

            # Validation loop (after training epoch)
            validation_loss = 0.0
            correct_val = 0
            total_val = 0
            net.eval()
            with torch.no_grad():
                for data in validation_loader:
                    inputs_val, labels_val = data
                    inputs_val, labels_val = inputs_val.to(device),labels_val.long().to(device) 
                    outputs_val = net(inputs_val)
                    outputs_val = outputs_val.float()

                    val_loss = loss_func(outputs_val, labels_val)
                    validation_loss += val_loss.item()

                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()

                average_validation_loss = validation_loss / len(validation_loader)
                validation_losses.append(average_validation_loss)

                validation_accuracy = 100 * correct_val / total_val
                validation_accuracies.append(validation_accuracy)

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_hyperparameters = {'lr': lr, 'batch_size': batch_size}

                print(f'Epoch [{epoch + 1}/{n_epochs}] '
                      f'Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%, '
                      f'Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%')

# Use best hyperparameters for final testing
net = PointNet(use_dropout=False).to(device)
optimizer = optim.Adam(net.parameters(), lr=best_hyperparameters['lr'])
train_loader, validation_loader, test_loader = get_model_net_10(
    metadata_path, batch_size=best_hyperparameters['batch_size'], num_points=1024
)

test_losses = []
net.eval()
with torch.no_grad():
    total_test_loss = 0.0
    correct_test = 0
    total_test = 0
    for data in test_loader:
        inputs_test, labels_test = data
        inputs_test, labels_test = inputs_test.to(device), labels_test.long().to(device)
        outputs_test = net(inputs_test)
        outputs_test = outputs_test.float()

        loss_test = loss_func(outputs_test, labels_test)
        total_test_loss += loss_test.item()

        _, predicted_test = torch.max(outputs_test.data, 1)
        total_test += labels_test.size(0)
        correct_test += (predicted_test == labels_test).sum().item()

    average_test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_losses.append(average_test_loss)

    print(f'Final Testing '
          f'Test Loss: {average_test_loss:.4f}, '
          f'Test Accuracy: {test_accuracy:.2f}%')

# Save training metrics
np.savetxt(os.path.join(run_dir, 'training_losses.txt'), np.array(training_losses))
np.savetxt(os.path.join(run_dir, 'training_accuracies.txt'), np.array(training_accuracies))
np.savetxt(os.path.join(run_dir, 'validation_losses.txt'), np.array(validation_losses))
np.savetxt(os.path.join(run_dir, 'validation_accuracies.txt'), np.array(validation_accuracies))
np.savetxt(os.path.join(run_dir, 'test_losses.txt'), np.array(test_losses))
print("Training complete.")

# Save best hyperparameters
with open(os.path.join(run_dir, 'best_hyperparameters.json'), 'w') as f:
    json.dump(best_hyperparameters, f)



