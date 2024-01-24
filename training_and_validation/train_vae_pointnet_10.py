
import os
import json
from datetime import datetime


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.utils.data as torch_data
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim


from src.pointNetDataset import ModelNet, RandomRotateTransform, RandomJitterTransform, ScaleTransform
from src.models.vae_pointnet_10 import PointNet, HybridPointVAE

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
config = {
    "lr": 0.0002,
    "batch_size": 128,
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

# VAE loss function 
def loss_func_vae(output, target, mu, logvar):

    # Compute MSE loss

    bce_loss = F.mse_loss(output, target)
    #bce_loss = F.binary_cross_entropy(output, target)


    # KL Divergence term (
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    total_loss = bce_loss + kl_divergence

    return total_loss


# Instantiate the hybrid model
model = HybridPointVAE(num_classes=10, num_points=1024, use_dropout=True, batch_size=config['batch_size'])

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=config['lr'])
loss_func_cls = nn.CrossEntropyLoss()

# Training loop
num_epochs = 20
training_losses = []
training_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total_loss = 0.0
    total = 0


    # Use tqdm for a progress bar
    for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs, labels = data

         # Move the inputs to the same device as the model
        inputs, labels = inputs.to(device), torch.tensor(labels, dtype=torch.long).to(device)



        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        cls_outputs,vae_outputs,mu, logvar = model(inputs)

        vae_loss = loss_func_vae(vae_outputs, inputs, mu, logvar)


        cls_loss = loss_func_cls(cls_outputs, labels)



        # Total loss is the sum of classification and VAE loss
        #total_loss = cls_loss + vae_loss
        total_loss =  cls_loss


        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()


        _, predicted = torch.max(cls_outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    #print(f"Epoch {epoch + 1}/{num_epochs}")

    training_loss = running_loss / len(train_loader)
    
    training_accuracy = 100 * correct / total
    print(training_accuracy)

    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)

# Save the trained model
#torch.save(model.state_dict(), 'trained_hybrid_model.pth')





    # Inside the evaluation loop
    with torch.no_grad():
        total_test_loss = 0.0
        correct = 0.0
        total = 0.0
        # Initialize total test loss to zero

        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), torch.tensor(labels, dtype=torch.long).to(device)  # Convert labels 

            cls_outputs,vae_outputs,mu, logvar = model(inputs)

            vae_loss = loss_func_vae(vae_outputs, inputs, mu, logvar)


            cls_loss = loss_func_cls(cls_outputs, labels)
              # Total loss is the sum of classification and VAE loss
            #total_loss = cls_loss + vae_loss
            total_loss = cls_loss
            # Accumulate the test loss
            total_test_loss += total_loss.item()


            _, predicted = torch.max(cls_outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        test_accuracy = 100 * correct / total

        test_accuracies.append(test_accuracy)




        # Calculate the average test loss over all batches
        average_test_loss =  total_test_loss / len(test_loader)


        # Append the average test loss to the list
        test_losses.append(average_test_loss)


# Save all training and test metrics in the same folder
np.savetxt(os.path.join(run_dir, 'training_losses.txt'), np.array(training_losses))
np.savetxt(os.path.join(run_dir, 'training_accuracies.txt'), np.array(training_accuracies))
np.savetxt(os.path.join(run_dir, 'test_accuracies.txt'), np.array(test_accuracies))
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
    test_accuracies = np.loadtxt(os.path.join(run_dir, 'test_accuracies.txt'))

    # Plot and save the combined losses
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss', color='#a83655')
    plt.plot(test_losses, label='Test Loss', color='#550f6d')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    losses_plot_path = os.path.join(run_dir, 'combined_losses_plot.png')
    plt.savefig(losses_plot_path)
    plt.grid(True)
    plt.show()
    plt.close()


    # Plot and save the combined accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(training_accuracies, label='Training Accuracy', color='#a83655')
    plt.plot(test_accuracies, label='Test Accuracy', color='#550f6d')


    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    accuracies_plot_path = os.path.join(run_dir, 'accuracies_plot.png')
    plt.savefig(accuracies_plot_path)

    plt.grid(True)
    plt.show()
    plt.close()

    return losses_plot_path, accuracies_plot_path



plot_and_save_metrics(run_dir)