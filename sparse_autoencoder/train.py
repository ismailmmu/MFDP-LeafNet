import warnings

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet152, mobilenet_v2, densenet121
from tqdm import tqdm

from feature_extractor import FeatureExtraction

warnings.filterwarnings("ignore")
from sparse_autoencoder.autoencoder import SparseAutoencoder, sparse_loss_function



def train_autoencoder(combined_loader, model, feature_extraction, optimizer, sparsity_lambda, device, hidden_dim, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in combined_loader:
            images = batch[0].to(device)  # Extract the features from the dataset
            optimizer.zero_grad()
            features = feature_extraction(images)
            outputs, encoded = model(features)
            loss = sparse_loss_function(outputs, features, encoded, sparsity_lambda)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(combined_loader)}")

    model_save_path = f"../weights/sparse_auto_encoder_resnet18{hidden_dim}.pth"
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # # Load the two datasets using ImageFolder
    dataset1 = datasets.ImageFolder(root='C:\Dataset\Flavia', transform=transform)
    dataset2 = datasets.ImageFolder(root='C:\Dataset\SwedishLeaf', transform=transform)
    combined_dataset = ConcatDataset([dataset1, dataset2],)
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extraction = FeatureExtraction(device)

    sparsity_lambda = 1e-3
    input_dim = feature_extraction(torch.rand(1, 3, 224, 224).to(device)).shape[1]
    hidden_dim = [2500]
    hidden_dim.sort(reverse=True)
    output_dim = input_dim

    for i in range(len(hidden_dim)):
        model = SparseAutoencoder(input_dim, hidden_dim[i], output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_autoencoder(combined_loader, model, feature_extraction, optimizer, sparsity_lambda, device, hidden_dim[i], num_epochs=10)
