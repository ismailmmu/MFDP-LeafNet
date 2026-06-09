import multiprocessing

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from tqdm import tqdm
from easyfsl.samplers import TaskSampler
from time import time
import numpy as np
from model import FSL_Network
from torch.cuda.amp import autocast
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


def get_labels():
    return [instance[1] for instance in test_set.samples]


def evaluate_on_one_task(support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor,
                         query_labels: torch.Tensor, model) -> (torch.Tensor, torch.Tensor):
    # Move data to GPU
    support_images, support_labels, query_images, query_labels = (
        support_images.cuda(), support_labels.cuda(), query_images.cuda(), query_labels.cuda()
    )

    # Mixed precision inference
    with autocast():
        predictions = torch.max(model(support_images, support_labels, query_images).detach().data, 1)[1]

    return predictions, query_labels


def evaluate(data_loader: DataLoader, model):
    all_predictions = []
    all_true_labels = []

    model.eval()
    start_time = time()

    with torch.no_grad():
        for episode_index, (support_images, support_labels, query_images, query_labels, class_ids) in tqdm(
                enumerate(data_loader), total=len(data_loader)):
            predictions, true_labels = evaluate_on_one_task(support_images, support_labels, query_images, query_labels,
                                                            model)
            predictions = predictions.cpu().numpy()
            true_labels = true_labels.cpu().numpy()
            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)
            del predictions, true_labels

    # Convert lists to numpy arrays for sklearn functions
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Calculate metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_true_labels, all_predictions,
                                                                     average='weighted')
    accuracy = accuracy_score(all_true_labels, all_predictions)

    end_time = time()
    elapsed_time = end_time - start_time

    print(f"Model tested on {len(data_loader)} tasks.")
    print(f"Accuracy: {accuracy * 100:.2f}")
    print(f"Precision: {precision * 100:.2f}")
    print(f"Recall: {recall * 100:.2f}")
    print(f"F1 Score: {f1_score * 100:.2f}")
    print(f"Time: {elapsed_time:.2f} seconds")
    return accuracy * 100


# Image transformation and dataset loading
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # Enable optimized GPU performance

    image_size = 224
    #val_dir = "C:\\Dataset\\plantVillage\\validation_data"
    val_dir = "C:\\Dataset\\EgyptianPlantLeaf"

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_set = ImageFolder(root=val_dir, transform=test_transform)

    N_WAY = 5 #number of classes
    N_SHOTs = [1,5] #
    N_QUERY = 5
    N_EVALUATION_TASKS = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Initialize the ResNet-50 backbone
    # convolutional_network = resnet50(pretrained=True)
    # convolutional_network = convolutional_network.cuda()  # Move ResNet to GPU

    model = FSL_Network(device, weight_path="weights/sparse_auto_encoder2500.pth", n_way=N_WAY).to(
        'cuda')
    # model = FSL_Network(device, weight_path="weights/sparse_auto_encoder2500.pth", n_way=N_WAY).to('cuda')
    # model = GradientBasedPrototypicalNetworks(convolutional_network, n_way=N_WAY, embedding_size=301056).to('cuda')
    test_set.get_labels = get_labels

    num_workers = 1 # Dynamically set num_workers based on CPU cores

    for i in range(len(N_SHOTs)):
        N_SHOT = N_SHOTs[i]
        print(f"================Running on {N_SHOT} shots================================")
        test_sampler = TaskSampler(
            test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
        )
        test_loader = DataLoader(
            test_set,
            batch_sampler=test_sampler,
            num_workers=num_workers,  # Optimized number of workers for data loading
            pin_memory=True,
            collate_fn=test_sampler.episodic_collate_fn,
        )
        sum = 0
        for j in range(5):
            start_time = time()
            sum += evaluate(test_loader, model)
        print(sum/5)
        print("\n=======================================================================================\n")
