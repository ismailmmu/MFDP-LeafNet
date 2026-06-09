# Model Definition with hooks
import torch
from torch import nn
from torchvision.models import resnet152, mobilenet_v2, densenet121

from feature_extractor import FeatureExtraction
from sparse_autoencoder.autoencoder import SparseAutoencoder


class FSL_Network(nn.Module):
    def __init__(self, device, weight_path, n_way:int):
        super(FSL_Network, self).__init__()
        self.feature_extractor = FeatureExtraction(device)
        self.n_way = n_way
        self.embedding_size = 0
        self.device = device
        self.input_dim = self.feature_extractor(torch.rand(1, 3, 224, 224).to(device)).shape[1]
        #self.hidden_dim = 4352
        self.hidden_dim = 2500
        self.output_dim = self.input_dim
        self.sparse_autoencoder = SparseAutoencoder(self.input_dim, self.hidden_dim, self.output_dim).to(device)
        self._get_pretrained_sprse_weight(weight_path)

    def _get_pretrained_sprse_weight(self, weight_path):
        self.sparse_autoencoder.load_state_dict(torch.load(weight_path))

    def _extract_features(self, x):
        features = self.feature_extractor(x)
        _, encoded = self.sparse_autoencoder(features)
        if self.embedding_size == 0:
            self.embedding_size = encoded.shape[1]
            self.prototypes = nn.Parameter(torch.zeros(self.n_way, self.embedding_size), requires_grad=True).to(self.device)
        return encoded


    # def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor,
    #             query_images: torch.Tensor) -> torch.Tensor:
    #     z_support = self._extract_features(support_images)
    #     z_query = self._extract_features(query_images)
    #
    #     updated_prototypes = torch.zeros_like(self.prototypes)
    #     for label in range(self.n_way):
    #         updated_prototypes[label] = z_support[torch.nonzero(support_labels == label, as_tuple=True)].mean(0)
    #
    #     self.prototypes.data = updated_prototypes.data
    #     dists = torch.cdist(z_query, self.prototypes)
    #     scores = -dists
    #
    #     return scores

    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor,
                query_images: torch.Tensor) -> torch.Tensor:
        z_support = self._extract_features(support_images)
        z_query = self._extract_features(query_images)

        updated_prototypes = torch.zeros_like(self.prototypes)
        for label in range(self.n_way):
            # Compute the new prototype for the current label
            new_prototype = z_support[torch.nonzero(support_labels == label, as_tuple=True)].mean(0)

            # Get the historical prototype for the current label
            historical_prototype = self.prototypes[label]

            # Update the prototype as per the provided equation
            updated_prototypes[label] = 0.5 * (new_prototype + historical_prototype)

        # Update the prototype data
        self.prototypes.data = updated_prototypes.data

        # Calculate distances and return scores
        dists = torch.cdist(z_query, self.prototypes)
        scores = -dists
        return scores




