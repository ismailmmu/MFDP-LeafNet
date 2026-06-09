import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152, mobilenet_v2, densenet121
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class FeatureExtraction(nn.Module):
    def __init__(self, device):
        super(FeatureExtraction, self).__init__()
        self.device = device

        # Load pre-trained models
        self.resnet = resnet152(pretrained=True).to(device)
        self.resnet.fc = nn.Identity()  # Removing final FC layer

        self.mobile = mobilenet_v2(pretrained=True).to(device)
        self.mobile.classifier = nn.Identity()  # Removing final FC layer

        self.dense = densenet121(pretrained=True).to(device)
        self.dense.classifier = nn.Identity()  # Removing final FC layer

        # Hook variables for gradients and activations
        self.res_gradients = None
        self.mobile_gradients = None
        self.dense_gradients = None

        self.res_activations = None
        self.mobile_activations = None
        self.dense_activations = None

    def save_res_gradients(self, grad):
        self.res_gradients = grad

    def save_mobile_gradients(self, grad):
        self.mobile_gradients = grad

    def save_dense_gradients(self, grad):
        self.dense_gradients = grad

    def forward(self, x):
        res_features = torch.flatten(self.resnet(x), start_dim=1)
        mobile_features = torch.flatten(self.mobile(x), start_dim=1)
        dense_features = torch.flatten(self.dense(x), start_dim=1)

        concatenated_features = torch.cat((res_features, mobile_features, dense_features), dim=1)
        return concatenated_features

    def register_hooks(self):
        # Register forward and backward hooks for each model
        self.resnet.layer4[-1].register_forward_hook(self.forward_res_hook)
        self.resnet.layer4[-1].register_backward_hook(
            lambda module, grad_in, grad_out: self.save_res_gradients(grad_out[0]))

        self.mobile.features[-1].register_forward_hook(self.forward_mobile_hook)
        self.mobile.features[-1].register_backward_hook(
            lambda module, grad_in, grad_out: self.save_mobile_gradients(grad_out[0]))

        self.dense.features[-1].register_forward_hook(self.forward_dense_hook)
        self.dense.features[-1].register_backward_hook(
            lambda module, grad_in, grad_out: self.save_dense_gradients(grad_out[0]))

    def forward_res_hook(self, module, input, output):
        self.res_activations = output

    def forward_mobile_hook(self, module, input, output):
        self.mobile_activations = output

    def forward_dense_hook(self, module, input, output):
        self.dense_activations = output

    def get_combined_gradcam_heatmap(self, x):
        # Perform forward pass to register activations
        self.forward(x)

        # Backward pass to get gradients for each model
        self.zero_grad()
        output = self.forward(x)
        target_class = output[0].max(0)[1]
        output[:, target_class].backward()

        # Compute Grad-CAM for ResNet
        res_heatmap = self.compute_gradcam(self.res_activations, self.res_gradients)

        # Compute Grad-CAM for MobileNet
        mobile_heatmap = self.compute_gradcam(self.mobile_activations, self.mobile_gradients)

        # Compute Grad-CAM for DenseNet
        dense_heatmap = self.compute_gradcam(self.dense_activations, self.dense_gradients)

        # Combine the heatmaps (e.g., average)
        combined_heatmap = (res_heatmap + mobile_heatmap + dense_heatmap) / 3

        return combined_heatmap

    def compute_gradcam(self, activations, gradients):
        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weight the activations by the pooled gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Compute the heatmap by taking the mean of the weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()

        # Normalize and resize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap


# Load and preprocess your image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return input_tensor


# Function to visualize heatmap
def apply_heatmap_on_image(heatmap, image_path):
    # Convert to heatmap format
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Load the original image and apply the heatmap on top
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Show or save the image
    cv2.imwrite('combined_gradcam_output.jpg', superimposed_img)
    cv2.imshow('Grad-CAM', superimposed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main block to run the code
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    model = FeatureExtraction(device).to(device)

    # Register hooks to capture gradients and activations
    model.register_hooks()

    # Preprocess image
    image_path = "C:/Flavia/3/1123.jpg"
    input_tensor = preprocess_image(image_path).to(device)

    # Generate combined Grad-CAM heatmap
    combined_heatmap = model.get_combined_gradcam_heatmap(input_tensor)

    # Apply combined heatmap on image
    apply_heatmap_on_image(combined_heatmap, image_path)
