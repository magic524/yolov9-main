import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define Conv and RepConv classes (simplified versions)
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1):
        super(RepConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Define RepCIB class
class RepCIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, padding=1, groups=c1),
            Conv(c1, 2 * c_, 1),
            RepConv(2 * c_, 2 * c_, 3, padding=1, groups=2 * c_) if not lk else Conv(2 * c_, 2 * c_, 3, padding=1, groups=2 * c_),
            Conv(2 * c_, c2, 1),
            RepConv(c2, c2, 3, padding=1, groups=c2),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

# Load and preprocess an example image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def visualize_feature_maps(feature_maps, title):
    num_maps = feature_maps.size(1)
    fig, axes = plt.subplots(1, min(num_maps, 8), figsize=(15, 15))  # Show up to 8 feature maps
    for i in range(min(num_maps, 8)):
        axes[i].imshow(feature_maps[0, i].detach().cpu().numpy(), cmap='viridis')
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()

# Example usage
image_path = 'D:\\pythondata\\gitYOLO\\yolov9-main\\data\\images\\0000001_05999_d_0000011.jpg'  # Replace with your image path
image = load_image(image_path)

# Define the model
model = RepCIB(c1=3, c2=64, shortcut=True, e=0.5, lk=False)

# Forward pass and visualize feature maps
layer_outputs = []

# Hook to capture intermediate outputs
def hook_fn(module, input, output):
    layer_outputs.append((module, output))

# Register hooks for each layer in the model
for layer in model.cv1:
    layer.register_forward_hook(hook_fn)

# Forward pass
output = model(image)

# Visualize all collected feature maps
for i, (layer, feature_maps) in enumerate(layer_outputs):
    visualize_feature_maps(feature_maps, f'Layer {i+1}: {layer.__class__.__name__}')

# Visualize final output
visualize_feature_maps(output, 'Final Output')
