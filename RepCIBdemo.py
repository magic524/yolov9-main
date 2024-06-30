import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

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
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and expansion."""
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
    original_size = image.size
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image, original_size

def save_feature_maps(feature_maps, layer_name, save_dir, input_size, max_maps=3):
    num_maps = min(feature_maps.size(1), max_maps)
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_maps):
        plt.figure(figsize=(input_size[0] / 100, input_size[1] / 100))
        feature_map_resized = transforms.functional.resize(
            transforms.ToPILImage()(feature_maps[0, i].detach().cpu()), (input_size[1], input_size[0])
        )
        plt.imshow(feature_map_resized, cmap='viridis')
        plt.axis('off')
        save_path = os.path.join(save_dir, f"{layer_name}_feature_map_{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# Example usage
image_path = 'D:\\pythondata\\gitYOLO\\yolov9-main\\data\\images\\9999951_00000_d_0000020.jpg'  # Replace with your image path
image, original_size = load_image(image_path)

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

# Directory to save feature maps
save_dir = 'D:\\pythondata\\gitYOLO\\yolov9-main\\runs\\RepCIBdemo\\4'

# Save all collected feature maps
for i, (layer, feature_maps) in enumerate(layer_outputs):
    layer_name = f'Layer_{i+1}_{layer.__class__.__name__}'
    save_feature_maps(feature_maps, layer_name, save_dir, original_size)

# Save final output feature maps
save_feature_maps(output, 'Final_Output', save_dir, original_size)
