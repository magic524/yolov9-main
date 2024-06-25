import torch
import torch.nn as nn
 
 
class DualConv(nn.Module):
 
    def __init__(self, in_channels, out_channels, stride, g=2):
        """
        Initialize the DualConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param g: the value of G used in DualConv
        """
        super(DualConv, self).__init__()
        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
 
    def forward(self, input_data):
        """
        Define how DualConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        return self.gc(input_data) + self.pwc(input_data)
    
class DualConv2or(nn.Module):
 
    def __init__(self, in_channels, out_channels, stride=1, g=2):
        """
        Initialize the DualConv class.
        
        :param in_channels: int, the number of input channels
        :param out_channels: int, the number of output channels
        :param stride: int, convolution stride, default is 1
        :param g: int, the number of groups for the group convolution, default is 2
        """
        super().__init__()

        if in_channels % g != 0:
            raise ValueError("in_channels must be divisible by the number of groups (g).")
        if out_channels % g != 0:
            raise ValueError("out_channels must be divisible by the number of groups (g).")

        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        self.gc_bn = nn.BatchNorm2d(out_channels)
        
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.pwc_bn = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        """
        Define how DualConv processes the input images or input feature maps.
        
        :param x: torch.Tensor, input images or input feature maps
        :return: torch.Tensor, output feature maps
        """
        gc_out = self.gc_bn(self.gc(x))
        pwc_out = self.pwc_bn(self.pwc(x))
        out = gc_out + pwc_out
        
        return self.relu(out)
 
class DualConv2(nn.Module):
 
    def __init__(self, in_channels, out_channels, stride=1, g=2):
        """
        Initialize the DualConv class.
        
        :param in_channels: int, the number of input channels
        :param out_channels: int, the number of output channels
        :param stride: int, convolution stride, default is 1
        :param g: int, the number of groups for the group convolution, default is 2
        """
        super().__init__()

        if in_channels % g != 0:
            raise ValueError("in_channels must be divisible by the number of groups (g).")
        if out_channels % g != 0:
            raise ValueError("out_channels must be divisible by the number of groups (g).")

        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        self.gc_bn = nn.BatchNorm2d(out_channels)
        
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.pwc_bn = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)

        # Flag to indicate whether to use reparameterized structure
        self.reparameterized = False

    def reparameterize(self):
        """
        Reparameterize the structure by merging the group convolution and pointwise convolution
        into a single convolutional layer.
        """
        with torch.no_grad():
            # Combine weights and biases of gc and pwc
            gc_weight = self.gc.weight
            pwc_weight = self.pwc.weight

            combined_weight = gc_weight + nn.functional.pad(pwc_weight, [1, 1, 1, 1])
            combined_bn_weight = self.gc_bn.weight + self.pwc_bn.weight
            combined_bn_bias = self.gc_bn.bias + self.pwc_bn.bias

            # Create a new convolutional layer with combined weights and biases
            self.reparam_conv = nn.Conv2d(self.gc.in_channels,
                                          self.gc.out_channels,
                                          kernel_size=3,
                                          stride=self.gc.stride,
                                          padding=1,
                                          bias=True)

            self.reparam_conv.weight = nn.Parameter(combined_weight)
            self.reparam_conv.bias = nn.Parameter(combined_bn_bias)

            # Remove original layers
            del self.gc
            del self.gc_bn
            del self.pwc
            del self.pwc_bn

            self.reparameterized = True

    def forward(self, x):
        """
        Define how DualConv processes the input images or input feature maps.
        
        :param x: torch.Tensor, input images or input feature maps
        :return: torch.Tensor, output feature maps
        """
        if self.reparameterized:
            out = self.reparam_conv(x)
        else:
            gc_out = self.gc_bn(self.gc(x))
            pwc_out = self.pwc_bn(self.pwc(x))
            out = gc_out + pwc_out
        
        return self.relu(out)
