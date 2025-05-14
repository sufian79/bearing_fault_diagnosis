# ============================================================
# 📚 Import Libraries
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

# ==============================================================================
# 🧠 PYTORCH DEEP LEARNING MODEL FUNCTIONS AND CLASSES DEFINITION
# ==============================================================================
# Utilities functions

# Global weight initialization function
def initialize_weights(model, mode='xavier'):
    """
    Applies weight initialization to the given model.

    Args:
        model (nn.Module): Your PyTorch model.
        mode (str): One of ['xavier', 'kaiming', 'orthogonal', 'normal'].
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            if mode == 'xavier':
                init.xavier_uniform_(m.weight)
            elif mode == 'kaiming':
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif mode == 'orthogonal':
                init.orthogonal_(m.weight)
            elif mode == 'normal':
                init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown init mode: {mode}")
            if m.bias is not None:
                init.zeros_(m.bias)

# ==============================================================================
# 🧠 Model 00: MLP1D (Baseline)
# ==============================================================================
class MLP1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super().__init__()

        self.fc1 = nn.Linear(input_channels * input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))
        return out_check, out_type, out_size_cls, out_type_size, out_size_reg
    
# ==============================================================================
# 🧠 Model 01: CNN
# ==============================================================================
class CNN1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(CNN1D_pt, self).__init__()

        self.input_channels = input_channels

        # Convolutional + Pooling Layers with Batch Normalization
        self.conv1 = nn.Conv1d(in_channels=self.input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Compute reduced feature size dynamically
        reduced_size = input_size // (2 ** 3)

        # Shared Fully Connected Layers with Dropout
        self.fc1 = nn.Linear(128 * reduced_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

        # Output Heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Apply convolutional layers with Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten the output for fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout

        # Outputs
        out_check = torch.sigmoid(self.fc_check(x))  # Binary classification
        out_type = self.fc_type(x)                   # Fault type classification
        out_size_cls = self.fc_size_cls(x)           # Fault size classification
        out_type_size = self.fc_type_size(x)         # Combined type & size classification
        out_size_reg = F.softplus(self.fc_size_reg(x))  # Fault size regression with softplus

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 02: ResNet
# ==============================================================================
# Residual block for 1D convolutions
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False):
        super(ResidualBlock1D, self).__init__()
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if downsample or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# ResNet1D with multi-channel input
class ResNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(ResNet1D_pt, self).__init__()

        # Residual layers
        self.layer1 = ResidualBlock1D(in_channels=input_channels, out_channels=32, downsample=False)
        self.layer2 = ResidualBlock1D(in_channels=32, out_channels=64, downsample=True)
        self.layer3 = ResidualBlock1D(in_channels=64, out_channels=128, downsample=True)

        # Global pooling and FC layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

        # Output heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Weight initialization
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 03: BiLSTM
# ==============================================================================
class BiLSTM1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(BiLSTM1D_pt, self).__init__()
        self.sequence_length = input_size
        self.hidden_size = hidden_size
        self.input_channels = input_channels

        # New CNN frontend
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # Reduce temporal dim by half each time

        # Bidirectional LSTM — updated input_size from CNN output channels
        self.bilstm = nn.LSTM(
            input_size=32,             # CNN output channels
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Accept input shape [B, C, T] or [B, T, C]
        if x.ndim == 3:
            if x.shape[1] in [2, 3]:  # Likely [B, C, T]
                pass  # Keep as is for CNN
            elif x.shape[2] in [2, 3]:  # Likely [B, T, C], convert to [B, C, T]
                x = x.permute(0, 2, 1)
            else:
                raise ValueError(f"Expected one dim to be 2 or 3, got shape: {x.shape}")
        else:
            raise ValueError(f"Expected 3D tensor, got shape: {x.shape}")

        x = x[:, :, :self.sequence_length]  # Trim/pad to sequence length

        # CNN Block
        x = self.pool(F.relu(self.conv1(x)))   # [B, 16, T/2]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 32, T/4]

        # Convert to LSTM input: [B, T', C]
        x = x.permute(0, 2, 1)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        x = lstm_out.mean(dim=1)  # Global average pooling

        # Fully connected path
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Output heads
        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 04: DenseNet
# ==============================================================================
class DenseLayer1D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer1D, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], dim=1)

class DenseBlock1D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock1D, self).__init__()
        layers = []
        self.channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer1D(self.channels, growth_rate))
            self.channels += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DenseNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, growth_rate=32,
                 num_blocks=3, num_layers=4, num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):  # Added param
        super(DenseNet1D_pt, self).__init__()
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        # Initial convolution layer to reduce input size (3 channels -> growth_rate*2 channels)
        self.init_conv = nn.Conv1d(input_channels, growth_rate * 2, kernel_size=7, padding=3)
        channels = growth_rate * 2

        # Dense blocks with max-pooling after each block
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = DenseBlock1D(channels, growth_rate, num_layers)
            self.blocks.append(block)
            channels = block.channels
            self.blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers for classification and regression tasks
        self.fc1 = nn.Linear(channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Multi-head outputs
        self.fc_check = nn.Linear(128, 1)                          # Binary classification (fault/no-fault)
        self.fc_type = nn.Linear(128, num_type_classes)            # Fault type classification (logits)
        self.fc_size_cls = nn.Linear(128, num_size_classes)        # Fault size classification (logits)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)  # Joint type & size classification (logits)
        self.fc_size_reg = nn.Linear(128, 1)                       # Fault size regression (continuous)

        # Weight initialization
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Input shape: (B, C, T) -> (B, growth_rate*2, T) after initial convolution
        x = self.init_conv(x)

        # Pass through Dense blocks and max pooling
        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = self.global_pool(x).squeeze(-1)

        # Fully connected layers for feature extraction
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Multi-task output
        out_check = torch.sigmoid(self.fc_check(x))                 # Binary classification
        out_type = self.fc_type(x)                                  # Fault type (logits)
        out_size_cls = self.fc_size_cls(x)                          # Fault size classification (logits)
        out_type_size = self.fc_type_size(x)                        # Type-size classification (logits)
        out_size_reg = F.softplus(self.fc_size_reg(x))              # Fault size regression (non-negative)

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 05: TinyVGG
# ==============================================================================
class TinyVGG1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(TinyVGG1D_pt, self).__init__()

        # Adjusted for 3 input channels
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Compute the output size after 2 pooling layers (input_size // 4)
        self.flattened_size = 64 * (input_size // 4)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Multi-head outputs
        self.fc_check = nn.Linear(128, 1)                          # Binary classification (Fault/no Fault)
        self.fc_type = nn.Linear(128, num_type_classes)             # Fault type classification
        self.fc_size_cls = nn.Linear(128, num_size_classes)         # Fault size classification
        self.fc_type_size = nn.Linear(128, num_type_size_classes)   # Combined type-size classification
        self.fc_size_reg = nn.Linear(128, 1)                        # Fault size regression (non-negative)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Pass through convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Output layers for multi-task learning
        out_check = torch.sigmoid(self.fc_check(x))                 # Binary classification
        out_type = self.fc_type(x)                                  # Fault type (logits)
        out_size_cls = self.fc_size_cls(x)                          # Fault size classification (logits)
        out_type_size = self.fc_type_size(x)                        # Type-size classification (logits)

        # Fault size regression with Softplus to ensure non-negative values
        out_size_reg = F.softplus(self.fc_size_reg(x))              # Fault size regression (non-negative)

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 06: Xception
# ==============================================================================
class Xception1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(Xception1D_pt, self).__init__()

        # Adjust the first convolution to handle 3 input channels
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Depthwise separable convolutions
        self.dwconv1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pwconv1 = nn.Conv1d(64, 128, kernel_size=1)

        self.dwconv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.pwconv2 = nn.Conv1d(128, 256, kernel_size=1)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout_rate)

        # Output layers for multi-task learning
        self.fc_check = nn.Linear(256, 1)              # Binary classification
        self.fc_type = nn.Linear(256, num_type_classes)  # Fault type classification
        self.fc_size_cls = nn.Linear(256, num_size_classes)  # Fault size classification
        self.fc_type_size = nn.Linear(256, num_type_size_classes)  # Combined type-size classification
        self.fc_size_reg = nn.Linear(256, 1)           # Fault size regression

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Apply first convolutional block with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)

        # Apply second convolutional block with ReLU activation
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)

        # Apply depthwise separable convolutions
        x = F.relu(self.dwconv1(x))
        x = F.relu(self.pwconv1(x))

        x = F.relu(self.dwconv2(x))
        x = F.relu(self.pwconv2(x))

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer

        # Apply fully connected layers with ReLU activation
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Output layers
        out_check = torch.sigmoid(self.fc_check(x))                # Binary classification
        out_type = self.fc_type(x)                                 # Fault type (logits)
        out_size_cls = self.fc_size_cls(x)                         # Fault size classification (logits)
        out_type_size = self.fc_type_size(x)                       # Type-size classification (logits)

        # Fault size regression with Softplus (for smooth non-negative output)
        out_size_reg = F.softplus(self.fc_size_reg(x))             # Fault size regression (non-negative)

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 07: MobileNet
# ==============================================================================
class MobileNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(MobileNet1D_pt, self).__init__()

        # Depthwise separable convolution (MobileNet architecture)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1)  # Adjusted to accept 3 channels
        self.dwconv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise1 = nn.Conv1d(32, 64, kernel_size=1)

        # Additional layers
        self.fc1 = nn.Linear(64 * input_size // 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(256, 1)                  # Binary classification (fault check)
        self.fc_type = nn.Linear(256, num_type_classes)    # Fault type classification
        self.fc_size_cls = nn.Linear(256, num_size_classes)  # Fault size classification
        self.fc_type_size = nn.Linear(256, num_type_size_classes)  # Combined type-size classification
        self.fc_size_reg = nn.Linear(256, 1)               # Fault size regression

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Depthwise separable convolution
        x = F.relu(self.conv1(x))  # Convolution with 3 channels
        x = F.relu(self.dwconv1(x))
        x = F.relu(self.pointwise1(x))

        x = F.max_pool1d(x, kernel_size=2)  # Downsampling
        x = x.view(x.size(0), -1)  # Flatten

        # Apply fully connected layers with ReLU activation
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Output heads
        out_check = torch.sigmoid(self.fc_check(x))   # Binary classification
        out_type = self.fc_type(x)                    # Fault type (logits)
        out_size_cls = self.fc_size_cls(x)            # Fault size classification (logits)
        out_type_size = self.fc_type_size(x)          # Type-size classification (logits)

        # Softplus for regression to avoid negative values and improve smoothness
        out_size_reg = F.softplus(self.fc_size_reg(x))  # Fault size regression with Softplus

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 08: EfficientNet
# ==============================================================================
class EfficientNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(EfficientNet1D_pt, self).__init__()

        # EfficientNet-like architecture: depthwise separable convolution
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dwconv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise1 = nn.Conv1d(32, 64, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * input_size // 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(256, 1)                 # Binary classification (fault check)
        self.fc_type = nn.Linear(256, num_type_classes)   # Fault type classification
        self.fc_size_cls = nn.Linear(256, num_size_classes)              # Fault size classification
        self.fc_type_size = nn.Linear(256, num_type_size_classes)  # Combined type-size classification
        self.fc_size_reg = nn.Linear(256, 1)              # Fault size regression

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # EfficientNet-like depthwise separable convolution
        x = F.relu(self.conv1(x))  # Now accepts 3 channels as input
        x = F.relu(self.dwconv1(x))
        x = F.relu(self.pointwise1(x))

        x = F.max_pool1d(x, kernel_size=2)  # Downsampling
        x = x.view(x.size(0), -1)  # Flatten

        # Apply fully connected layers with ReLU activation
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Output heads
        out_check = torch.sigmoid(self.fc_check(x))       # Binary classification (fault check)
        out_type = self.fc_type(x)                        # Fault type (logits)
        out_size_cls = self.fc_size_cls(x)                # Fault size classification (logits)
        out_type_size = self.fc_type_size(x)              # Combined type-size classification (logits)

        # Softplus for regression to avoid negative values
        out_size_reg = F.softplus(self.fc_size_reg(x))    # Fault size regression (non-negative)

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 09: DeepCNN
# ==============================================================================
class DeepCNN1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.3):
        super(DeepCNN1D_pt, self).__init__()

        self.input_channels = input_channels
        self.input_size = input_size

        # First Conv layer with 3 input channels
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        self.flatten_dim = self._get_flatten_dim()

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        # Output heads
        self.fc_check = nn.Linear(128, 1)                        # Binary Classification
        self.fc_type = nn.Linear(128, num_type_classes)         # Multi-Class Classification
        self.fc_size_cls = nn.Linear(128, num_size_classes)     # Fault Size Classification
        self.fc_type_size = nn.Linear(128, num_type_size_classes)  # Combined Type+Size Classification
        self.fc_size_reg = nn.Linear(128, 1)                     # Fault Size Regression

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def _get_flatten_dim(self):
        dummy_input = torch.randn(1, self.input_channels, self.input_size)
        dummy_output = self._forward_conv(dummy_input)
        return dummy_output.view(1, -1).size(1)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with ReLU activation
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))        # Binary classification (fault check)
        out_type = self.fc_type(x)                         # Fault type classification (logits)
        out_size_cls = self.fc_size_cls(x)                 # Fault size classification (logits)
        out_type_size = self.fc_type_size(x)               # Combined type-size classification (logits)

        out_size_reg = F.softplus(self.fc_size_reg(x))     # Fault size regression (non-negative)

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 10: LSTM1D
# ==============================================================================
class LSTM1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128, num_layers=2,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(LSTM1D_pt, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # x shape: [B, C, T] → permute to [B, T, C]
        if x.shape[1] in [2, 3]:
            x = x.permute(0, 2, 1)

        x = x[:, :self.input_size, :]  # ensure sequence length within bounds

        batch_size = x.size(0)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # [B, T, H]

        # Use the last time step output
        x = out[:, -1, :]  # [B, H]

        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))  # Softplus for regression

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 11: GRU1D
# ==============================================================================
class GRU1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=256,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(GRU1D_pt, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # Bidirectional GRU
        self.gru = nn.GRU(input_size=input_channels, hidden_size=hidden_size,
                          num_layers=2, batch_first=True, bidirectional=True)

        # Layer normalization after GRU
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Fully connected layers (input doubled because bidirectional + pooled features)
        self.fc1 = nn.Linear(hidden_size * 4, 128)  # mean + max pooling combined
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(64, 1)  # Binary classification
        self.fc_type = nn.Linear(64, num_type_classes)  # Fault type classification
        self.fc_size_cls = nn.Linear(64, num_size_classes)  # Fault size classification
        self.fc_type_size = nn.Linear(64, num_type_size_classes)  # Combined classification
        self.fc_size_reg = nn.Linear(64, 1)  # Regression for fault size estimation

        # Initialize weights
        initialize_weights(self, init_mode)

    def forward(self, x):
        if x.shape[1] in [2, 3]:
            x = x.permute(0, 2, 1)  # [B, T, C]

        x = x[:, :self.input_size, :]

        out, _ = self.gru(x)  # [B, T, 2 * hidden_size]

        out = self.layer_norm(out)

        # Pool across time dimension
        out_mean = out.mean(dim=1)
        out_max = out.max(dim=1).values

        x = torch.cat([out_mean, out_max], dim=1)  # [B, 4 * hidden_size // 2]

        # Fully connected layers with ReLU
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Outputs
        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 12: FCN1D (Fully Convolutional Network)
# ==============================================================================
class FCN1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(FCN1D_pt, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.block(x)
        x = self.global_pool(x).squeeze(-1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 13: TCN1D (Temporal Convolution Network)
# ==============================================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(TCN1D_pt, self).__init__()
        layers = []
        num_channels = [64, 128, 128]
        kernel_size = 3
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation // 2
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation, padding=padding))
        self.network = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(num_channels[-1], 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 14: WaveNet1D (Dilated Causal Convolutions)
# ==============================================================================
class WaveBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(WaveBlock1D, self).__init__()
        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_residual = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        tanh_out = torch.tanh(self.conv_filter(x))
        sigm_out = torch.sigmoid(self.conv_gate(x))
        out = tanh_out * sigm_out
        res = self.conv_residual(out)
        return x + res


class WaveNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(WaveNet1D_pt, self).__init__()
        self.init_conv = nn.Conv1d(input_channels, 64, kernel_size=1)

        self.wavenet_blocks = nn.Sequential(
            WaveBlock1D(64, 128, dilation=1),
            WaveBlock1D(64, 128, dilation=2),
            WaveBlock1D(64, 128, dilation=4)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.wavenet_blocks(x)
        x = self.global_pool(x).squeeze(-1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 15: Transformer1D
# ==============================================================================

class Transformer1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(Transformer1D_pt, self).__init__()

        # Initial projection to d_model for transformer
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_size, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(d_model * input_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout_rate)

        # Output Heads
        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # x: [B, C=1, L]
        x = self.input_proj(x)        # [B, d_model, L]
        x = x.permute(0, 2, 1)        # [B, L, d_model]

        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding

        x = self.transformer_encoder(x)  # [B, L, d_model]

        x = x.permute(0, 2, 1)        # [B, d_model, L]

        x = torch.flatten(x, 1)       # [B, d_model * L]

        x = F.gelu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        # Outputs
        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 16: ConvNeXt1D
# ==============================================================================
class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(self.pwconv1(x))
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)
        return x + residual


class ConvNeXt1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, dim=64,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super().__init__()

        self.stem = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.blocks = nn.Sequential(*[ConvNeXtBlock1D(dim) for _ in range(4)])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_pool(x).squeeze(-1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))
        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 17: SqueezeNet1D
# ==============================================================================
class FireModule1D(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()
        self.squeeze = nn.Conv1d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv1d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv1d(squeeze_channels, expand_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.expand1x1(x)), F.relu(self.expand3x3(x))], dim=1)


class SqueezeNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            FireModule1D(96, 16, 64),
            FireModule1D(128, 16, 64),
            FireModule1D(128, 32, 128)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))
        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 18: ShuffleNet1D (simplified)
# ==============================================================================
class ShuffleBlock1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return x1 + x2


class ShuffleNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super().__init__()

        self.init_conv = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.block = ShuffleBlock1D(64)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block(x)
        x = self.global_pool(x).squeeze(-1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))
        return out_check, out_type, out_size_cls, out_type_size, out_size_reg


