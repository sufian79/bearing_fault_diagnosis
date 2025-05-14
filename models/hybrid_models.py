# ============================================================
# 📚 Import Libraries
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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
# 🧠 Hybrid Model 01: CNN_BiGRU
# ==============================================================================
# 🔹 CNN for feature extraction
# 🔹 BiGRU for sequence modeling
# 🔹 Supports: multi-channel input, multi-output (binary/type/size/type+size classification and size regression)
class CNN_BiGRU_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(CNN_BiGRU_pt, self).__init__()

        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # BiGRU for temporal modeling
        self.gru_input_size = input_size // (2 ** 3)  # After 3 pooling layers
        self.bigru = nn.GRU(input_size=128, hidden_size=hidden_size, num_layers=2,
                            batch_first=True, bidirectional=True)

        # Shared Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Output Heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # CNN Path
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # Output shape: [B, 128, T']

        # Prepare for GRU: [B, 128, T'] -> [B, T', 128]
        x = x.permute(0, 2, 1)

        # BiGRU
        x, _ = self.bigru(x)
        x = x.mean(dim=1)  # Global average pooling over time

        # Fully Connected Path
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Output Heads
        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Hybrid Model 02: CNN_BiGRU_Attention
# ==============================================================================
# 🔹 CNN for local feature extraction
# 🔹 BiGRU for temporal modeling
# 🔹 Attention layer for focus on informative timesteps
# 🔹 Supports multi-channel input and multi-output heads
class CNN_BiGRU_Attention_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(CNN_BiGRU_Attention_pt, self).__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Sequence length after 3 pools
        self.seq_len = input_size // (2 ** 3)

        # BiGRU
        self.bigru = nn.GRU(input_size=128, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)

        # Attention Layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Shared FC Layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # CNN Path
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # [B, 128, T']

        # Prepare for GRU: [B, 128, T'] -> [B, T', 128]
        x = x.permute(0, 2, 1)

        # BiGRU
        out, _ = self.bigru(x)  # [B, T, 2*H]

        # Attention mechanism
        attn_weights = torch.softmax(self.attention(out), dim=1)  # [B, T, 1]
        x = torch.sum(attn_weights * out, dim=1)  # [B, 2*H]

        # Fully Connected Layers
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
# 🧠 Hybrid Model 03: ResNet_BiGRU
# ==============================================================================
# 🔹 ResNet for hierarchical local features
# 🔹 BiGRU for sequential dependency modeling
# 🔹 Multi-output heads for classification and regression
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock1D, self).__init__()

        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Downsampling layer
        if downsample:
            self.conv_downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.conv_downsample = None

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Convolutional path
        residual = x  # Save the original input for the skip connection

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # Downsample if required
        if self.downsample:
            residual = self.conv_downsample(residual)

        # Add the residual (skip connection)
        x += residual

        return F.relu(x)  # Apply ReLU after adding the residual
class ResNet_BiGRU_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(ResNet_BiGRU_pt, self).__init__()

        # Residual block 1
        self.layer1 = ResidualBlock1D(in_channels=input_channels, out_channels=32, downsample=False)
        self.layer2 = ResidualBlock1D(in_channels=32, out_channels=64, downsample=True)
        self.layer3 = ResidualBlock1D(in_channels=64, out_channels=128, downsample=True)

        # Calculate sequence length after 2 downsamples
        self.seq_len = input_size // (2 ** 2)  # downsample=True uses stride=2 twice

        # BiGRU
        self.bigru = nn.GRU(input_size=128, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)

        # Shared Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # ResNet Path
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # [B, 128, T']

        # GRU Path [B, T, 128]
        x = x.permute(0, 2, 1)
        x, _ = self.bigru(x)
        x = x.mean(dim=1)  # Global average pooling

        # Fully connected layers
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
# 🧠 Hybrid Model 04: CNN_Transformer
# ==============================================================================
# 🔹 CNN layers for extracting short-range features
# 🔹 Transformer encoder for capturing long-range dependencies
# 🔹 Designed for sequence-aware classification and regression tasks
class CNN_Transformer_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, d_model=128, nhead=8, num_layers=2,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(CNN_Transformer_pt, self).__init__()

        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Project CNN output to Transformer dimension
        self.proj = nn.Conv1d(128, d_model, kernel_size=1)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Shared Fully Connected Layers
        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # CNN path
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # Shape: [B, 128, T']

        # Project to transformer dimension
        x = self.proj(x)  # [B, d_model, T']
        x = x.permute(0, 2, 1)  # [B, T', d_model]

        # Transformer Encoder
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling across time

        # FC path
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
# 🧠 Hybrid Model 05: DenseNet_Transformer
# ==============================================================================
# 🔹 DenseNet for deep local feature reuse
# 🔹 Transformer encoder for long-range temporal modeling
# 🔹 Ideal for detailed fault signature learning
class DenseBlock1D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock1D, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)
            channels += growth_rate  # increase channels after each concat
        self.channels = channels

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

class DenseNet_Transformer_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, growth_rate=32, num_blocks=3, num_layers=4,
                 d_model=128, nhead=8, transformer_layers=2,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(DenseNet_Transformer_pt, self).__init__()

        # Initial Convolution
        self.init_conv = nn.Conv1d(input_channels, growth_rate * 2, kernel_size=7, padding=3)
        channels = growth_rate * 2

        # DenseNet blocks with pooling
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = DenseBlock1D(channels, growth_rate, num_layers)
            self.blocks.append(block)
            channels = block.channels
            self.blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Project to transformer dimension
        self.proj = nn.Conv1d(channels, d_model, kernel_size=1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Shared FC Layers
        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Initial conv
        x = self.init_conv(x)

        # Dense blocks + pooling
        for block in self.blocks:
            x = block(x)

        # Project to Transformer dimension
        x = self.proj(x)  # [B, d_model, T']
        x = x.permute(0, 2, 1)  # [B, T', d_model]

        # Transformer
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global avg pooling

        # Fully connected layers
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
# 🧠 Hybrid Model 06: CNN_LSTM_Attention
# ==============================================================================
# 🔹 CNN layers for robust local features
# 🔹 LSTM to capture long-term dependencies
# 🔹 Attention to highlight salient timesteps
# 🔹 Suitable for regression-focused tasks like fault size prediction
class CNN_LSTM_Attention_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(CNN_LSTM_Attention_pt, self).__init__()

        # CNN Feature Extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Sequence length after 3 pooling layers
        self.seq_len = input_size // (2 ** 3)

        # LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)

        # Attention
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Shared FC Layers
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Output Heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Weight initialization
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # CNN path
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # [B, 128, T']

        # LSTM path
        x = x.permute(0, 2, 1)  # [B, T', 128]
        out, _ = self.lstm(x)  # [B, T', H]

        # Attention mechanism
        attn_weights = torch.softmax(self.attn(out), dim=1)  # [B, T', 1]
        x = torch.sum(attn_weights * out, dim=1)  # [B, H]

        # Fully connected
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
# 🧠 Hybrid Model 07: CNN_BiLSTM_Attention
# ==============================================================================
# 🔹 CNN for local receptive feature extraction
# 🔹 BiLSTM for bidirectional temporal learning
# 🔹 Attention for enhancing fault-specific feature representation
# 🔹 Great fit for noisy industrial signals like CWRU
class CNN_BiLSTM_Attention_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(CNN_BiLSTM_Attention_pt, self).__init__()

        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Sequence length after 3 pooling layers
        self.seq_len = input_size // (2 ** 3)

        # BiLSTM Layer
        self.bilstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                              num_layers=2, batch_first=True, bidirectional=True)

        # Attention Layer
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.fc_check = nn.Linear(128, 1)
        self.fc_type = nn.Linear(128, num_type_classes)
        self.fc_size_cls = nn.Linear(128, num_size_classes)
        self.fc_type_size = nn.Linear(128, num_type_size_classes)
        self.fc_size_reg = nn.Linear(128, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # CNN Path
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # [B, 128, T']

        # Prepare for BiLSTM: [B, 128, T'] -> [B, T', 128]
        x = x.permute(0, 2, 1)

        # BiLSTM
        out, _ = self.bilstm(x)  # [B, T', 2*H]

        # Attention
        attn_weights = torch.softmax(self.attn(out), dim=1)  # [B, T', 1]
        x = torch.sum(attn_weights * out, dim=1)  # [B, 2*H]

        # Fully Connected Layers
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
# 🧠 Hybrid Model 08: DeepCNN_BiGRU
# ==============================================================================
# 🔹 Deep CNN layers for enriched hierarchical features
# 🔹 BiGRU for sequential temporal learning
# 🔹 Combines spatial-temporal depth for enhanced accuracy in fault classification and regression
class DeepCNN_BiGRU_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(DeepCNN_BiGRU_pt, self).__init__()

        # Deep CNN
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate output length after pooling
        self.seq_len = input_size // (2 ** 2)

        # BiGRU
        self.bigru = nn.GRU(input_size=128, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)

        # Shared fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)
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
        # Deep CNN path
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)  # [B, 128, T']

        # Prepare for BiGRU
        x = x.permute(0, 2, 1)  # [B, T', 128]
        x, _ = self.bigru(x)
        x = x.mean(dim=1)

        # Fully connected
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
# 🧠 Hybrid Model 09: DenseCNN_Transformer
# ==============================================================================
class DenseCNN_Transformer_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(DenseCNN_Transformer_pt, self).__init__()

        # Dense CNN
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate output length after pooling
        self.seq_len = input_size // (2 ** 2)

        # Transformer Encoder
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        # Fully connected layers
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
        # Dense CNN path
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)  # [B, 128, T']

        # Transformer Encoder
        x = x.permute(2, 0, 1)  # [T', B, 128]
        x = self.transformer_encoder(x)  # [T', B, 128]
        x = x.mean(dim=0)

        # Fully connected
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
# 🧠 Hybrid Model 10: CNN_GRU_Attention
# ==============================================================================
class CNN_GRU_Attention_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_size=128,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(CNN_GRU_Attention_pt, self).__init__()

        # CNN
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        # GRU
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_size, num_layers=2, 
                          batch_first=True, bidirectional=False)

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 256)
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
        # CNN path
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)  # [B, 128, T']

        # Prepare for GRU
        x = x.permute(0, 2, 1)  # [B, T', 128]
        x, _ = self.gru(x)  # [B, T', hidden_size]

        # Attention mechanism
        attention_weights = self.attention(x)  # [B, T', 1]
        x = (x * attention_weights).sum(dim=1)  # [B, hidden_size]

        # Fully connected
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
# 🧠 Hybrid Model 11: DenseResNet1D 
# ==============================================================================
# Squeeze-and-Excitation Block for attention
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


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


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock1D, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
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

class DenseResNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 growth_rate=32, num_dense_layers=4, num_dense_blocks=2, res_channels=[64, 128], se_reduction=16,
                 init_mode='xavier', dropout_rate=0.3):
        super(DenseResNet1D_pt, self).__init__()

        # Initial convolution layer
        self.init_conv = nn.Conv1d(input_channels, growth_rate * 2, kernel_size=7, padding=3)
        channels = growth_rate * 2

        # Dense blocks
        self.dense_blocks = nn.ModuleList()
        for _ in range(num_dense_blocks):
            dense_block = DenseBlock1D(channels, growth_rate, num_dense_layers)
            self.dense_blocks.append(dense_block)
            channels = dense_block.channels
            self.dense_blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(channels if i == 0 else res_channels[i-1], res_channels[i], downsample=True)
              for i in range(len(res_channels))]
        )
        channels = res_channels[-1]

        # Squeeze-and-Excitation block
        self.se = SEBlock1D(channels, reduction=se_reduction)

        # Global pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        # Shared Dense Layer
        self.fc1 = nn.Linear(channels, 256)
        self.fc2 = nn.Linear(256, 128)

        # Output heads
        self.fc_check = nn.Linear(128, 1)  # Binary classification output
        self.fc_type = nn.Linear(128, num_type_classes)  # Multi-class classification
        self.fc_size_cls = nn.Linear(128, num_size_classes)  # Multi-class classification
        self.fc_type_size = nn.Linear(128, num_type_size_classes)  # Multi-class classification

        # Improved regression branch with Softplus
        self.fc_reg1 = nn.Linear(128, 64)
        self.fc_reg2 = nn.Linear(64, 32)
        self.fc_size_reg = nn.Linear(32, 1)  # Regression output

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)

        # Passing through Dense blocks
        for block in self.dense_blocks:
            x = block(x)

        # Residual blocks + SE block
        x = self.res_blocks(x)
        x = self.se(x)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)

        # Shared dense layer and dropout
        x_shared = self.dropout(F.relu(self.fc1(x)))
        x_common = F.relu(self.fc2(x_shared))

        # Classification outputs
        out_check = torch.sigmoid(self.fc_check(x_common))  # Binary classification output
        out_type = self.fc_type(x_common)  # Multi-class classification
        out_size_cls = self.fc_size_cls(x_common)  # Multi-class classification
        out_type_size = self.fc_type_size(x_common)  # Multi-class classification

        # Regression output with Softplus
        x_reg = F.relu(self.fc_reg1(x_common))
        x_reg = F.relu(self.fc_reg2(x_reg))
        out_size_reg = F.softplus(self.fc_size_reg(x_reg))  # Apply Softplus activation for regression

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg
