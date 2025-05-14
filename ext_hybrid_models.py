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
# 🧠 Model 01: HyenaPatchTST1D (Hyena-inspired Patch-based Transformer)
# ==============================================================================
# integrates Hyena's attention-free long-range regression mechanism with Patch-based 
# Transformer architecture, allowing for faster, non-attention-based time-series processing.
class HyenaPatchTST1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, patch_size=16, d_model=64, nhead=4, num_layers=2, 
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.hyena_projection = nn.Conv1d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # x: [B, C, T]
        x = self.hyena_projection(x).permute(0, 2, 1)  # [B, N, D]
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 02: SAMformer1D (SAM-inspired Sharpness-Aware Regression Transformer)
# ==============================================================================
# merges the sharpness-aware model, SAM, with the transformer structure for robust and adaptive regression on time-series data.
class SAMformer1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, patch_size=16, d_model=64, nhead=4, num_layers=2, 
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.sampler_projection = nn.Conv1d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # x: [B, C, T]
        x = self.sampler_projection(x).permute(0, 2, 1)  # [B, N, D]
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 03: Hyena_LSTM (Attention-Free Long-Range Regression + LSTM)
# ==============================================================================
# This model combines Hyena, which efficiently handles long-range regression tasks 
# without attention mechanisms, with LSTM for sequential learning.
# Placeholder for HyenaBlock (Long-range regression without attention)
# Hyena block placeholder for long-range dependency modeling
class HyenaBlock1D(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(HyenaBlock1D, self).__init__()
        self.fc = nn.Linear(input_size, hidden_dim)

    def forward(self, x):
        return self.fc(x)

# Hyena_LSTM1D with multi-output classification and regression heads
class Hyena_LSTM1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 hidden_dim=64, num_layers=2,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.5):
        super(Hyena_LSTM1D_pt, self).__init__()

        # Hyena block (attention-free long-range modeling)
        self.hyena = HyenaBlock1D(input_size, hidden_dim)

        # LSTM block for sequential modeling
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output heads
        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        # Weight initialization
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Hyena and LSTM processing
        x = self.hyena(x)                  # [B, Seq, Hidden]
        x, _ = self.lstm(x)                # [B, Seq, Hidden]
        x = x.mean(dim=1)                  # Global average pooling over sequence

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output predictions
        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 04: PatchTST_CNN_BiGRU (Patch-based Transformer + CNN + BiGRU)
# ==============================================================================
# This model leverages PatchTST for patch-based feature extraction, followed by 
# CNN for feature refinement and BiGRU for sequential learning.
class PatchTST_CNN_BiGRU1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, patch_size=16, 
                 d_model=64, nhead=4, num_layers=2,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13, 
                 init_mode='xavier', dropout_rate=0.5):
        super(PatchTST_CNN_BiGRU1D_pt, self).__init__()

        # Patch embedding (PatchTST-style)
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.linear_proj = nn.Conv1d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # BiGRU block
        self.bigru = nn.GRU(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.bi_out_dim = 64 * 2  # Since BiGRU is bidirectional

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.bi_out_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output heads
        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        # Weight initialization
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # Patch embedding
        x = self.linear_proj(x).permute(0, 2, 1)  # [B, Patches, d_model]
        x = x + self.pos_embedding[:, :x.size(1), :]  # Positional encoding
        x = self.transformer(x)                      # [B, Patches, d_model]

        # CNN processing
        x = self.cnn(x.permute(0, 2, 1))             # [B, 64, Seq]
        x = x.permute(0, 2, 1)                       # [B, Seq, 64]

        # BiGRU
        x, _ = self.bigru(x)                         # [B, Seq, 2*hidden]
        x = x.mean(dim=1)                            # Global average pooling

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output predictions
        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 05: SAMformer_BiGRU1D (Sharpness-Aware Regression Transformer + BiGRU)
# ==============================================================================
# Incorporates SAMformer for sharpness-aware optimization with BiGRU for sequence modeling.
# Placeholder for SAMformerBlock (Sharpness-Aware Regression Transformer)
class SAMformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(SAMformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

class SAMformer_BiGRU1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, d_model=64, nhead=4, num_layers=2,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 dropout_rate=0.5, init_mode='xavier'):
        super(SAMformer_BiGRU1D_pt, self).__init__()

        # Initial projection: map 1D signals to embedding space
        self.proj = nn.Conv1d(input_channels, d_model, kernel_size=3, stride=1, padding=1)

        # SAMformer (TransformerEncoder)
        self.samformer = SAMformerBlock(d_model=d_model, nhead=nhead, num_layers=num_layers)

        # BiGRU
        self.bigru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.bi_out_dim = d_model * 2

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.bi_out_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output heads
        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.proj(x)                       # [B, d_model, L]
        x = x.permute(0, 2, 1)                 # [B, Seq, d_model]

        x = self.samformer(x)                  # [B, Seq, d_model]
        x, _ = self.bigru(x)                   # [B, Seq, 2*d_model]

        x = x.mean(dim=1)                      # Global Average Pooling over sequence

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
# 🧠 Model 06: Autoformer_ResNet1D (Autoformer + ResNet)
# ==============================================================================
# Merges Autoformer's trend and seasonality decomposition with ResNet's deep residual learning.
# Placeholder for AutoformerBlock (Trend & Seasonality Decomposition)
class AutoformerBlock(nn.Module):
    def __init__(self, d_model):
        super(AutoformerBlock, self).__init__()
        self.fc_trend = nn.Linear(d_model, d_model)
        self.fc_seasonality = nn.Linear(d_model, d_model)

    def forward(self, x):
        trend = F.relu(self.fc_trend(x))
        seasonality = F.relu(self.fc_seasonality(x))
        return trend + seasonality

class ResNetBlock(nn.Module):
    def __init__(self, d_model, num_layers):
        super(ResNetBlock, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = F.relu(layer(x))
        return x + residual

class Autoformer_ResNet1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 d_model=128, num_layers=3, init_mode='xavier', dropout_rate=0.5):
        super(Autoformer_ResNet1D_pt, self).__init__()

        # Input projection to match d_model
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Autoformer block (trend + seasonality)
        self.autoformer = AutoformerBlock(d_model=d_model)

        # ResNet block
        self.resnet = ResNetBlock(d_model=d_model, num_layers=num_layers)

        # Global pooling + dropout
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(d_model, 256)
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
        # x shape: (batch, channels, length)
        x = self.input_proj(x)  # project to d_model channels

        # Permute for Autoformer (batch, length, d_model)
        x = x.permute(0, 2, 1)

        # Apply Autoformer + ResNet blocks
        x = self.autoformer(x)
        x = self.resnet(x)

        # Permute back (batch, d_model, length) for pooling
        x = x.permute(0, 2, 1)

        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output heads
        out_check = torch.sigmoid(self.fc_check(x))
        out_type = self.fc_type(x)
        out_size_cls = self.fc_size_cls(x)
        out_type_size = self.fc_type_size(x)
        out_size_reg = F.softplus(self.fc_size_reg(x))

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 07: TreeDRNet_LSTM1D (Robust Doubly-Residual MLP + LSTM)
# ==============================================================================
# Combines TreeDRNet's robust residual learning with LSTM for sequential modeling.
# Placeholder for TreeDRNetBlock (residual block)
# ==============================================================================
# 🌲 Model 06: TreeDRNet + LSTM
# ==============================================================================
# TreeDRNet captures residual and deep representation patterns; LSTM handles temporal dependencies.

class TreeDRNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(TreeDRNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(self.fc3(x))
        return x + residual  # Residual connection

class TreeDRNet_LSTM1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, hidden_dim=64, num_layers=2, 
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 dropout_rate=0.5, init_mode='xavier'):
        super().__init__()

        # Project input channels to feature dimension
        self.proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        
        # TreeDRNet residual block
        self.tree_drnet = TreeDRNetBlock(hidden_dim, hidden_dim, dropout_rate)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output heads
        self.fc_check = nn.Linear(64, 1)
        self.fc_type = nn.Linear(64, num_type_classes)
        self.fc_size_cls = nn.Linear(64, num_size_classes)
        self.fc_type_size = nn.Linear(64, num_type_size_classes)
        self.fc_size_reg = nn.Linear(64, 1)

        # Weight initialization
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        x = self.proj(x)              # [B, hidden_dim, Seq]
        x = x.permute(0, 2, 1)        # [B, Seq, hidden_dim]
        x = self.tree_drnet(x)        # Residual block
        x, _ = self.lstm(x)           # [B, Seq, hidden_dim]
        x = x.mean(dim=1)             # Global average pooling

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
# 🧠 Model 08: LITE_CNN1D (Lightweight InceptionTime Variant + CNN)
# ==============================================================================
# Merges LITE's lightweight InceptionTime variant with CNN for enhanced feature extraction.

# -------------------------------
# LITE Block for Multi-Kernel Features
# -------------------------------
class LITEBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LITEBlock1D, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels * 3)

    def forward(self, x):
        x1 = F.relu(self.branch1(x))
        x2 = F.relu(self.branch2(x))
        x3 = F.relu(self.branch3(x))
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        return out

# -------------------------------
# LITE_CNN1D_pt Model
# -------------------------------
class LITE_CNN1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, d_model=32,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13,
                 init_mode='xavier', dropout_rate=0.3):
        super(LITE_CNN1D_pt, self).__init__()

        self.input_channels = input_channels

        # LITE Block (multi-scale feature extraction)
        self.lite_block = LITEBlock1D(in_channels=input_channels, out_channels=d_model)

        # First Conv + BN + Pool
        self.conv1 = nn.Conv1d(in_channels=d_model * 3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second Conv + BN + Pool
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Calculate reduced size after two poolings
        reduced_size = input_size // (2 ** 2)

        # Shared Fully Connected Layers
        self.fc1 = nn.Linear(128 * reduced_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
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
        # LITE block
        x = self.lite_block(x)

        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.dropout(F.gelu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.gelu(self.bn_fc2(self.fc2(x))))

        # Output heads
        out_check = torch.sigmoid(self.fc_check(x))               # Binary classification
        out_type = self.fc_type(x)                                # Fault type classification
        out_size_cls = self.fc_size_cls(x)                        # Fault size classification
        out_type_size = self.fc_type_size(x)                      # Combined classification
        out_size_reg = F.softplus(self.fc_size_reg(x))            # Fault size regression

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 09: SCINet_GRU1D (Self-Correlation + GRU)
# ==============================================================================
# Combines SCINet's decomposition and self-correlation with GRU for efficient sequential modeling.
# --- SCINet Block ---
class SCINetBlock1D(nn.Module):
    def __init__(self, channels):
        super(SCINetBlock1D, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return x + F.relu(self.norm(self.conv(x)))

# --- SCINet + GRU Model ---
class SCINet_GRU1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(SCINet_GRU1D_pt, self).__init__()

        self.input_channels = input_channels

        # Project input_channels -> d_model
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # SCINet Block
        self.scinet = SCINetBlock1D(d_model)

        # GRU Layer
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(d_model, 128)
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

        # Weight Initialization
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # x shape: [B, input_channels, 1024]
        x = self.input_proj(x)     # -> [B, d_model, 1024]
        x = self.scinet(x)         # -> [B, d_model, 1024]
        x = x.permute(0, 2, 1)     # -> [B, 1024, d_model] for GRU

        x, _ = self.gru(x)         # -> [B, 1024, d_model]
        x = x[:, -1, :]            # -> [B, d_model]

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
# 🧠 Model 10: iTransformer_CNN1D (Instance-Level Attention Transformer + CNN)
# ==============================================================================
# Combines the instance-level attention mechanism of iTransformer with CNN for enhanced feature extraction.
# -------------------------------
# iTransformer Block
# -------------------------------
class iTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(iTransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, C] where C = d_model (embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

# -------------------------------
# iTransformer_CNN1D_pt Model
# -------------------------------
class iTransformer_CNN1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(iTransformer_CNN1D_pt, self).__init__()

        # Initial projection to d_model for attention (1D conv acts like embedding)
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # iTransformer Block (attention on channels/features)
        self.itransformer = iTransformerBlock(d_model=d_model, nhead=4)

        # CNN on top of iTransformer
        self.cnn = nn.Conv1d(d_model, 64, kernel_size=3, stride=1, padding=1)
        self.bn_cnn = nn.BatchNorm1d(64)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * input_size, 128)
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
        x = x.permute(0, 2, 1)        # [B, L, d_model] for attention

        x = self.itransformer(x)      # [B, L, d_model]
        x = x.permute(0, 2, 1)        # [B, d_model, L] for CNN

        x = self.bn_cnn(self.cnn(x))  # [B, 64, L]

        x = torch.flatten(x, 1)       # [B, 64*L]

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