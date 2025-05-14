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
# 🧠 Model 01: TSMixer1D (All-MLP Time Series Mixer)
# ==============================================================================
class TSMixerBlock(nn.Module):
    def __init__(self, d_model, input_size, dropout_rate=0.5):
        super(TSMixerBlock, self).__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(input_size),  # LayerNorm over L
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, input_size)
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(d_model),     # LayerNorm over d_model
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x: [B, L, d_model]
        x = x + self.token_mixing(x.permute(0, 2, 1)).permute(0, 2, 1)  # Token mixing over L
        x = x + self.channel_mixing(x)  # Channel mixing over d_model
        return x


class TSMixer1D_pt(nn.Module):
    def __init__(self, input_channels=3, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=4, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(TSMixer1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # TSMixer Block
        self.tsmixer = TSMixerBlock(d_model=d_model, input_size=input_size, dropout_rate=dropout_rate)

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

        x = self.tsmixer(x)           # [B, L, d_model]

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
# 🧠 Model 02: PatchTST1D (Patch-based Transformer for Time Series)
# ==============================================================================
class PatchEmbedding1D(nn.Module):
    def __init__(self, input_size, patch_size, d_model):
        super(PatchEmbedding1D, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(d_model * patch_size, d_model)  # FIXED HERE

    def forward(self, x):
        # x: [B, L, d_model] → assume permuted already
        B, L, D = x.shape
        num_patches = L // self.patch_size
        x = x[:, :num_patches * self.patch_size, :]  # trim extra
        x = x.view(B, num_patches, self.patch_size, D)
        x = x.permute(0, 1, 3, 2).reshape(B, num_patches, D * self.patch_size)
        x = self.proj(x)
        return x

class PatchTST1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, patch_size=16, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(PatchTST1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Patch embedding
        self.patch_embed = PatchEmbedding1D(input_size, patch_size, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(d_model * (input_size // patch_size), 128)
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
        x = self.input_proj(x)            # [B, d_model, L]
        x = x.permute(0, 2, 1)           # [B, L, d_model]

        x = self.patch_embed(x)          # [B, num_patches, d_model]

        x = self.transformer(x)          # [B, num_patches, d_model]

        x = x.permute(0, 2, 1)          # [B, d_model, num_patches]

        x = torch.flatten(x, 1)         # [B, d_model * num_patches]

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
# 🧠 Model 03: Hyena1D (Attention-free Long-Context Model)
# ==============================================================================
# ⚠️ Note: Since true Hyena is complex and based on custom long convolution kernels, 
# this is a simplified imitation of the Hyena idea using grouped dilated convolutions.
# Define Hyena Block (simplified)
class HyenaBlock1D(nn.Module):
    def __init__(self, d_model, dropout_rate=0.5):
        super(HyenaBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        return F.gelu(x + residual)

# Main Hyena1D_pt model
class Hyena1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(Hyena1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Hyena Block
        self.hyena_block = HyenaBlock1D(d_model, dropout_rate)

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
        x = self.input_proj(x)            # [B, d_model, L]

        x = self.hyena_block(x)          # [B, d_model, L]

        x = torch.flatten(x, 1)          # [B, d_model * L]

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
# 🧠 Model 04: SAMformer1D (Sharpness-Aware Transformer for regression)
# ==============================================================================
# Note: This is a simplified variant. True SAM (Sharpness-Aware Minimization) is 
# an optimizer concept, but we emulate its attention-efficient structure and smoother layers.
# Define SAM Block (simplified self-attention + MLP block)
# Define SAM Block (simplified self-attention + MLP block)
class SAMBlock1D(nn.Module):
    def __init__(self, d_model, nhead=4, dropout_rate=0.5):
        super(SAMBlock1D, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

# Main SAMformer1D_pt model
class SAMformer1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(SAMformer1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # SAM Block
        self.sam_block = SAMBlock1D(d_model, nhead=4, dropout_rate=dropout_rate)

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
        x = self.input_proj(x)            # [B, d_model, L]
        x = x.permute(0, 2, 1)           # [B, L, d_model]

        x = self.sam_block(x)            # [B, L, d_model]

        x = x.permute(0, 2, 1)           # [B, d_model, L]

        x = torch.flatten(x, 1)          # [B, d_model * L]

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
# 🧠 Model 05: Autoformer1D
# ==============================================================================
# This simplified version includes trend + residual separation. Real Autoformer 
# includes auto-correlation attention, which is too heavy for fast prototyping.
# Autoformer Trend Decomposition Layer (Trend + Seasonal Components)
# Autoformer Block (series decomposition + trend/seasonal modeling)
# Autoformer Block (series decomposition + trend/seasonal modeling)
class AutoformerBlock1D(nn.Module):
    def __init__(self, d_model, nhead=4, dropout_rate=0.5):
        super(AutoformerBlock1D, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Trend and seasonal decomposition can be modeled here if extended
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

# Main Autoformer1D_pt model
class Autoformer1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5, num_layers=2):
        super(Autoformer1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Stacked Autoformer blocks
        self.layers = nn.Sequential(*[
            AutoformerBlock1D(d_model, nhead=4, dropout_rate=dropout_rate) for _ in range(num_layers)
        ])

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
        x = self.input_proj(x)           # [B, d_model, L]
        x = x.permute(0, 2, 1)          # [B, L, d_model]

        x = self.layers(x)              # [B, L, d_model]

        x = x.permute(0, 2, 1)         # [B, d_model, L]
        x = torch.flatten(x, 1)        # [B, d_model * L]

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
# 🧠 Model 06: TreeDRNet1D (Tree-Structured Doubly Residual MLP)
# ==============================================================================
# A novel deep MLP-based model using both feature- and residual-based branching, ideal for structured signal regression.
class TreeDRBlock1D(nn.Module):
    def __init__(self, d_model, num_branches=4, dropout_rate=0.5):
        super(TreeDRBlock1D, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU()
            ) for _ in range(num_branches)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fusion = nn.Conv1d(num_branches * d_model, d_model, kernel_size=1)

    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]  # list of [B, d_model, L]
        out = torch.cat(branch_outs, dim=1)                   # [B, num_branches * d_model, L]
        out = self.dropout(out)
        out = self.fusion(out)                                # [B, d_model, L]
        return out

class TreeDRNet1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5, num_layers=2):
        super(TreeDRNet1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Stacked TreeDR blocks
        self.layers = nn.Sequential(*[
            TreeDRBlock1D(d_model, dropout_rate=dropout_rate) for _ in range(num_layers)
        ])

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
        x = self.input_proj(x)           # [B, d_model, L]
        x = self.layers(x)              # [B, d_model, L]

        x = torch.flatten(x, 1)        # [B, d_model * L]

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
# 🧠 Model 07: Informer1D (Efficient Transformer with Probabilistic Attention)
# ==============================================================================
# Sinusoidal Positional Encoding (from original Transformer/Informer)
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ProbSparse Self-Attention (approximate with standard MultiheadAttention)
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout_rate=0.5):
        super(ProbSparseSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x

# Informer1D Encoder-Only for Classification
class Informer1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5, num_layers=2, nhead=4):
        super(Informer1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Positional Encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=input_size)

        # Encoder layers
        self.layers = nn.ModuleList([
            ProbSparseSelfAttention(d_model, nhead, dropout_rate)
            for _ in range(num_layers)
        ])

        # Adaptive Pooling to collapse sequence dimension
        self.pool = nn.AdaptiveAvgPool1d(1)  # output shape: [B, d_model, 1]

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

        # Initialize weights
        initialize_weights(self, mode=init_mode)

    def forward(self, x):
        # x: [B, C=1, L]
        x = self.input_proj(x)           # [B, d_model, L]
        x = x.permute(0, 2, 1)          # [B, L, d_model]

        x = self.positional_encoding(x) # add positional encoding

        for layer in self.layers:
            x = layer(x)                # [B, L, d_model]

        x = x.permute(0, 2, 1)         # [B, d_model, L]
        x = self.pool(x).squeeze(-1)   # [B, d_model]

        x = F.gelu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        # Outputs
        out_check = torch.sigmoid(self.fc_check(x))         # binary
        out_type = self.fc_type(x)                         # multiclass logits
        out_size_cls = self.fc_size_cls(x)                 # multiclass logits
        out_type_size = self.fc_type_size(x)               # multiclass logits
        out_size_reg = F.softplus(self.fc_size_reg(x))     # regression (positive)

        return out_check, out_type, out_size_cls, out_type_size, out_size_reg

# ==============================================================================
# 🧠 Model 08: LITE1D
# ==============================================================================
# A lightweight version of InceptionTime, optimized for mobile inference and fast signal processing.
class LITELayer(nn.Module):
    def __init__(self, d_model, dropout_rate=0.5):
        super(LITELayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x

class LITE1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5, num_layers=2):
        super(LITE1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # LITE Encoder
        self.layers = nn.Sequential(*[
            LITELayer(d_model, dropout_rate) for _ in range(num_layers)
        ])

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
        x = self.input_proj(x)           # [B, d_model, L]
        x = x.permute(0, 2, 1)          # [B, L, d_model]

        x = self.layers(x)              # [B, L, d_model]

        x = x.permute(0, 2, 1)         # [B, d_model, L]
        x = torch.flatten(x, 1)        # [B, d_model * L]

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
# 🧠 Model 09: SCINet1D
# ==============================================================================
class SCINetBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout_rate=0.5):
        super(SCINetBlock, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x                          # [B, C, L]
        x = F.gelu(self.conv1(x))             # [B, C, L]
        x = self.dropout(x)
        x = self.conv2(x)                     # [B, C, L]
        x = self.dropout(x)
    
        # LayerNorm expects shape [B, L, C]
        x = x.permute(0, 2, 1)                # [B, L, C]
        residual = residual.permute(0, 2, 1)
        x = self.norm(x + residual)
        x = x.permute(0, 2, 1)                # back to [B, C, L]
    
        return x


class SCINet1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_type_classes=3,
                 num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5, num_blocks=2):
        super(SCINet1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # SCINet Blocks
        self.blocks = nn.Sequential(*[
            SCINetBlock(d_model, kernel_size=3, dropout_rate=dropout_rate) for _ in range(num_blocks)
        ])

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
        x = self.input_proj(x)           # [B, d_model, L]
    
        x = self.blocks(x)              # Keep as [B, d_model, L]
    
        x = torch.flatten(x, 1)         # [B, d_model * L]
    
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
# 🧠 Model 10: iTransformer1D
# ==============================================================================
# Instance-wise attention for strong interpretability and efficient signal modeling.
class iTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.5):
        super(iTransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Attention block
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward block
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class iTransformer1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, num_heads=8, num_blocks=4, 
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13, init_mode='xavier', 
                 dropout_rate=0.5):
        super(iTransformer1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # iTransformer Blocks
        self.blocks = nn.Sequential(*[
            iTransformerBlock(d_model, num_heads, dropout_rate=dropout_rate) for _ in range(num_blocks)
        ])

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
        x = self.input_proj(x)           # [B, d_model, L]
        x = x.permute(0, 2, 1)          # [B, L, d_model]

        x = self.blocks(x)              # [B, L, d_model]

        x = x.permute(0, 2, 1)         # [B, d_model, L]
        x = torch.flatten(x, 1)        # [B, d_model * L]

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
# 🧠 Model 11: ViT (Vision Transformer)
# ==============================================================================
# ViT Patch Embedding for 1D signals
class ViT1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, patch_size=16, num_heads=8, num_blocks=4,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(ViT1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Patch embedding
        self.patch_embed = PatchEmbedding1D(input_size, patch_size, d_model)

        # ViT Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        # Fully Connected Layers
        self.fc1 = nn.Linear(d_model * (input_size // patch_size), 128)
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
        x = self.input_proj(x)           # [B, d_model, L]
        x = x.permute(0, 2, 1)          # [B, L, d_model]

        x = self.patch_embed(x)          # [B, num_patches, d_model]

        x = self.transformer(x)          # [B, num_patches, d_model]

        x = x.permute(0, 2, 1)          # [B, d_model, num_patches]
        x = torch.flatten(x, 1)         # [B, d_model * num_patches]

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
# 🧠 Model 12: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# ==============================================================================
# Patch embedding for 1D signals
class SwinTransformer1D_pt(nn.Module):
    def __init__(self, input_channels=1, input_size=1024, d_model=64, patch_size=16, num_heads=8, num_blocks=4,
                 num_type_classes=3, num_size_classes=7, num_type_size_classes=13, init_mode='xavier', dropout_rate=0.5):
        super(SwinTransformer1D_pt, self).__init__()

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Patch embedding
        self.patch_embed = PatchEmbedding1D(input_size, patch_size, d_model)

        # Swin Transformer Encoder
        self.swin_block = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout_rate, batch_first=True)
            for _ in range(num_blocks)
        ])

        # Fully Connected Layers
        self.fc1 = nn.Linear(d_model * (input_size // patch_size), 128)
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
        x = self.input_proj(x)           # [B, d_model, L]
        x = x.permute(0, 2, 1)          # [B, L, d_model]

        x = self.patch_embed(x)          # [B, num_patches, d_model]

        for swin_layer in self.swin_block:
            x = swin_layer(x)           # [B, num_patches, d_model]

        x = x.permute(0, 2, 1)          # [B, d_model, num_patches]
        x = torch.flatten(x, 1)         # [B, d_model * num_patches]

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
