# model.py
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2

class StudentPolicy(nn.Module):
    def __init__(self, in_ch=3, meta_dim=0, aux_tlight=False):
        super().__init__()
        base = efficientnet_b2(weights="DEFAULT")
        # adapt first conv if stacking frames (in_ch=6)
        # Cache the original first-conv weights BEFORE we replace the layer so we can copy
        # the pretrained 3-channel weights into the new multi-frame conv.
        first = base.features[0][0]  # Conv2d
        orig_first_w = first.weight.data.clone()  # shape [out_ch, 3, k, k]
        if in_ch != 3:
            base.features[0][0] = nn.Conv2d(in_ch, first.out_channels, kernel_size=first.kernel_size,
                                            stride=first.stride, padding=first.padding, bias=False)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Spatial attention: learn where to look (lanes, cars, etc.)
        # Takes feature maps and outputs a spatial attention mask
        backbone_out_ch = 1408  # EfficientNet-B2 output channels
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(backbone_out_ch, backbone_out_ch // 8, 1),
            nn.SiLU(True),
            nn.Conv2d(backbone_out_ch // 8, 1, 1),
            nn.Sigmoid()
        )
        
        feat = base.classifier[1].in_features + meta_dim  # EfficientNet-B2 final feature dim (1408)
        
        # Decoupled heads: Both need substantial capacity
        # Steering: spatial reasoning (lane curvature, obstacles)
        # Pedals: temporal reasoning (stopping distance, traffic lights, relative speed)
        
        # Steering head with residual connections for better gradient flow
        # UPGRADE: Added LayerNorm and switched to SiLU (Swish) to match EfficientNet
        self.steer_fc1 = nn.Linear(feat, 768)
        self.steer_ln1 = nn.LayerNorm(768)
        self.steer_fc2 = nn.Linear(768, 512)
        self.steer_ln2 = nn.LayerNorm(512)
        self.steer_fc3 = nn.Linear(512, 256)
        self.steer_ln3 = nn.LayerNorm(256)
        
        self.steer_out = nn.Linear(256, 1)
        self.steer_shortcut = nn.Linear(feat, 256)  # skip connection
        self.steer_drop = nn.Dropout(0.3)
        
        # Pedal head (increased capacity for stopping distance and traffic light reasoning)
        self.pedal_head = nn.Sequential(
            nn.Linear(feat, 512), nn.LayerNorm(512), nn.SiLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        self.aux_tlight = aux_tlight
        if aux_tlight:
            self.thead = nn.Sequential(
                nn.Linear(feat, 128), nn.LayerNorm(128), nn.SiLU(True),
                nn.Linear(128, 4)    # red/yellow/green/none
            )
        
        # initialize the new input conv weights for temporal stacks
        if in_ch > 3:
            with torch.no_grad():
                # Use the cached original first-conv weights (3 input channels)
                old_w = orig_first_w  # shape [out_ch, 3, k, k]
                new_w = self.backbone[0][0].weight.data  # shape [out_ch, in_ch, k, k]

                # Explicitly reshape and repeat the weights for each frame
                num_frames = in_ch // 3
                scale = 1.0 / num_frames

                # Scale the weights first
                old_w_scaled = old_w * scale

                # Copy to each temporal position (each gets the pretrained 3-channel filter)
                for frame in range(num_frames):
                    start_ch = frame * 3
                    end_ch = start_ch + 3
                    new_w[:, start_ch:end_ch, :, :] = old_w_scaled.clone()

    def forward(self, x, meta=None):
        # x: [B,C,H,W]
        feat_maps = self.backbone(x)  # [B, 1280, H', W']
        
        # Apply spatial attention to focus on important regions
        attn_mask = self.spatial_attn(feat_maps)  # [B, 1, H', W']
        feat_maps = feat_maps * attn_mask  # element-wise multiply
        
        z = self.pool(feat_maps).flatten(1)
        if meta is not None:
            z = torch.cat([z, meta], 1)
            
        # Steering head with residual connection
        s = self.steer_fc1(z)
        s = self.steer_ln1(s)
        s = F.silu(s)
        s = self.steer_drop(s)
        
        s = self.steer_fc2(s)
        s = self.steer_ln2(s)
        s = F.silu(s)
        s = self.steer_drop(s)
        
        s = self.steer_fc3(s)
        s = self.steer_ln3(s)
        s = F.silu(s)
        
        # Add skip connection from input
        s = s + self.steer_shortcut(z)
        s_out = self.steer_out(s)
        
        p_out = self.pedal_head(z)
        
        steer = torch.tanh(s_out[:, 0])
        thr   = torch.sigmoid(p_out[:, 0])
        brk   = torch.sigmoid(p_out[:, 1])

        if self.aux_tlight:
            tlogits = self.thead(z)
            return torch.stack([steer, thr, brk], 1), tlogits
        return torch.stack([steer, thr, brk], 1)
