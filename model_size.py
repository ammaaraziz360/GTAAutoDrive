# count_params.py
import torch
from model import StudentPolicy

model = StudentPolicy(in_ch=9, meta_dim=0, aux_tlight=False)  # 3 frames = 9 channels

print("Parameter Count by Component:")
print("-" * 50)

backbone_params = sum(p.numel() for p in model.backbone.parameters())
attn_params = sum(p.numel() for p in model.spatial_attn.parameters())
steer_params = sum(p.numel() for n, p in model.named_parameters() if 'steer' in n)
pedal_params = sum(p.numel() for n, p in model.named_parameters() if 'pedal' in n)

print(f"Backbone (EfficientNet-B0): {backbone_params:,}")
print(f"Spatial Attention:          {attn_params:,}")
print(f"Steering Head:              {steer_params:,}")
print(f"Pedal Head:                 {pedal_params:,}")
print("-" * 50)
print(f"Total:                      {sum(p.numel() for p in model.parameters()):,}")