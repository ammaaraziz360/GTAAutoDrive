# export_onnx.py
import torch
from model import StudentPolicy

def main():
    ckpt = torch.load("checkpoints/best.pth", map_location="cpu")
    in_ch = 3*(2 if ckpt["args"]["prev_frames"]>0 else 1)
    model = StudentPolicy(in_ch=in_ch, meta_dim=0, aux_tlight=ckpt["args"]["aux_tlight"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    H = W = ckpt["args"]["size"]
    dummy = torch.randn(1, in_ch, H, W)
    torch.onnx.export(
        model, dummy, "policy.onnx",
        input_names=["image"], output_names=["controls" if not ckpt["args"]["aux_tlight"] else "tuple_out"],
        opset_version=17, dynamic_axes={"image": {0: "B"}}
    )
    print("✓ exported policy.onnx")

if __name__ == "__main__":
    main()
