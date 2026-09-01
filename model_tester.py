import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from model import StudentPolicy

def load_frame(frame_num=1, show=True):
    """Load a frame from session-0001 and optionally display it."""
    # Construct frame path
    frame_name = f"frame_{frame_num:06d}.jpg"
    frame_path = os.path.join("data", "frames", "session-0001", frame_name)
    
    if not os.path.exists(frame_path):
        raise ValueError(f"Frame not found: {frame_path}")
    
    # Load image
    img = Image.open(frame_path).convert("RGB")
    
    if show:
        plt.figure(figsize=(12,8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    return img

def process_image(img):
    """Convert PIL image to model input tensor."""
    # Match the preprocessing from dataset.py
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # add batch dimension

def test_model(frame_num=1, show_image=True):
    """Test the model on a specific frame."""
    # Load and show the frame
    img = load_frame(frame_num, show=show_image)
    
    # Preprocess image for model
    img_tensor = process_image(img)
    
    # Load and run model
    model = StudentPolicy()
    model.load_state_dict(torch.load("checkpoints/best.pth")["model"])
    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)
        print(f"\nModel output for frame {frame_num}:")
        print(f"Steering: {output[0,0].item():+.3f}")
        print(f"Throttle: {output[0,1].item():.3f}")
        print(f"Brake:    {output[0,2].item():.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=1, help="frame number to test (1-1403)")
    parser.add_argument("--no-show", action="store_true", help="don't display the image")
    args = parser.parse_args()
    
    try:
        test_model(args.frame, show_image=not args.no_show)
    except Exception as e:
        print(f"Error: {e}")
    