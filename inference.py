import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import our model
from swin_u2_matte.models.network import SwinU2Matte

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 768

# Directories
INPUT_DIR = "./data/test_images"
OUTPUT_DIR = "./results"
CHECKPOINT_ROOT = "./checkpoints/run_20251118_111711"

def get_model_path():
    """
    Smart model selector:
    1. Looks for './checkpoints/best_model.pth' (Your manual selection).
    2. If not found, looks for the latest run in './checkpoints/run_*/'.
    """
    # 1. Priority: Manually placed best model
    manual_best = os.path.join(CHECKPOINT_ROOT, "best_model.pth")
    if os.path.exists(manual_best):
        print(f"Found manually selected model: {manual_best}")
        return manual_best
    
    # 2. Fallback: Search for latest run
    print(f"'{manual_best}' not found. Searching for latest training run...")
    runs = glob.glob(os.path.join(CHECKPOINT_ROOT, "run_*"))
    if not runs:
        return None
    
    # Sort by time (latest first)
    runs.sort(key=os.path.getmtime, reverse=True)
    
    for run in runs:
        candidate = os.path.join(run, "best_model.pth")
        if os.path.exists(candidate):
            print(f"Found latest run model: {candidate}")
            return candidate
            
    return None

def load_model(path):
    print(f"Loading model weights...")
    model = SwinU2Matte(in_ch=3, out_ch=1).to(DEVICE)
    
    # Load weights
    checkpoint = torch.load(path, map_location=DEVICE)
    # Handle case where checkpoint saves 'model' state_dict nested or direct
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def preprocess(image_path):
    """Reads image, keeps original size, and normalizes."""
    img_pil = Image.open(image_path).convert("RGB")
    w, h = img_pil.size
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    return img_pil, img_tensor

def save_results(img_pil, pred_mask, filename):
    """Saves the mask and the transparent composite image."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name_no_ext = os.path.splitext(filename)[0]

    # Resize mask back to original image size
    pred_mask = F.interpolate(pred_mask, size=img_pil.size[::-1], mode='bilinear', align_corners=False)
    pred_mask = torch.sigmoid(pred_mask)
    
    # Convert to numpy (0-255)
    mask_np = pred_mask.squeeze().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)

    # Save Mask (Grayscale)
    mask_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_mask.png")
    cv2.imwrite(mask_path, mask_np)

    # Create Transparent Image (Matting)
    img_np = np.array(img_pil)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    b, g, r = cv2.split(img_np)
    rgba = [b, g, r, mask_np] # Use mask as Alpha channel
    dst = cv2.merge(rgba, 4)

    # Save Matte (PNG)
    matte_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_matte.png")
    cv2.imwrite(matte_path, dst)

def main():
    # 1. Find Model
    model_path = get_model_path()
    if model_path is None:
        print(f"❌ Error: No model found in {CHECKPOINT_ROOT}. Please put 'best_model.pth' there.")
        return

    # 2. Load Model
    model = load_model(model_path)

    # 3. Get Images
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*"))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_exts]

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} images. Starting inference...")

    # 4. Inference Loop
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            filename = os.path.basename(img_path)
            
            # Preprocess
            img_pil, input_tensor = preprocess(img_path)
            
            # Forward
            outputs = model(input_tensor)
            d0 = outputs[0] # Final output
            
            # Save
            save_results(img_pil, d0, filename)

    print(f"\n✅ Done! Results saved in: {OUTPUT_DIR}")
    print("Go check your transparent PNGs!")

if __name__ == "__main__":
    main()