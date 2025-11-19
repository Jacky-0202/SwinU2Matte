import os
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 1. Import Scheduler
from tqdm import tqdm
from datetime import datetime

# Import our modules
from swin_u2_matte.models.network import SwinU2Matte
from swin_u2_matte.utils.dataloader import get_loaders
from swin_u2_matte.utils.losses import SwinU2Loss
from swin_u2_matte.utils.metrics import calculate_metrics
from swin_u2_matte.utils.logger import get_logger
from swin_u2_matte.utils.plotter import plot_training_curves

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 100
IMG_SIZE = 768
DATA_ROOT = "./data/DIS5K"  # Your dataset path
ROOT_OUTPUT_DIR = "./checkpoints"
# If set to None or the path does not exist, training will start from scratch.
RESUME_PATH = "./checkpoints/swin_u2net.pth"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Train")
    total_loss = 0
    total_acc = 0
    total_f1 = 0
    
    for batch_idx, data in enumerate(loop):
        images = data['image'].to(DEVICE)
        masks = data['mask'].to(DEVICE)

        # Mixed Precision Forward
        with autocast('cuda'):
            outputs = model(images)
            
            # Deep Supervision: Sum loss of all outputs
            loss = 0
            for output in outputs:
                loss += loss_fn(output, masks)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics (Use only final output d0)
        acc, f1 = calculate_metrics(outputs[0], masks)

        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1
        
        loop.set_postfix(loss=loss.item(), acc=acc, f1=f1)

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    avg_f1 = total_f1 / len(loader)

    return avg_loss, avg_acc, avg_f1

def check_accuracy(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_f1 = 0
    loop = tqdm(loader, desc="Val")

    with torch.no_grad():
        for data in loop:
            images = data['image'].to(DEVICE)
            masks = data['mask'].to(DEVICE)

            outputs = model(images)
            d0 = outputs[0] # Final output
            
            loss = loss_fn(d0, masks)
            acc, f1 = calculate_metrics(d0, masks)

            total_loss += loss.item()
            total_acc += acc
            total_f1 += f1
            
            loop.set_postfix(val_loss=loss.item(), val_acc=acc, val_f1=f1)

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    avg_f1 = total_f1 / len(loader)

    return avg_loss, avg_acc, avg_f1

def main():
    # 1. Setup Run Directory (Timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ROOT_OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 2. Setup Logger
    logger = get_logger(run_dir, "training_log.txt")
    logger.info(f"--- Experiment Started: {timestamp} ---")
    logger.info(f"Output Directory: {run_dir}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Hyperparams: Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}")

    # 3. Load Data
    logger.info("Loading data...")
    train_loader, val_loader = get_loaders(
        DATA_ROOT, 
        img_size=IMG_SIZE, 
        batch_size=BATCH_SIZE,
        split_ratio=0.2
    )

    # 4. Initialize Model
    logger.info("Initializing SwinU2Matte...")
    model = SwinU2Matte(in_ch=3, out_ch=1).to(DEVICE)
    
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        logger.info(f"Wait... Loading checkpoint from: {RESUME_PATH}")
        try:
            state_dict = torch.load(RESUME_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            logger.info(">> Checkpoint loaded successfully! Continuing training...")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting from scratch instead.")
    else:
        if RESUME_PATH:
             logger.warning(f"Checkpoint path not found: {RESUME_PATH}")
        logger.info("No checkpoint loaded. Starting from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 2. Initialize Scheduler
    # Reduce LR if validation loss stops improving for 5 epochs
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    loss_fn = SwinU2Loss(bce_w=1.0, iou_w=1.0, ssim_w=1.0).to(DEVICE)
    scaler = GradScaler('cuda')

    # 5. Training Loop
    best_val_loss = float("inf")
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [] 
    }

    logger.info("Start Training...")
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # Train Step
        train_loss, train_acc, train_f1 = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Validation Step
        val_loss, val_acc, val_f1 = check_accuracy(val_loader, model, loss_fn)
        
        # 3. Step Scheduler
        scheduler.step(val_loss)
        
        # 4. Log Current LR
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"LR: {current_lr:.2e}")
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")

        # Update History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Plot Curves
        plot_training_curves(
            history['train_loss'], history['val_loss'],
            history['train_acc'], history['val_acc'],
            history['train_f1'], history['val_f1'],
            run_dir
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Best Model Saved! (Val Loss: {best_val_loss:.4f})")
        
        # Save Last Model
        torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))

    logger.info(f"--- Training Finished. All results saved in {run_dir} ---")

if __name__ == "__main__":
    main()