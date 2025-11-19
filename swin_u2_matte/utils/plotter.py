import os
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, save_dir):
    """
    Plots Loss, Accuracy, and F1-Score in a single combined image (1 row, 3 columns).
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Create a figure with 1 row and 3 columns, setting a wider figure size
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- 1. Plot Loss (Left: axes[0]) ---
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # --- 2. Plot Accuracy (Middle: axes[1]) ---
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc')
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc')
    axes[1].set_title('Pixel Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # --- 3. Plot F1-Score (Right: axes[2]) ---
    axes[2].plot(epochs, train_f1s, 'b-', label='Train F1')
    axes[2].plot(epochs, val_f1s, 'r-', label='Val F1')
    axes[2].set_title('F1-Score (Dice)')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    
    # Save the combined figure
    save_path = os.path.join(save_dir, 'training_metrics_combined.png')
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory