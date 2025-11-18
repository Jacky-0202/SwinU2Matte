import torch

def calculate_metrics(logits, targets, threshold=0.5):
    """
    Computes Pixel Accuracy and F1-Score (Dice).
    Args:
        logits: Model output before sigmoid [B, 1, H, W]
        targets: Ground truth mask [B, 1, H, W] (0 or 1)
    Returns:
        accuracy, f1_score
    """
    # Apply sigmoid and threshold
    preds = torch.sigmoid(logits)
    preds = (preds > threshold).float()

    # Flatten
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # Pixel Accuracy
    # (TP + TN) / Total
    correct = (preds_flat == targets_flat).sum()
    total = len(targets_flat)
    accuracy = correct.float() / total

    # F1-Score (Dice Coefficient)
    # 2 * (Intersection) / (Pred + Target)
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()
    
    f1_score = (2. * intersection + 1e-6) / (union + 1e-6)

    return accuracy.item(), f1_score.item()