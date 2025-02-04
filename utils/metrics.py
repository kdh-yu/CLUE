import torch

def dice_score(pred, target, smooth=1e-5):
    """
    Compute Dice score between prediction and target masks
    Args:
        pred: predicted binary mask
        target: ground truth binary mask
        smooth: smoothing factor to avoid division by zero
    Returns:
        dice score (float)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()
