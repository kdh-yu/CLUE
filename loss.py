import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NCELoss(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, yhat, y):
        pass
    
    
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, eps=1e-6):
        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = torch.where(union > 0, (2 * intersection + eps) / (union + eps), torch.tensor(1.0, device=pred.device))
        
        return (1 - dice).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, yhat, y):
        y_one_hot = F.one_hot(y, num_classes=2).float()
        bce_loss = F.binary_cross_entropy_with_logits(yhat, y_one_hot, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()

def _label(matrix):
    return 1 - torch.all(matrix.view(matrix.shape[0], -1), dim=1).to(torch.int64)

def get_image_label(mask):
    return (mask.view(mask.shape[0], -1).sum(dim=1) > 0).long()#.float()

class Loss(nn.Module):
    def __init__(self, seg_weight=1.0, cls_weight=1.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.seg_weight = seg_weight  # segmentation loss
        self.cls_weight = cls_weight  # classification loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, score, amap, mask):
        """
        Args:
            score (tensor): image-level anomaly score [B]
            amap (tensor): pixel-level anomaly map [B, H, W]
            mask (tensor): ground truth segmentation mask [B, H, W]
        """
        # Image-level classification loss
        image_label = get_image_label(mask)  # [B]
        #print(image_label.shape, score.shape)
        #print(score, image_label)
        cls_loss = self.ce_loss(score, image_label)
        
        # Pixel-level segmentation loss
        seg_loss = self.dice_loss(amap, mask)
        
        # Total loss
        total_loss = self.cls_weight * cls_loss + self.seg_weight * seg_loss
        
        return total_loss
        
