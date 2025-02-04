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
        dice = (2 * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)
        return (1 - dice).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, yhat, y):
        yhat = yhat.view(yhat.shape[0], -1)
        y = y.view(y.shape[0], -1)
        nll_probs = -torch.log(yhat + 1e-8)
        focal_loss = self.alpha * (1 - yhat)**self.gamma * nll_probs
        return focal_loss.mean()

def _label(matrix):
    return 1 - torch.all(matrix.view(matrix.shape[0], -1), dim=1).to(torch.int64)

def get_image_label(mask):
    """
    mask에서 image-level label 생성
    mask에 1이 하나라도 있으면 비정상(1), 아니면 정상(0)
    """
    return (mask.view(mask.shape[0], -1).sum(dim=1) > 0).long()

class Loss(nn.Module):
    def __init__(self, seg_weight=1.0, cls_weight=1.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.seg_weight = seg_weight  # segmentation loss 가중치
        self.cls_weight = cls_weight  # classification loss 가중치
        self.ce_loss = nn.CrossEntropyLoss()
        
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
        
