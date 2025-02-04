import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyModule(nn.Module):
    def __init__(self, num_layers=12):
        super().__init__()
        self.num_layers = num_layers
        
    def forward(self, image_features, text_features):
        """
        Args:
            image_features (list):  [B, 50, 512] * 12
            text_features (tensor): [B, 2, 512]
        Returns:
            anomaly_score (tensor): [B]
            anomaly_map (tensor):   [B, H, W]
        """
        B = text_features.shape[0]
        H = W = 7
        
        layer_maps = []
        layer_scores = []
        
        for layer_idx in range(self.num_layers):
            patch_features = image_features[layer_idx][:, 1:, :].float()  # [B, 49, 512]
            global_features = image_features[layer_idx][:, 0, :].float()  # [B, 512]
            
            normal_text = text_features[:, 0]    # [B, 512]
            abnormal_text = text_features[:, 1]  # [B, 512]
            
            normal_sim = F.cosine_similarity(patch_features, normal_text.unsqueeze(1), dim=2)     # [B, 49]
            abnormal_sim = F.cosine_similarity(patch_features, abnormal_text.unsqueeze(1), dim=2) # [B, 49]
            
            curr_map = (abnormal_sim - normal_sim).view(B, H, W)  # [B, 7, 7]
            layer_maps.append(curr_map)
            
            normal_global_sim = F.cosine_similarity(global_features, normal_text, dim=1)    # [B]
            abnormal_global_sim = F.cosine_similarity(global_features, abnormal_text, dim=1) # [B]
            curr_score = abnormal_global_sim - normal_global_sim  # [B]
            layer_scores.append(torch.stack([normal_global_sim, abnormal_global_sim], dim=1))
        
        # Reweighting
        layer_maps = torch.stack(layer_maps, dim=1)  # [B, L, 7, 7]
        layer_weights = F.softmax(layer_maps.mean(dim=[2, 3]), dim=1)  # [B, L]
        anomaly_map = torch.sum(layer_maps * layer_weights.unsqueeze(-1).unsqueeze(-1), dim=1)  # [B, 7, 7]
        
        layer_scores = torch.stack(layer_scores, dim=1)  # [B, L, 2]
        anomaly_score = torch.mean(layer_scores, dim=1)  # [B]
        
        # Upscaling 
        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(1),  # [B, 1, 7, 7]
            size=(224, 224),
            mode='bicubic',
            align_corners=False
        ).squeeze(1)  # [B, 224, 224]
        
        # Normalization
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        #anomaly_score = torch.sigmoid(anomaly_score)
        
        return anomaly_score, anomaly_map