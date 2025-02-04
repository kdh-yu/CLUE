########## Import ##########
# Torch Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Models & Modules
from CLIP import clip
from utils.transform import transform, transform_mask
import modules
from loss import Loss

# Other Utils
from tqdm import tqdm
from datetime import datetime
import numpy as np
from BraTS import BraTSDataset
from IXI import IXIDataset
from Chest import ChestDataset
import nibabel as nib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
from datetime import datetime
import sys
from utils.metrics import dice_score  # dice score 계산을 위한 함수 추가

########## Define Model ##########
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
clue = modules.CLUE().cuda()
am = modules.AnomalyModule().cuda()

########## Hyperparameters ##########
LEARNING_RATE = 2e-4
EPOCH = 20

optim = torch.optim.AdamW(clue.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
loss_fn = Loss()

########## Train setting ##########
DATASET = sys.argv[1]
SEQUENCE = sys.argv[2]
VIEW = sys.argv[3]

# BraTS dataset
trainset = BraTSDataset(
    base_dir='/home/kdh/code/CLUE/metadata',
    sequence=SEQUENCE,
    view=VIEW,
    train=True,
    transform=[transform, transform_mask]
)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
log = open(f'./log/train_{SEQUENCE}_{VIEW}.txt', 'a')

########## Train ##########
import matplotlib.pyplot as plt
clue.train()
prompt_class = 'brain'
for epoch in range(EPOCH):
    epoch_loss = 0
    epoch_dice = 0
    epoch_acc = 0
    total_batches = 0
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        optim.zero_grad()
        image, label = data
        image = image.to(device)    # (B, 3, 224, 224)
        label = label.to(device)    # (B,)
        
        anomaly_score, anomaly_map = model(image, prompt_class, CLUE=clue, am=am)
        loss = loss_fn(anomaly_score, anomaly_map, label)
        
        pred_mask = (anomaly_map > 0.5).float()
        batch_dice = dice_score(pred_mask, label)
        batch_acc = (pred_mask == label).float().mean()
        
        epoch_loss += loss.item()
        epoch_dice += batch_dice
        epoch_acc += batch_acc
        total_batches += 1
        
        if i % 10 == 0:
            idx = np.random.randint(1, 8)
            plt.figure(figsize=(25, 5), dpi=100)
            plt.subplot(1, 5, 1)
            plt.imshow(image[idx, 0].cpu(), cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 5, 2)
            plt.imshow(label[idx, 0].cpu(), cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 5, 3)
            plt.imshow(image[idx, 0].cpu(), cmap='gray')
            plt.imshow(label[idx, 0].cpu(), cmap='jet', alpha=0.3)
            plt.axis('off')
            
            plt.subplot(1, 5, 4)
            plt.imshow(anomaly_map[idx].detach().cpu(), cmap='jet')
            plt.axis('off')
            
            plt.subplot(1, 5, 5)
            plt.imshow(image[idx, 0].cpu(), cmap='gray')
            plt.imshow(anomaly_map[idx].detach().cpu(), cmap='jet', alpha=0.3)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'./pictures/train_{VIEW}_{SEQUENCE}_{epoch+1}.png')
            
        log.write(f"[{datetime.now().strftime('%X')}]\tEPOCH: {epoch+1}\t"
                    f"Batch: {i}\tLoss: {loss.item():.4f}\t"
                    f"Dice: {batch_dice:.4f}\tAcc: {batch_acc:.4f}\n")
            
        loss.backward()
        optim.step()
    
    avg_loss = epoch_loss / total_batches
    avg_dice = epoch_dice / total_batches
    avg_acc = epoch_acc / total_batches
    
    log.write(f"\n=== EPOCH {epoch+1} SUMMARY ===\n"
             f"Average Loss: {avg_loss:.4f}\n"
             f"Average Dice Score: {avg_dice:.4f}\n"
             f"Average Accuracy: {avg_acc:.4f}\n"
             f"============================\n\n")
    
    if (epoch+1) % 5 == 0:
        torch.save(clue.state_dict(), f'./checkpoints/CLUE_{VIEW}_{SEQUENCE}_{epoch+1}.pt')
log.close()