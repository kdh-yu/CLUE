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
model, preprocess = clip.load("ViT-L/14", device=device)
for param in model.parameters():
    param.requires_grad = False
clue = modules.CLUE().cuda()
am = modules.AnomalyModule().cuda()

########## Hyperparameters ##########
LEARNING_RATE = 2e-4
EPOCH = 10

optim = torch.optim.AdamW(clue.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
loss_fn = Loss(cls_weight=0.5)

########## Utils ##########
def save_filtered_model(model, save_path):
    exclude_keys = ['model']
    model_state_dict = model.state_dict()
    filtered_state_dict = {key: value for key, value in model_state_dict.items() if key not in exclude_keys}
    torch.save(filtered_state_dict, save_path)

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
log = open(f'./log/train_ViTL14_{SEQUENCE}_{VIEW}.txt', 'a')

########## Train ##########
import matplotlib.pyplot as plt
prompt_class = DATASET
#clue.train()
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
        
        anomaly_score, anomaly_map, anomaly_map_patch = model(image, prompt_class, CLUE=clue, am=am)
        loss = loss_fn(anomaly_score, anomaly_map, label)
        if loss == torch.nan:
            print(anomaly_score)
            print(anomaly_map_patch)
            exit()
        
        label_long = (label.view(label.shape[0], -1).sum(dim=1) > 0).long()
        batch_acc = (anomaly_score.argmax(dim=1) == label_long).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += batch_acc
        total_batches += 1
        
        if i % 10 == 0:
            idx = np.random.randint(1, 8)
            
            plt.figure(figsize=(25, 5), dpi=100)
            plt.subplot(1, 6, 1)
            plt.imshow(image[idx, 0].cpu(), cmap='gray')
            plt.title('Input')
            plt.axis('off')
            
            plt.subplot(1, 6, 2)
            plt.imshow(label[idx, 0].cpu(), cmap='gray')
            plt.title('GT')
            plt.axis('off')
            
            plt.subplot(1, 6, 3)
            plt.imshow(image[idx, 0].cpu(), cmap='gray')
            plt.imshow(label[idx, 0].cpu(), cmap='jet', alpha=0.5, vmax=1, vmin=0)
            plt.title('Input + GT')
            plt.axis('off')
            
            plt.subplot(1, 6, 4)
            plt.imshow(anomaly_map_patch[idx].detach().cpu(), cmap='jet', vmax=1, vmin=0)
            plt.title('Anomaly Map (Patches)')
            plt.axis('off')
            
            pred = ['Normal', 'Abnormal']
            plt.subplot(1, 6, 5)
            plt.imshow(anomaly_map[idx].detach().cpu(), cmap='jet', vmax=1, vmin=0)
            plt.title(f'Anomaly Map, Prediction: {pred[anomaly_score.argmax(dim=1)[idx]]}')
            plt.axis('off')
            
            plt.subplot(1, 6, 6)
            plt.imshow(image[idx, 0].cpu(), cmap='gray')
            plt.imshow(anomaly_map[idx].detach().cpu(), cmap='jet', alpha=0.5, vmax=1, vmin=0)
            plt.title('Input + Anomaly Map')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'./pictures/train_{VIEW}_{SEQUENCE}_{epoch+1}.png')
            plt.close()
            
        log.write(f"[{datetime.now().strftime('%X')}]\tEPOCH: {epoch+1}\t"
                    f"Batch: {i}\tLoss: {loss.item():.4f}\t"
                    f"Accuracy: {100*batch_acc:.4f}%\n")
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clue.parameters(), max_norm=1.0)
        optim.step()
        
        if i % 50 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = epoch_loss / total_batches
    avg_acc = epoch_acc / total_batches
    
    log.write(f"\n=== EPOCH {epoch+1} SUMMARY ===\n"
             f"Average Loss: {avg_loss:.4f}\n"
             f"Average Accuracy: {100*avg_acc:.4f}%\n"
             f"============================\n\n")
    
    #if (epoch+1) % 5 == 0:
    save_filtered_model(clue, f'./checkpoints/CLUE_{VIEW}_{SEQUENCE}.pt')
log.close()