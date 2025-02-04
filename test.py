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
import matplotlib.pyplot as plt

from BraTS import BraTSDataset
from Chest import ChestDataset
from RESC import RESCDataset

from IXI import IXIDataset
import nibabel as nib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
from datetime import datetime
import sys
from utils.metrics import dice_score  # dice score 계산을 위한 함수 추가
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

########## Define Model ##########
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
clue = modules.CLUE().cuda()
clue.load_state_dict(torch.load('/home/kdh/code/CLUE/checkpoints/CLUE_axial_t1_10.pt', weights_only=True))
am = modules.AnomalyModule().cuda()

class RunningROCAUC:
    def __init__(self):
        self.all_labels = []
        self.all_scores = []
        
    def update(self, labels, scores):
        # scores가 [확률1, 확률2] 형태일 경우 확률2를 사용 (비정상 클래스의 확률)
        if isinstance(scores[0], list):
            scores = [score[1] for score in scores]
            
        # 입력이 2D 리스트인 경우 1D로 평탄화 (pixel level용)
        if isinstance(labels[0], list):
            labels = [item for sublist in labels for item in sublist]
            scores = [item for sublist in scores for item in sublist]
            
        self.all_labels.extend(labels)
        self.all_scores.extend(scores)
        
    def compute(self):
        return roc_auc_score(self.all_labels, self.all_scores)


########## Test setting ##########
DATASET = sys.argv[1]

if DATASET == 'chest':
    testset = ChestDataset(
        base_dir='/home/kdh/code/CLUE/metadata',
        train=False,
    )
elif DATASET == 'retina':
    testset = RESCDataset(
        base_dir='/home/kdh/code/CLUE/metadata',
        train=False
    )
elif DATASET == 'brain':
    testset = BraTSDataset(
        base_dir='/home/kdh/code/CLUE/metadata',
        sequence=sys.argv[2],
        view=sys.argv[3],
        train=False,
        transform=[transform, transform_mask]
    )
else:
    raise Exception("You should choose among chest, retina, and brain")

testloader = DataLoader(testset, batch_size=8, shuffle=True)
if DATASET == 'brain':
    log = open(f'./log/test_{DATASET}_{sys.argv[2]}_{sys.argv[3]}.txt', 'w')
else:
    log = open(f'./log/test_{DATASET}.txt', 'w')

########## Test ##########
clue.eval()
with torch.no_grad():
    image_auroc = RunningROCAUC()
    pixel_auroc = RunningROCAUC()
    
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        image, label = data
        image = image.to(device)    # (B, 3, 224, 224)
        #label = label.to(device)    # (B,)
        
        anomaly_score, anomaly_map = model(image, DATASET, CLUE=clue, am=am)
        anomaly_score = torch.sigmoid(anomaly_score)
        
        batch_GT_image = [1 if torch.any(l!=0) else 0 for l in label]
        batch_pred_image = anomaly_score.cpu().tolist()
        
        # 이미지 레벨 AUROC 업데이트
        image_auroc.update(batch_GT_image, batch_pred_image)
        
        if DATASET != 'chest':
            # 픽셀 레벨 AUROC 업데이트
            batch_GT_pixel = label.reshape(label.shape[0], -1).cpu().numpy().tolist()
            batch_pred_pixel = anomaly_map.cpu().reshape(anomaly_map.shape[0], -1).numpy().tolist()
            pixel_auroc.update(batch_GT_pixel, batch_pred_pixel)

        batch_GT_image = [str(gt) for gt in batch_GT_image]
        batch_pred_image = [str(pred) for pred in batch_pred_image]
        if DATASET != 'chest':
            batch_GT_pixel = [str(pix) for pix in batch_GT_pixel]
            batch_pred_pixel = [str(pix) for pix in batch_pred_pixel]
            
        if False:
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
            if DATASET != 'brain':
                plt.savefig(f'./pictures/test_{DATASET}_{sys.argv[2]}_{sys.argv[3]}_{i}.png')
            else:
                plt.savefig(f'./pictures/test_{DATASET}_{i}.png')
           
    # 최종 결과 계산 및 기록
    if DATASET != 'chest':
        log.write(f"\n=== SUMMARY ===\n"
                f"Average Image-AUROC: {image_auroc.compute():.4f}\n"
                f"Average Pixel-AUROC: {pixel_auroc.compute():.4f}\n"
                f"============================\n\n")
        print((f"\n=== SUMMARY ===\n",
               f"Average Image-AUROC: {image_auroc.compute():.4f}\n",
               f"Average Pixel-AUROC: {pixel_auroc.compute():.4f}\n",
               f"============================\n\n"))
    else:
        log.write(f"\n=== SUMMARY ===\n"
                f"Average Image-AUROC: {image_auroc.compute():.4f}\n"
                f"============================\n\n")
        print(f"\n=== SUMMARY ===\n",
              f"Average Image-AUROC: {image_auroc.compute():.4f}\n",
              f"============================\n\n")


log.close()