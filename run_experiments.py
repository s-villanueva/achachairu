import os
import json
import shutil
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import densenet121, DenseNet121_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from roboflow import Roboflow

# Hyperparameters
NUM_CLASSES  = 3
LEARNING_RATE = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
STEP_LR      = 5
GAMMA_LR     = 0.1

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, json_path, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(json_path) as f:
            coco = json.load(f)

        self.imgs = {img['id']: img for img in coco['images']}
        self.img_ids = list(self.imgs.keys())

        self.anns = {img_id: [] for img_id in self.img_ids}
        for ann in coco['annotations']:
            self.anns[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.imgs[img_id]

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        image = TF.to_tensor(image)

        anns = self.anns[img_id]

        boxes  = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes':    boxes,
            'labels':   labels,
            'image_id': torch.tensor([img_id])
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))

def setup_dataset():
    # Descargar el dataset solo si no existe el directorio base
    if not os.path.exists('./Achachairu-1'):
        rf = Roboflow(api_key="iAo0Rm2Dwt2m0yvaCcBm")
        project = rf.workspace("santiagos-workspace-2quku").project("achachairu-bh5vl")
        version = project.version(1)
        dataset = version.download("coco")
        location = dataset.location
    else:
        print("Dataset './Achachairu-1' encontrado. Omitiendo descarga.")
        location = './Achachairu-1'
        
    TRAIN_DIR = os.path.join(location, 'train')
    VALID_DIR = os.path.join(location, 'valid')
    TEST_DIR  = os.path.join(location, 'test')
    
    TRAIN_JSON = os.path.join(TRAIN_DIR, '_annotations.coco.json')
    VALID_JSON = os.path.join(VALID_DIR, '_annotations.coco.json')
    TEST_JSON  = os.path.join(TEST_DIR,  '_annotations.coco.json')
    return TRAIN_DIR, VALID_DIR, TEST_DIR, TRAIN_JSON, VALID_JSON, TEST_JSON


def create_model():
    densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
    backbone = densenet.features
    backbone.out_channels = 1024

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model


def run_experiment(batch_size, epochs, run_name, train_dir, train_json, valid_dir, valid_json, device):
    print(f"\n{'='*55}\nIniciando escenario: Batch Size={batch_size}, Epochs={epochs}\n{'='*55}")
    
    output_dir = os.path.join("results", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = COCODataset(train_dir, train_json)
    valid_dataset = COCODataset(valid_dir, valid_json)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0 # Configurado en 0 para evitar problemas de multiprocesamiento en Windows
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = create_model()
    model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_LR,
        gamma=GAMMA_LR
    )

    history = {
        'epoch': [],
        'train_loss': [],
        'val_map50':  [],
        'val_map':    [],
        'val_recall': [],
        'val_f1':     [],
        'lr':         []
    }

    best_map50 = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # VALIDACION
        model.eval()
        metric = MeanAveragePrecision(iou_thresholds=[0.5], extended_summary=True)

        with torch.no_grad():
            for images, targets in valid_loader:
                images      = [img.to(device) for img in images]
                preds       = model(images)
                preds_cpu   = [{k: v.cpu() for k, v in p.items()} for p in preds]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
                metric.update(preds_cpu, targets_cpu)

        results = metric.compute()
        map50   = results['map_50'].item()
        map_val = results['map'].item() if not torch.isnan(results['map']) else 0.0
        recall  = results['mar_100'].item() if not torch.isnan(results['mar_100']) else 0.0
        f1      = (2 * map50 * recall) / (map50 + recall + 1e-7)

        current_lr = lr_scheduler.get_last_lr()[0]
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['val_map50'].append(map50)
        history['val_map'].append(map_val)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['lr'].append(current_lr)

        print(f'Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  '
              f'mAP50: {map50:.4f}  mAP: {map_val:.4f}  '
              f'Recall: {recall:.4f}  F1: {f1:.4f}  '
              f'LR: {current_lr:.6f}')
              
        if map50 > best_map50:
            best_map50 = map50
            # Guardar el modelo con mejor mAP50
            torch.save(model.state_dict(), os.path.join(output_dir, 'fasterrcnn_densenet121_best.pth'))

    # Guardar los pesos finales
    torch.save(model.state_dict(), os.path.join(output_dir, 'fasterrcnn_densenet121_final.pth'))

    # Exportar metricas a CSV
    df_metrics = pd.DataFrame(history)
    df_metrics.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    # Gráfico 1: Loss y mAP50
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, history['train_loss'], color='steelblue', label='Train Loss')
    ax1.set_title('Loss de entrenamiento')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs_range, history['val_map50'], color='green', label='Val mAP50')
    ax2.set_title('mAP50 en validación')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP50')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_map50.png'))
    plt.close(fig)

    # Gráfico 2: Resumen de las 4 metricas (Loss, mAP, Recall, F1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(epochs_range, history['train_loss'], color='steelblue')
    axes[0,0].set_title('Train Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].grid(True)
    
    axes[0,1].plot(epochs_range, history['val_map50'], color='green', label='mAP50')
    axes[0,1].plot(epochs_range, history['val_map'],   color='olive', label='mAP50-95')
    axes[0,1].set_title('mAP')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    axes[1,0].plot(epochs_range, history['val_recall'], color='orange')
    axes[1,0].set_title('Recall')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].grid(True)
    
    axes[1,1].plot(epochs_range, history['val_f1'], color='red')
    axes[1,1].set_title('F1 Score')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics.png'))
    plt.close(fig)
    
    return {
        'Run': run_name,
        'Batch Size': batch_size,
        'Epochs': epochs,
        'Best mAP50': best_map50,
        'Best mAP': max(history['val_map']),
        'Best Recall': max(history['val_recall']),
        'Best F1': max(history['val_f1'])
    }

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Usando el dispositivo: {device}')
    
    # Preparar el dataset usando Roboflow
    TRAIN_DIR, VALID_DIR, TEST_DIR, TRAIN_JSON, VALID_JSON, TEST_JSON = setup_dataset()
    
    # Lista de escenarios a correr (batch_size, epochs)
    scenarios = [
        (4, 150),
        (8, 150),
        (16, 100),
        (16, 150),
        (32, 100),
        (32, 150)
    ]
    
    all_results = []
    
    for batch_size, epochs in scenarios:
        run_name = f"batch{batch_size}_epochs{epochs}"
        res = run_experiment(
            batch_size=batch_size,
            epochs=epochs,
            run_name=run_name,
            train_dir=TRAIN_DIR,
            train_json=TRAIN_JSON,
            valid_dir=VALID_DIR,
            valid_json=VALID_JSON,
            device=device
        )
        all_results.append(res)
        
        # Limpiar la VRAM de CUDA para evitar quedarse sin memoria entre experimentos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Guardar y mostrar el resumen global de todos los escenarios
    summary_df = pd.DataFrame(all_results)
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv('results/summary.csv', index=False)
    print("\n--- RESUMEN FINAL ---")
    print(summary_df.to_string())
    print("\nTodos los escenarios completados satisfactoriamente.")

if __name__ == '__main__':
    main()
