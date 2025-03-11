import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from glob import glob
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    ScaleIntensityRanged,
    RandRotateD,
    RandAffineD,
    RandZoomD,
    RandGaussianNoiseD,
)
from monai.utils import set_determinism
from monai.data import Dataset

# 设置 logging
logging.basicConfig(
    filename='/home/u3861345/data_50/log_data/preprocess_and_resnet_50_1.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

def prepare(in_dir, label_file, pixdim=(1.0, 1.0, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 128]):
    set_determinism(seed=0)

    # 读取 labels.csv，指定列名
    labels_df = pd.read_csv(label_file, header=None, names=[
        'patient_id', 'tumor_size', 'smoking_behavior',
        'betel_nut_chewing_behavior', 'drinking_behavior', 'label'
    ])
    
    # 获取训练和测试影像的路径
    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))

    # 构建训练数据列表，包含影像路径、标签和新的特征
    train_files = []
    for image_name in path_train_volumes:
        patient_id = os.path.basename(image_name).replace('.nii.gz', '')
        patient_data = labels_df.loc[labels_df['patient_id'] == patient_id]
        
        if patient_data.empty:
            continue  # 如果没有找到对应的患者数据，跳过

        label_value = patient_data['label'].values[0]
        tumor_size = patient_data['tumor_size'].values[0]
        smoking_behavior = patient_data['smoking_behavior'].values[0]
        betel_nut_chewing_behavior = patient_data['betel_nut_chewing_behavior'].values[0]
        drinking_behavior = patient_data['drinking_behavior'].values[0]
        
        train_files.append({
        "vol": image_name,
        "label": np.array(label_value, dtype=np.float32),
        "tumor_size": np.array(tumor_size, dtype=np.float32),
        "smoking_behavior": np.array(smoking_behavior, dtype=np.float32),
        "betel_nut_chewing_behavior": np.array(betel_nut_chewing_behavior, dtype=np.float32),
        "drinking_behavior": np.array(drinking_behavior, dtype=np.float32)
    })


    # 构建测试数据列表
    test_files = []
    for image_name in path_test_volumes:
        patient_id = os.path.basename(image_name).replace('.nii.gz', '')
        patient_data = labels_df.loc[labels_df['patient_id'] == patient_id]
        
        if patient_data.empty:
            continue  # 如果没有找到对应的患者数据，跳过

        label_value = patient_data['label'].values[0]
        tumor_size = patient_data['tumor_size'].values[0]
        smoking_behavior = patient_data['smoking_behavior'].values[0]
        betel_nut_chewing_behavior = patient_data['betel_nut_chewing_behavior'].values[0]
        drinking_behavior = patient_data['drinking_behavior'].values[0]
        
        test_files.append({
            "vol": image_name,
            "label": np.array(label_value, dtype=np.float32),
            "tumor_size": np.array(tumor_size, dtype=np.float32),
            "smoking_behavior": np.array(smoking_behavior, dtype=np.float32),
            "betel_nut_chewing_behavior": np.array(betel_nut_chewing_behavior, dtype=np.float32),
            "drinking_behavior": np.array(drinking_behavior, dtype=np.float32)
        })
    
    # 定义需要转换为 Tensor 的键
    keys_to_transform = ["vol", "label", "tumor_size", "smoking_behavior", "betel_nut_chewing_behavior", "drinking_behavior"]

    # 定义训练数据的转换
    train_transforms = Compose(
        [
            LoadImaged(keys=["vol"]),
            EnsureChannelFirstD(keys=["vol"]),
            Spacingd(keys=["vol"], pixdim=pixdim, mode="bilinear"),
            ScaleIntensityRanged(
                keys=["vol"], a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0, clip=True
            ),
            Resized(keys=["vol"], spatial_size=spatial_size),
            
            # 数据增强操作
            RandRotateD(
                keys=["vol"], range_x=np.pi/12, prob=0.5
            ),
            RandAffineD(
                keys=["vol"], prob=0.5,
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
                translate_range=(10, 10, 10),
                scale_range=(0.9, 1.1)
            ),
            RandZoomD(
                keys=["vol"], min_zoom=0.9, max_zoom=1.1, prob=0.5
            ),
            RandGaussianNoiseD(
                keys=["vol"], mean=0.0, std=0.1, prob=0.5
            ),
            
            ToTensord(keys=keys_to_transform),
        ]
    )

    # 定义测试数据的转换
    test_transforms = Compose(
        [
            LoadImaged(keys=["vol"]),
            EnsureChannelFirstD(keys=["vol"]),
            Spacingd(keys=["vol"], pixdim=pixdim, mode="bilinear"),
            ScaleIntensityRanged(
                keys=["vol"], a_min=a_min, a_max=a_max,
                b_min=0.0, b_max=1.0, clip=True
            ),
            Resized(keys=["vol"], spatial_size=spatial_size),
            ToTensord(keys=keys_to_transform),
        ]
    )

    # 创建训练和测试数据集
    train_ds = Dataset(data=train_files, transform=train_transforms)
    test_ds = Dataset(data=test_files, transform=test_transforms)

    return train_ds, test_ds

# 定义模型
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        # 3D 卷积层 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        # 3D 卷积层 2
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 如果输入与输出通道不匹配，则使用 1x1 卷积进行降维
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_classes, num_features):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.layer2 = self._make_layer(64, 128, stride=2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.dropout4 = nn.Dropout(p=0.2)
        
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.num_features = num_features
        self.fc_image = nn.Linear(512, 256)
        self.fc_features = nn.Linear(num_features, 64)
        self.fc_combined = nn.Linear(256 + 64, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock3D(in_channels, out_channels, stride),
            BasicBlock3D(out_channels, out_channels, 1)
        )

    def forward(self, x, features):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.dropout1(out)

        out = self.layer1(out)
        out = self.pool2(out)
        out = self.dropout2(out)

        out = self.layer2(out)
        out = self.pool3(out)
        out = self.dropout3(out)

        out = self.layer3(out)
        out = self.dropout4(out)

        out = self.layer4(out)
        out = self.pool5(out)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = out.view(out.size(0), -1)

        out_image = F.relu(self.fc_image(out))
        out_features = F.relu(self.fc_features(features))

        out_combined = torch.cat((out_image, out_features), dim=1)
        out = self.fc_combined(out_combined)
        return out

def train_model(train_loader, model, criterion, optimizer, num_epochs, device):
    # model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs = batch["vol"].to(device)
            labels = batch["label"].float().to(device).view(-1, 1)
            
            # 提取新的特征，已经是 Tensor
            features = torch.cat([
                batch["tumor_size"].view(-1, 1),
                batch["smoking_behavior"].view(-1, 1),
                batch["betel_nut_chewing_behavior"].view(-1, 1),
                batch["drinking_behavior"].view(-1, 1)
            ], dim=1).float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, features)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        
        logging.info('Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, train_accuracy))

def test_model(test_loader, model, criterion, device):
    model.eval()
    # model.to(device)
    total_test_loss = 0.0
    correct_test = 0
    total_test = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["vol"].to(device)
            labels = batch["label"].float().to(device).view(-1, 1)
            
            features = torch.cat([
                batch["tumor_size"].view(-1, 1),
                batch["smoking_behavior"].view(-1, 1),
                batch["betel_nut_chewing_behavior"].view(-1, 1),
                batch["drinking_behavior"].view(-1, 1)
            ], dim=1).float().to(device)
            
            outputs = model(inputs, features)
            
            # 將 labels 和 outputs 移動到 CPU，以便進行日誌記錄
            labels_cpu = labels.cpu()
            outputs_cpu = outputs.cpu()
            
            # 新增的 logging 語句，記錄測試標籤和模型的原始輸出
            logging.info('Test Labels: {}, Type: {}, Unique Values: {}'.format(
                labels_cpu.numpy(), labels_cpu.dtype, torch.unique(labels_cpu)))
            logging.info('Raw outputs: {}, Type: {}'.format(
                outputs_cpu.numpy(), outputs_cpu.dtype))
            
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            all_labels.extend(labels_cpu.numpy())
            all_predictions.extend(predicted.cpu().numpy())

        from sklearn.metrics import f1_score, confusion_matrix
        f1 = f1_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        
        average_test_loss = total_test_loss / len(test_loader)
        test_accuracy = correct_test / total_test if total_test > 0 else 0
        
        # 混淆矩陣中的 TN, FP, FN, TP
        if cm.size == 4:
            TN, FP, FN, TP = cm.ravel()
        else:
            TN = FP = FN = TP = 0
            if len(cm) == 1:
                if all_labels[0] == 0:
                    TN = cm[0][0]
                else:
                    TP = cm[0][0]
        
        # 計算靈敏度和特異性
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        logging.info('Average Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(
            average_test_loss, test_accuracy))
        logging.info('Sensitivity: {:.4f}, Specificity: {:.4f}'.format(
            sensitivity, specificity))
        logging.info('F1 Score: {:.4f}'.format(f1))
        logging.info('Confusion Matrix:\n{}'.format(cm))


if __name__ == "__main__":
    # 数据目录和标签文件
    data_directory = "/home/u3861345/data_50/data_train_test"
    label_file = "/home/u3861345/data_50/data_train_test/labels.csv"
    
    # 准备数据
    train_ds, test_ds = prepare(in_dir=data_directory, label_file=label_file)
    
    # 设置批次大小和工作线程数
    batch_size = 1  # 根据您的资源调整
    num_workers = 1  # 根据您的资源调整，建议设为1或0，避免警告
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 提取训练集的标签
    labels = [data['label'] for data in train_ds.data]
    labels = np.array(labels).flatten()
    
    # 确保标签是 0 或 1
    labels = labels.astype(int)
    
    # 计算正类和负类样本数量
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)
    logging.info("正类样本数量：{}, 负类样本数量：{}".format(num_pos, num_neg))
    
    # 计算 pos_weight
    pos_weight_value = num_pos / num_neg
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    
    # 初始化模型、损失函数和优化器
    num_classes = 1  # 二分类问题
    num_features = 4  # 临床特征的数量
    
    model = ResNet3D(num_classes=num_classes, num_features=num_features)
    model.to(device)  # 将模型移动到设备上
    
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 使用 pos_weight
    criterion = FocalLoss(alpha=0.7, gamma=1)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 训练参数
    num_epochs = 100  # 根据需要调整
    
    # 训练模型
    train_model(train_loader, model, criterion, optimizer, num_epochs, device)
    
    # 测试模型
    logging.info("Final Test on External Test Set:")
    test_model(test_loader, model, criterion, device)
    torch.save(model, "trained_model.pth")

