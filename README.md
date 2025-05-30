# Comprehensive-Esophageal-Cancer-Precision-Prediction-Model

# MedicalNet GN Model Training

## 環境需求
| 套件名稱       | 版本            | 備註                        |
|----------------|-----------------|-----------------------------|
| `torch`        | 1.9.0+cu111     | ⚠️ 建議升級至 ≥1.10         |
| `monai`        | 1.3.2           | ✅ 滿足 `>=1.0.1` 要求      |
| `numpy`        | 1.24.4          |                             |
| `pandas`       | 2.0.3           |                             |
| `scikit-learn` | 1.3.2           |                             |
| `nibabel`      | 5.2.1           |                             |

## 執行步驟
1. 確保以下路徑存在並正確：
   - 資料夾：`data_train_test_2/TrainVolumes`、`TestVolumes`
   - 標籤檔：`labels.csv`
   - MedicalNet 預訓練權重：`resnet_200.pth`
   - 模型程式：`resnet.py` 放在 `MedicalNet/models/`
