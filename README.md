# Comprehensive-Esophageal-Cancer-Precision-Prediction-Model
# 全方位食道癌精準預測模型

# 1. Dataset Preparation
Images & masks
data_train_test_2/
  ├── TrainVolumes/   # training NIfTI volumes
  └── TestVolumes/    # testing  NIfTI volumes

Labels
labels.csv with columns PatientID, Outcome, plus 4 clinical features

Pre-processed samples
Examples (cropped & resampled to 50) are in data_50/.

# 2. Pre-trained Weights
Download resnet_200.pth from MedicalNet and place it under
MedicalNet/pretrain/resnet_200.pth

# 3. Training & Testing
train + validate (default 5-fold CV)
bash model.sh                 # edit the script to adjust parameters

# 4. Outputs
Metrics (AUC, Accuracy, F1)
log

# 5. Project Structure
├── MedicalNet/                # sub-repo with resnet.py and weights
├── data_train_test_2/         # raw volumes & labels
├── data_50/                   # pre-processed samples
├── clinical_data_processing.py
├── data_augmentation.py
├── model.sh                   # training script
├── model_best.py              # inference script
└── README.md
