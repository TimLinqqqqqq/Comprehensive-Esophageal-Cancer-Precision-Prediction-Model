import numpy as np
import nibabel as nib
from monai.transforms import (
    LoadImage,
    Resize,
    RandRotate,
    RandAffine,
    RandZoom,
    RandGaussianNoise,
    Compose,
)

# 資料路徑
data_directory = "/home/u3861345/data_train_test/add_more_info/TrainVolumes/resized_mask_and_patient_01.nii.gz"
output_file = "/home/u3861345/data_train_test/add_more_info/TrainVolumes/transforms_patient_01.nii.gz"

# 定義資料增強操作
spatial_size = (128, 128, 128)  # 根據需求設定空間大小
transforms = Compose([
    LoadImage(image_only=True),  # 使用檔案路徑讀取影像
    Resize(spatial_size=spatial_size),  # 調整影像大小
    RandRotate(range_x=np.pi / 12, prob=0.5),  # 隨機旋轉
    RandAffine(
        prob=0.5,
        rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),  # 隨機旋轉
        translate_range=(10, 10, 10),  # 隨機平移
        scale_range=(0.9, 1.1),  # 隨機縮放
    ),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),  # 隨機縮放
    RandGaussianNoise(mean=0.0, std=0.1, prob=0.5),  # 加入高斯噪聲
])

# 應用資料增強
augmented_data = transforms(data_directory)

# 將資料從 PyTorch Tensor 轉換回 NumPy 陣列
augmented_data_np = augmented_data.numpy()

# 保存增強後的資料
nib.save(nib.Nifti1Image(augmented_data_np, affine=np.eye(4)), output_file)
print(f"Saved augmented file: {output_file}")
