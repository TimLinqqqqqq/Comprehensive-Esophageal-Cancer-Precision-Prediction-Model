import nibabel as nib
import numpy as np
import os
from monai.transforms import Resize

def resize_and_pad_slices(data, target_shape=(128, 128), target_slices=128):
    # Resize 空間維度 (128x128)
    resizer = Resize(spatial_size=target_shape)
    resized_data = resizer(data)

    # 取得當前的切片數
    current_slices = resized_data.shape[0]

    # 如果切片數小於目標數，則補零
    if current_slices < target_slices:
        pad_width = ((0, target_slices - current_slices), (0, 0), (0, 0))
        resized_data = np.pad(resized_data, pad_width, mode='constant', constant_values=0)
    
    # 如果切片數超過目標數，則進行截斷
    elif current_slices > target_slices:
        resized_data = resized_data[:target_slices, :, :]

    resized_data = np.transpose(resized_data, (1, 2, 0))  # 轉換維度順序

    return resized_data

# 讀取 .nii.gz 檔案
file_path = '/home/u3861345/find_eso/after_cropped/cropped_mask_and_patient_30.nii.gz'  # 修改為你的檔案路徑
img = nib.load(file_path)
data = img.get_fdata()

# 如果資料是四維，移除多餘的維度
if data.ndim == 4:
    data = np.squeeze(data)

# 統一每個切片的大小 (128x128) 和切片數 (64)
resized_data = resize_and_pad_slices(data, target_shape=(128, 128), target_slices=128)

# 將 torch.Tensor 轉換為 numpy 陣列
if isinstance(resized_data, np.ndarray) is False:
    resized_data_np = resized_data.cpu().numpy()
else:
    resized_data_np = resized_data

# 裁切後保存的新檔案路徑
save_directory = '/home/u3861345/find_eso/final_after_resize'  # 修改為你的保存路徑
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save_path = os.path.join(save_directory, 'resized_mask_and_patient_30.nii.gz')

# 保存 resize 後的結果
resized_img = nib.Nifti1Image(resized_data_np, img.affine)
nib.save(resized_img, save_path)

print(f"已將資料 resize 成 {resized_data_np.shape}，並保存為 {save_path}")
