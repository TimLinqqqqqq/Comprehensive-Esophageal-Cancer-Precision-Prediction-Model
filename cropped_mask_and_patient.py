import nibabel as nib
import numpy as np
import os

# 讀取 .nii.gz 檔案
file_path = r'D:\find_eso\mix_mask_and_patient\mask_and_patient_30.nii.gz'
img = nib.load(file_path)
data = img.get_fdata()

# 找到數據中非零的範圍
non_zero_indices = np.argwhere(data != 0)

# 分別找出每個維度的最小與最大索引
x_min, y_min, z_min = np.min(non_zero_indices, axis=0)
x_max, y_max, z_max = np.max(non_zero_indices, axis=0)

# 根據非零範圍進行裁切
cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

# 裁切後保存的新檔案路徑
save_directory = r'D:\find_eso\after_cropped'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save_path = os.path.join(save_directory, 'cropped_mask_and_patient_30.nii.gz')

# 保存裁切後的結果
cropped_img = nib.Nifti1Image(cropped_data, img.affine)
nib.save(cropped_img, save_path)

print(f"裁切範圍: x({x_min}, {x_max}), y({y_min}, {y_max}), z({z_min}, {z_max})")
print(f"裁切完成，已保存為 {save_path}")
