# 維度變化 把5維改成3維
import nibabel as nib
import numpy as np

# 讀取 .nii.gz 檔案
nii_file = r'D:\datasets\data_train_test\TrainVolumes\18_201908-RTCT.nii.gz'
img = nib.load(nii_file)

# 取得資料陣列
data = img.get_fdata()

# 檢查資料的維度
print(f"Original shape: {data.shape}")

# 如果資料是5維，選擇第五維度的第一個通道，並移除第四維度
if data.ndim == 5:
    data = data[:, :, :, 0, 0]  # 保留前三個維度

# 檢查修改後的維度
print(f"Modified shape: {data.shape}")

# 建立新的NIfTI影像物件
new_img = nib.Nifti1Image(data, img.affine, img.header)

# 儲存處理後的資料
output_file = r'D:\datasets\data_train_test\TrainVolumes\18_201908-RTCT.nii.gz'
nib.save(new_img, output_file)

print(f"Modified file saved as: {output_file}")
