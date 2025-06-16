import nibabel as nib
import numpy as np
import os

# 載入mask和patient的醫學影像
mask = nib.load(r'D:\find_eso\eso_mask_30.nii.gz')
patient = nib.load(r'D:\find_eso\patient_30.nii.gz')

# 查看mask的形狀
print("Mask shape:", mask.shape)

# 查看影像資料的數據型態
# print("Mask data type:", mask.get_data_dtype())

# 獲取mask影像數據陣列
data_mask = mask.get_fdata()

print("All mask data:")
print(data_mask)

# 查看patient的形狀
# print("Patient shape:", patient.shape)

# 查看影像資料的數據型態
print("Patient data type:", patient.get_data_dtype())

# 獲取patient影像數據陣列
data_patient = patient.get_fdata()

# print("All patient data:")
# print(data_patient)

# 確保兩個矩陣的形狀相同才能相乘
if data_mask.shape == data_patient.shape:
    # 進行逐元素相乘
    result = np.multiply(data_mask, data_patient)
    print("Result of mask * patient:")
    print(result)

    # 創建保存結果的目錄，如果不存在就創建
    output_dir = r'D:\find_eso\mix_mask_and_patient'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 創建 NIfTI 影像，使用 mask 的仿射矩陣和標頭（header）來保存結果
    result_img = nib.Nifti1Image(result, affine=mask.affine, header=mask.header)

    # 保存輸出為 .nii.gz 文件
    output_path = os.path.join(output_dir, 'mask_and_patient_30.nii.gz')
    nib.save(result_img, output_path)

    print(f"Result saved to: {output_path}")
else:
    print("Error: Mask and patient shapes do not match!")
