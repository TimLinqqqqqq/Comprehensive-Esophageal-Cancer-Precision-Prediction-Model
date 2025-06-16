import nibabel as nib

# 使用雙反斜杠或原始字串避免轉義字符問題
img1 = nib.load('/home/u3861345/find_eso/final_after_resize/resized_mask_and_patient_01.nii.gz')

# 查看影像數據的形狀
print(img1.shape)

# 查看影像資料的數據型態
print(img1.get_data_dtype())

# 獲取影像數據陣列
# data1 = img1.get_fdata()

# print("All data:")
# print(data1)

# 使用雙反斜杠或原始字串避免轉義字符問題
img2 = nib.load('/home/u3861345/data_train_test/first_try_data/TrainVolumes/01_201804-RTCT.nii.gz')

# 查看影像數據的形狀
print(img2.shape)

# 查看影像資料的數據型態
print(img2.get_data_dtype())
