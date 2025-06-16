import pandas as pd
import numpy as np

clinical_data_path = "去識別化-食道癌-癌登資料20240207.xlsx"
df = pd.read_excel(clinical_data_path, usecols=["性別", "診斷年齡", # gender, age
                                                "腫瘤大小", "臨床T", # tumor size, clinical T
                                                "放射治療臨床標靶體積摘要", "最高放射劑量臨床標靶體積劑量", # clinical target volume, max dose CTV
                                                "首次復發型式", "生存狀態", # recurrence type, survival state
                                                "體重", "身高", # weight, height
                                                "吸菸行為", "嚼檳榔行為", "喝酒行為" # smoking, Betel Nut chewing and drinking state
                                                ])
# Check for NaN values
nan_rows = df[df.isna().any(axis=1)]

# Print the row numbers where NaN values are found
print("Rows with NaN values:")
print(nan_rows.index.tolist())

# Drop rows where all elements are NaN
df = df.dropna(how='all')

# Show raw data
df

# Check the cancer type
def check_cancer(cdp):
    df = pd.read_excel(cdp, usecols=["原發部位"]) # type of cancer, we only consider esophageal cancer
    for patient_i in range(len(df)):
        if(pd.isna(df.iloc[patient_i, 0])): # if unknown
            print('Patient {}: Unknown'.format(patient_i)) # still not sure what to do
        elif(df.iloc[patient_i, 0][:3] == 'C15'):
            # print('Correct')
            pass
        else: # if not esophageal cancer
            raise ValueError('Not Esophageal Cancer')
check_cancer(clinical_data_path)

# Processing function for each data
# Please refer to the codebook
def gender_pro(value):
    if(pd.isna(value)): # if unknown
        return 1 # default male
    digits = int(value)
    if(digits == 2):
        return 0 # female
    else:
        return 1 # male

def age_pro(value):
    if(pd.isna(value)):
        return 0.4 # default 60, assume max 120, min 20
    else:
        return np.clip((value - 20.) / 100., 0., 120.)

def ori_tumor_mean_median(df):
    tumor_size = []
    for patient_i in range(len(df)):
        if(pd.isna(df.iloc[patient_i,2]) or df.iloc[patient_i,2] > 989):
            pass
        else:
            tumor_size.append(df.iloc[patient_i,2])
    tumor_size = np.array(tumor_size)
    print('Number of valid size:', len(tumor_size))
    print('Mean:', np.mean(tumor_size))
    print('Median:', np.median(tumor_size))
    
def ori_tumor_pro(value):
    if(pd.isna(value) or value > 989):
        # raise ValueError('Ori tumor size not found') # Should not be unknown, should contact hospital for real value
        return 70.0 / 150 # default 70
    else:
        return value / 150. if value <= 150. else 1.0 # assume max 150mm, min 0mm

def clinical_T_pro(value):
    if(pd.isna(value)):
        raise ValueError('Clinical T not found')
        return
    else:
        value = str(value)[0] # capture the first character
        if(value == '1'):
            return 0.0
        elif(value == '2'):
            return 0.33
        elif(value == '3'):
            return 0.66
        elif(value == '4'):
            return 1.0
        else: # include not applicable '8888', unknown'9999' and other cancer type
            raise ValueError('Clinical T not found')
            return

def PTV_pro(value): # this is recorded as binary, 011 means no Metastasis, but lymph node and tumor are infected
    if(pd.isna(value) or value <= 0):
        raise ValueError('PTV not found')
    else:
        return np.clip(int(value) / 7.0, 0.0, 1.0) # if 111 or even worse than 111, return 1.0

def CTV_pro(value): # Dose to CTV_H (cGy)
    if(pd.isna(value) or value > 10000): # Although max value is 99997, but usually less than 10000
        raise ValueError('CTV dose not found')
    else:
        return value / 10000.
        
def recurrence_pro(value): # Disease-free or not
    if(pd.isna(value) or value >= 88):
        raise ValueError('Recurrence not found')
    else:
        return int(value > 0.) # False if no-recurrence
        
def survival_pro(value):
    if(pd.isna(value)):
        raise ValueError('Survival not found')
    else:
        return int(value) # 1 if survive
        
def height_pro(value):
    if(pd.isna(value) or value == 999):
        # raise ValueError('Height not found')
        return 0.5 # assume 160cm
    else:
        return np.clip((value - 100) / 120., 0.0, 1.0) # assume min 100cm, max 220cm
        
def weight_pro(value):
    if(pd.isna(value) or value == 999):
        raise ValueError('Weight not found')
    else:
        return np.clip((value - 30) / (120. - 30.), 0.0, 1.0) # assume min 30kg, max 120kg

def smoke_and_chew_pro(value): # same as Betel Nut Chewing
    if(pd.isna(value)): # default no smoke or no chew
        return 0
    first_2digits = int(value / 10000)
    if(first_2digits == 0 or first_2digits == 99): # or unknown
        return 0
    else:
        return 1

def drink_pro(value):
    if(pd.isna(value) or value == 999): # default drinking
        return 1
    if(value <= 1): # no drinking or quited
        return 0
    else:
        return 1

ori_tumor_mean_median(df)

df_processed = df.copy()

for patient_i in range(len(df)):
    try:
        df_processed.iloc[patient_i, 0] = gender_pro(df.iloc[patient_i,0])
        df_processed.iloc[patient_i, 1] = age_pro(df.iloc[patient_i,1])
        df_processed.iloc[patient_i, 2] = ori_tumor_pro(df.iloc[patient_i,2])
        df_processed.iloc[patient_i, 3] = clinical_T_pro(df.iloc[patient_i,3])
        df_processed.iloc[patient_i, 4] = PTV_pro(df.iloc[patient_i,4])
        df_processed.iloc[patient_i, 5] = CTV_pro(df.iloc[patient_i,5])
        df_processed.iloc[patient_i, 6] = recurrence_pro(df.iloc[patient_i,6])
        df_processed.iloc[patient_i, 7] = survival_pro(df.iloc[patient_i,7])
        df_processed.iloc[patient_i, 8] = height_pro(df.iloc[patient_i,8])
        df_processed.iloc[patient_i, 9] = weight_pro(df.iloc[patient_i,9])
        df_processed.iloc[patient_i, 10] = smoke_and_chew_pro(df.iloc[patient_i,10])
        df_processed.iloc[patient_i, 11] = smoke_and_chew_pro(df.iloc[patient_i,11])
        df_processed.iloc[patient_i, 12] = drink_pro(df.iloc[patient_i,12])
    except Exception as e:
        print(e, ', Patient', patient_i)
        df_processed.loc[patient_i, :] = 0.0

print(df_processed)
