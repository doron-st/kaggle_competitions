import numpy as np
from time import time
import shutil
from skimage import exposure
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pydicom
from scipy.stats import pearsonr
train_meta = pd.read_csv('input/osic-pulmonary-fibrosis-progression/train.csv')
test_meta = pd.read_csv('input/osic-pulmonary-fibrosis-progression/test.csv')

base_measure = train_meta.groupby(by='Patient')

data_time = train_meta.groupby(by="Patient")["Weeks"].count().reset_index()
train_meta["Time"] = 0

for patient, times in zip(data_time["Patient"], data_time["Weeks"]):
    train_meta.loc[train_meta["Patient"] == patient, 'Time'] = range(1, times+1)
    
secondary_measures = train_meta[train_meta['Time'] > 1]

base_and_secondary_pairs = base_measure.merge(secondary_measures, how='inner', on=['Patient', 'Sex', 'Age', 'SmokingStatus'])
base_and_secondary_pairs['WeeksDiff'] = base_and_secondary_pairs['Weeks_y'] - base_and_secondary_pairs['Weeks_x']
base_and_secondary_pairs['FVCDiff'] = base_and_secondary_pairs['FVC_y'] - base_and_secondary_pairs['FVC_x']

def plot_weeks_diff_to_fvc(base_and_secondary_pairs):
    fig  = plt.figure()
    plt.scatter(base_and_secondary_pairs['WeeksDiff'], base_and_secondary_pairs['FVCDiff'], marker='o', alpha=0.2)

    corr = pearsonr(base_and_secondary_pairs['WeeksDiff'], base_and_secondary_pairs['FVCDiff'])
    print(f'pearsonr = {corr}')
    
    m, b = np.polyfit(base_and_secondary_pairs['WeeksDiff'], base_and_secondary_pairs['FVCDiff'], 1)
    plt.plot(base_and_secondary_pairs['WeeksDiff'], m * base_and_secondary_pairs['WeeksDiff'] + b, color='red')
    fig.show()

plot_weeks_diff_to_fvc(base_and_secondary_pairs)    

train_y = base_and_secondary_pairs['FVCDiff']
train_x = base_and_secondary_pairs.drop(['FVCDiff'], axis=1)


    

min_fvc = train[train["Time"] == 1][["Patient", "FVC"]].reset_index(drop=True)

idx = train.groupby(["Patient"])["Weeks"].transform(max) == train["Weeks"]
max_fvc = train[idx][["Patient", "FVC"]].reset_index(drop=True)

# Compute difference and select only top patients with biggest difference
data = pd.merge(min_fvc, max_fvc, how="inner", on="Patient")
data["Dif"] = data["FVC_x"] - data["FVC_y"]

# Select only top n
l = list(data.sort_values("Dif", ascending=False).head(100)["Patient"])
x = train[train["Patient"].isin(l)]
gbp = train_meta.groupby('Patient').agg([
                                        lambda x: x.iloc[0], 
                                        lambda x: x.iloc[1],
                                        lambda x: x.iloc[2],
                                        lambda x: x.iloc[3],
                                        lambda x: x.iloc[4]
                                        ])
gvp = gbp[['Weeks', 'FVC']]
gbp.head()

def even_select(array, m):
    """
    select m elements from array, trying to pick them as evenly as possible
    :param array: inputs array
    :param m: number of elements to select, must by < array.shape[0]
    :return: sub-array
    """
    n = array.shape[0]
    assert m <= n
    if n == m:
        return array
    elif m > n/2:
        cut = np.ones(n, dtype=int)
        q, r = divmod(n, n - m)
        indices = [q * i + min(i, r) for i in range(n - m)]
        cut[indices] = 0
    else:
        cut = np.zeros(n, dtype=int)
        q, r = divmod(n, m)
        indices = [q * i + min(i, r) for i in range(m)]
        cut[indices] = 1
    return array[cut == 1]

def load_ct_scans_paths():
    import os
    train_dir = '../inputs/osic-pulmonary-fibrosis-progression/train'
    scan_paths = {}
    ct_scan_subdirs = os.listdir(train_dir)
    print(f'training set size: {len(ct_scan_subdirs)}')
    num_of_scans = []
    for ct_scan_subdir in ct_scan_subdirs:
        scans_per_patient = os.listdir(f'{train_dir}/{ct_scan_subdir}')
        scans_per_patient = sorted(scans_per_patient, key=lambda x: int(x.split('.')[0]))
        scan_paths[ct_scan_subdir] = scans_per_patient
        num_of_scans.append(len(scans_per_patient))
    sns.distplot(num_of_scans, kde=False, bins=50)

def load_and_notmalize_Ct_scans():


    number_of_scans_per_patient = 12
    start = time()
    scan_sizes_list = []
    scans_per_patient = {}
    i = 0
    ct_volumes = []

    for patient_id in scan_paths:
        scans = []
        fig, axs = plt.subplots(1, number_of_scans_per_patient, figsize=(20, 20 * number_of_scans_per_patient))
        # output_patient_dir = f'/kaggle/working/{patient_dir}'
        # os.makedirs(output_patient_dir, exist_ok=True)
        scans_array = np.array(scan_paths[patient_id])
        # print(f'{i}) {patient_id}')
        j = 0
        for scan_file in even_select(scans_array, number_of_scans_per_patient):
            dicom = pydicom.dcmread(f'{train_dir}/{patient_id}/{scan_file}')
            pixels = dicom.pixel_array
            scans.append(pixels)
            # np.save(f'{output_patient_dir}/{scan_file}.npy', pixels)
            scan_sizes_list.append(pixels.shape)
            # axs[j].title(dicom['PatientID']._value +': ' + dicom['BodyPartExamined']._value)
            j += 1
        i += 1
        scans_per_patient[patient_id] = scans

        scans_3d = np.stack(scans, axis=0)
        scans_3d = exposure.equalize_hist(scans_3d)
        # scans_3d = (scans_3d - np.mean(scans_3d)) / np.std(scans_3d)
        # print(f'mean: {np.mean(scans_3d)}, min: {np.min(scans_3d)}, max: {np.max(scans_3d)}')
        # convert to 0-255 int scale
        scans_3d += np.abs(np.min(scans_3d))
        scans_3d = (((scans_3d) / np.max(scans_3d)) * 255).astype('uint8')
        # print(f'mean: {np.mean(scans_3d)}, min: {np.min(scans_3d)}, max: {np.max(scans_3d)}')
        if i < 10:
            for j in range(number_of_scans_per_patient):
                axs[j].imshow(scans_3d[j], cmap="plasma")
            plt.show()
        ct_volumes.append(scans_3d)
    scan_sizes = pd.DataFrame(scan_sizes_list)
    scan_sizes.describe()

    print(f'time = {time() - start}')
