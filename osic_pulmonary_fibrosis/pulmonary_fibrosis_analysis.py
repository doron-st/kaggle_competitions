import numpy as np
from time import time
import shutil
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import statsmodels.api as sm


####################
# meta-data analysis
####################

def group_base_and_secondary_measurements(train_meta):
    base_measure = train_meta.groupby(by='Patient').first()
    
    data_time = train_meta.groupby(by="Patient")["Weeks"].count().reset_index()
    train_meta["Time"] = 0
    
    for patient, times in zip(data_time["Patient"], data_time["Weeks"]):
        train_meta.loc[train_meta["Patient"] == patient, 'Time'] = range(1, times+1)
        
    secondary_measures = train_meta[train_meta['Time'] > 1]
    
    base_and_secondary_pairs = base_measure.merge(secondary_measures, how='inner', on=['Patient', 'Sex', 'Age', 'SmokingStatus'])
    base_and_secondary_pairs['WeeksDiff'] = base_and_secondary_pairs['Weeks_y'] - base_and_secondary_pairs['Weeks_x']
    base_and_secondary_pairs['FVCDiff'] = base_and_secondary_pairs['FVC_y'] - base_and_secondary_pairs['FVC_x']
    
    base_and_secondary_pairs = base_and_secondary_pairs.set_index(['Patient', 'Weeks_y'])
    base_and_secondary_pairs = base_and_secondary_pairs.drop(['Time'], axis=1)
    base_and_secondary_pairs.columns = ['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'FVCTarget', 'PercentTarget', 'WeeksDiff', 'FVCDiff']
    return base_and_secondary_pairs


def plot_single_var_to_fvc_diff(x, y, column_name):
    label = 'FVCDiff'
    feature = x[column_name]
    fig  = plt.figure()
    plt.scatter(feature, y, marker='o', alpha=0.2)
    m, b = np.polyfit(feature, y, 1)
    plt.plot(feature, m * feature + b, color='red')
    plt.xlabel(column_name)
    plt.ylabel(label)
    fig.show()
    # Fit and summarize OLS model on single param
    reg = sm.OLS(y, feature)
    print(reg.fit().summary())



def encode_categories(df):
    label_encoder = LabelEncoder()
    df['SmokingStatus'] = label_encoder.fit_transform(df['SmokingStatus'])
    df['Sex'] = (df['Sex'] == 'Male') + 0
    return df, label_encoder

def define_features(df):
    # drop label, columns that will not be available in test-set (leaking)
    train_x = df.drop(['FVCDiff', 'FVCTarget', 'PercentTarget'], axis=1)
    # drop non-informative and collinear variable
    train_x = train_x.drop(['Percent'], axis=1)
    return train_x

def scale_features(df, scaler=None):    
    if scaler is None:
        scaler = MinMaxScaler() 
        df_scaled = pd.DataFrame(scaler.fit_transform(df), 
                                                columns=df.columns, 
                                                index=df.index)
    else:
        df_scaled = pd.DataFrame(scaler.transform(df), 
                                                columns=df.columns, 
                                                index=df.index)
    return df_scaled, scaler

def make_test_data_to_model_compatible(test_meta, label_encoder, scaler):
    possible_weeks = pd.DataFrame(np.arange(-12,133), columns=['PossibleWeeks'])    
    possible_weeks['cross_product_key'] = 1
    test_meta['cross_product_key'] = 1
    df = test_meta.merge(possible_weeks, on='cross_product_key')
    df['WeeksDiff'] = df['PossibleWeeks'] - df['Weeks']
    #df = df[(df['WeeksDiff'] >= -12) & (df['WeeksDiff'] <= 133)]
    df = df.drop('cross_product_key', axis=1)
    
    df['SmokingStatus'] = label_encoder.transform(df['SmokingStatus'])
    df['Sex'] = (df['Sex'] == 'Male') + 0

    test_x_dont_scale = df[['Patient', 'PossibleWeeks']]
    patient_week = pd.Series(test_x_dont_scale['Patient'] + '_' 
                         + test_x_dont_scale['PossibleWeeks'].astype(str), 
                         name='Patient_Week')
    df = df.drop(['Patient', 'PossibleWeeks', 'Percent'], axis=1)
    df_scaled, _ = scale_features(df, scaler)
    return df, df_scaled, patient_week

def analyse_each_variable(train_x, train_y):
    plot_single_var_to_fvc_diff(train_x, train_y, 'WeeksDiff')    
    plot_single_var_to_fvc_diff(train_x, train_y, 'FVC')
    plot_single_var_to_fvc_diff(train_x, train_y, 'Weeks')    
    plot_single_var_to_fvc_diff(train_x, train_y, 'Age')    
    plot_single_var_to_fvc_diff(train_x, train_y, 'Sex')    
    plot_single_var_to_fvc_diff(train_x, train_y, 'SmokingStatus')    
    

def plot_actual_fvc_vs_predicted_in_training_set(fvc_target, fvc_predicted):
    plt.figure()
    plt.xlabel('actual FVC at secondary checkup')
    plt.ylabel('predicted FVC at secondary checkup')
    plt.scatter(fvc_target, fvc_predicted, alpha=0.2)
    plt.show()
    ols_model = sm.OLS(fvc_target, fvc_predicted)
    predictor = ols_model.fit()
    print(predictor.summary())
    
# Load meta-data
train_meta = pd.read_csv('input/osic-pulmonary-fibrosis-progression/train.csv')
test_meta = pd.read_csv('input/osic-pulmonary-fibrosis-progression/test.csv')

base_and_secondary_pairs = group_base_and_secondary_measurements(train_meta)
base_and_secondary_pairs_enc, label_encoder = encode_categories(base_and_secondary_pairs)

# prepare final training-set and labels
train_y = base_and_secondary_pairs_enc['FVCDiff']
train_x = define_features(base_and_secondary_pairs_enc)
analyse_each_variable(train_x, train_y)
train_x_scaled, scaler = scale_features(train_x)

# Fit and summarize OLS model
ols_model = sm.OLS(train_y, train_x_scaled)
predictor = ols_model.fit()
print(predictor.summary())

# sanity check prediction on training-set
predicted_fvc_diff = predictor.predict(train_x_scaled)
fvc_predicted_training = predicted_fvc_diff + train_x['FVC']
plot_actual_fvc_vs_predicted_in_training_set(base_and_secondary_pairs['FVCTarget'], fvc_predicted_training)

# process and predict from test_data
test_x, test_x_scaled, patient_week = make_test_data_to_model_compatible(test_meta, label_encoder, scaler)
fvc_diff_prediction = predictor.predict(test_x_scaled)
fvc_prediction = pd.Series(test_x['FVC'] + fvc_diff_prediction, name='FVC').astype(int)

# set confidence above regression on training-set confidence interval
confidence = pd.Series(fvc_prediction * 0.15, name='Confidence').astype(int)
prediction = pd.concat([patient_week, fvc_prediction, confidence ], axis=1)
prediction.to_csv('submission.csv', index=False)
###################
# CT scans analysis
###################
from skimage import exposure
import pydicom

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
