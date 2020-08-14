import numpy as np
from time import time
import shutil
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
        train_meta.loc[train_meta["Patient"] == patient, 'Time'] = range(1, times + 1)

    secondary_measures = train_meta[train_meta['Time'] > 1]

    base_and_secondary_pairs = base_measure.merge(secondary_measures, how='inner',
                                                  on=['Patient', 'Sex', 'Age', 'SmokingStatus'])
    base_and_secondary_pairs['WeeksDiff'] = base_and_secondary_pairs['Weeks_y'] - base_and_secondary_pairs['Weeks_x']
    base_and_secondary_pairs['FVCDiff'] = base_and_secondary_pairs['FVC_y'] - base_and_secondary_pairs['FVC_x']

    base_and_secondary_pairs = base_and_secondary_pairs.set_index(['Patient', 'Weeks_y'])
    base_and_secondary_pairs = base_and_secondary_pairs.drop(['Time'], axis=1)
    base_and_secondary_pairs.columns = ['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'FVCTarget',
                                        'PercentTarget', 'WeeksDiff', 'FVCDiff']
    return base_and_secondary_pairs


def plot_single_var_to_fvc_diff(x, y, column_name):
    label = 'FVCDiff'
    feature = x[column_name]
    fig = plt.figure()
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
    possible_weeks = pd.DataFrame(np.arange(-12, 133 + 1), columns=['PossibleWeeks'])
    possible_weeks['cross_product_key'] = 1
    test_meta['cross_product_key'] = 1
    df = test_meta.merge(possible_weeks, on='cross_product_key')
    df['WeeksDiff'] = df['PossibleWeeks'] - df['Weeks']
    # df = df[(df['WeeksDiff'] >= -12) & (df['WeeksDiff'] <= 133)]
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


def main():
    # Load meta-data
    train_meta = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
    test_meta = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

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
    prediction = pd.concat([patient_week, fvc_prediction, confidence], axis=1)
    prediction.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
else:
    main()
