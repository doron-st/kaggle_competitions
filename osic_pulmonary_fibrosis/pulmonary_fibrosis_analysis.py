from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


####################
# meta-data analysis
####################

# Current best OLS cross-validation score -6.7918
# Current best RF cross-validation score  -6.7788 (n_estimators=50, random_state=0, max_leaf_nodes=5)) StandardScaler
# current best GB cross-validation score  -6.7695 (n_estimators=25, learning_rate=0.05, n_jobs=4, max_depth=3)
# current best NN cross-validation score  -6.7644
# RF performs better than GB on competition test-data (which base the score on final 3 measurements

RANDOM_STATE = 2
MAXIMAL_SCORE_DIFF = 1000
MINIMAL_SCORE_STD = 70


@dataclass(frozen=True)
class NueralNetConfig:
    epochs: int
    learning_rate: float
    layer_size: int
    batch_size: int
    dropout_rate: float
    fit_verbosity: int


nn_config = NueralNetConfig(epochs=5,
                            learning_rate=0.01,
                            layer_size=100,
                            batch_size=128,
                            dropout_rate=0.2,
                            fit_verbosity=0)


def explore_data(df):
    label_encoder = LabelEncoder()
    df.loc[:, 'SmokingStatus_code'] = label_encoder.fit_transform(df['SmokingStatus'])
    df.loc[:, 'Sex_code'] = (df['Sex'] == 'Male') + 0
    plt.figure()
    sns.pairplot(df)
    plt.figure()
    sns.boxplot(x=df['SmokingStatus'], y=df['FVC'])
    plt.figure()
    sns.boxplot(x=df['SmokingStatus'], y=df['Percent'])


def group_base_and_secondary_measurements(labeled_data):
    """
    :param labeled_data: labeled data of Patient, Week (since CT scan), FVC, Age, Sex, SmokingStatus
    :return: labled data of paired records between first measurement of different weeks from the same patient.
               Weeks (initial), FVC (initial), Percent (initial), Age, Sex, SmokingStatus, FVCTarget, PercentTarget, WeeksDiff, FVCDiff
    """
    labeled_data_cp = labeled_data.copy()  # to avoid side-affects and warnings by modifying original data_frame
    base_measure = labeled_data.groupby(by='Patient').first()
    data_time = labeled_data.groupby(by="Patient")["Weeks"].count().reset_index()

    labeled_data_cp['Time'] = 0

    for patient, times in zip(data_time["Patient"], data_time["Weeks"]):
        labeled_data_cp.loc[labeled_data_cp["Patient"] == patient, 'Time'] = range(1, times + 1)

    secondary_measures = labeled_data_cp[labeled_data_cp['Time'] > 1]

    base_and_secondary_pairs = base_measure.merge(secondary_measures, how='inner',
                                                  on=['Patient', 'Sex', 'Age', 'SmokingStatus'])
    base_and_secondary_pairs.loc[:, 'WeeksDiff'] = base_and_secondary_pairs['Weeks_y'] - base_and_secondary_pairs[
        'Weeks_x']
    base_and_secondary_pairs.loc[:, 'FVCDiff'] = base_and_secondary_pairs['FVC_y'] - base_and_secondary_pairs['FVC_x']

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


def encode_categories(df, fit=True, label_encoder=None):
    df.loc[:, 'Sex'] = (df['Sex'] == 'Male') + 0

    if fit:
        label_encoder = LabelEncoder()
        df.loc[:, 'SmokingStatus'] = label_encoder.fit_transform(df['SmokingStatus'])
        return df, label_encoder
    else:
        df.loc[:, 'SmokingStatus'] = label_encoder.transform(df['SmokingStatus'])
        return df


def define_features(df):
    # drop label, columns that will not be available in test-set (leaking)
    train_x = df.drop(['FVCDiff', 'FVCTarget', 'PercentTarget'], axis=1)
    # drop non-informative and collinear variable
    train_x = train_x.drop(['Percent'], axis=1)
    return train_x


def scale_features(df, scaler=None, scaler_name=None):
    # fit scaler
    if scaler is None:
        if scaler_name == 'min_max':
            scaler = MinMaxScaler()
        elif scaler_name == 'standard':
            scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df),
                                 columns=df.columns,
                                 index=df.index)
        return df_scaled, scaler
    # transform using existing scaler
    else:
        df_scaled = pd.DataFrame(scaler.transform(df),
                                 columns=df.columns,
                                 index=df.index)
        return df_scaled


def analyse_each_variable(train_x, train_y):
    plot_single_var_to_fvc_diff(train_x, train_y, 'WeeksDiff')
    plot_single_var_to_fvc_diff(train_x, train_y, 'FVC')
    plot_single_var_to_fvc_diff(train_x, train_y, 'Weeks')
    plot_single_var_to_fvc_diff(train_x, train_y, 'Age')
    plot_single_var_to_fvc_diff(train_x, train_y, 'Sex')
    # plot_single_var_to_fvc_diff(train_x, train_y, 'SmokingStatus')


def plot_actual_fvc_vs_predicted_in_training_set(fvc_predicted, fvc_true):
    plt.figure()
    plt.xlabel('actual FVC at secondary checkup')
    plt.ylabel('predicted FVC at secondary checkup')
    plt.scatter(fvc_true, fvc_predicted, alpha=0.2)
    plt.show()
    ols_model = sm.OLS(fvc_true, fvc_predicted)
    predictor = ols_model.fit()
    print(predictor.summary())


def calc_tensor_score(fvc_true, fvc_predicted):
    tf.dtypes.cast(fvc_true, tf.float32)
    tf.dtypes.cast(fvc_predicted, tf.float32)
    sigma = tf.dtypes.cast(np.ones(fvc_true.shape[0]) * 230, tf.float32)
    fvc_pred = fvc_predicted[:, 0]

    # sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, MINIMAL_SCORE_STD)
    delta = tf.abs(fvc_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, MAXIMAL_SCORE_DIFF)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return keras.backend.mean(metric)


def calc_score(fvc_true, fvc_predicted, std):
    assert fvc_true.shape == fvc_predicted.shape
    assert fvc_true.shape == std.shape

    std_clipped = np.maximum(std, MINIMAL_SCORE_STD)
    delta = np.minimum(np.abs(fvc_true - fvc_predicted), MAXIMAL_SCORE_DIFF)
    metric = (-1 * sqrt(2) * delta).divide(std_clipped) - np.log(sqrt(2) * std_clipped)
    # plt.hist(metric, bins=20)
    return np.mean(metric)


def calc_confidence(fvc_prediction, weeks_diff, fvc_fraction, confidence_baseline):
    scaled_weeks_diff_multiplier = 40
    base_line_weight = 0.6
    fvc_prediction_weight = 1 - base_line_weight
    # stay close to data-driven confidence_baseline
    confidence_baseline_term = np.ones(fvc_prediction.shape[0]) * confidence_baseline
    fvc_proportion_term = fvc_prediction * fvc_fraction  # less confident for higher fvc values
    # we are less confident as time from first measurement increases
    time_dependent_term = weeks_diff * scaled_weeks_diff_multiplier
    return pd.Series(fvc_proportion_term * fvc_prediction_weight
                     + confidence_baseline_term * base_line_weight
                     + time_dependent_term,
                     name='Confidence').astype(int)


def make_test_data_to_model_compatible(test_data, label_encoder, scaler):
    test_data_copy = test_data.copy()  # copy df to avoid side-affects
    possible_weeks = pd.DataFrame(np.arange(-12, 133 + 1), columns=['PossibleWeeks'])
    possible_weeks['cross_product_key'] = 1
    test_data_copy['cross_product_key'] = 1
    df = test_data_copy.merge(possible_weeks, on='cross_product_key')
    df.loc[:, 'WeeksDiff'] = df['PossibleWeeks'] - df['Weeks']

    df = df.drop('cross_product_key', axis=1)
    df.loc[:, 'SmokingStatus'] = label_encoder.transform(df['SmokingStatus'])
    df.loc[:, 'Sex'] = (df['Sex'] == 'Male') + 0

    test_x_dont_scale = df[['Patient', 'PossibleWeeks']]
    patient_week = pd.Series(test_x_dont_scale['Patient'] + '_'
                             + test_x_dont_scale['PossibleWeeks'].astype(str),
                             name='Patient_Week')
    df = df.drop(['Patient', 'PossibleWeeks', 'Percent'], axis=1)
    df_scaled = scale_features(df, scaler)
    return df, df_scaled, patient_week


def preprocess_training_data(train_data, verbose, scaler_name):
    base_and_secondary_pairs = group_base_and_secondary_measurements(train_data)
    base_and_secondary_pairs_enc, label_encoder = encode_categories(base_and_secondary_pairs)
    fvc_true = base_and_secondary_pairs['FVCTarget']
    fvc_initial = base_and_secondary_pairs_enc['FVC']
    # prepare final training-set and labels
    fvc_diff = base_and_secondary_pairs_enc['FVCDiff']
    train_x = define_features(base_and_secondary_pairs_enc)
    if verbose > 1:
        analyse_each_variable(train_x, fvc_diff)
    x_scaled, scaler = scale_features(train_x, scaler_name=scaler_name)
    return fvc_initial, fvc_true, label_encoder, scaler, x_scaled, fvc_diff


def preprocess_validation_data(val_data, scaler, label_encoder):
    val_base_and_secondary_pairs = group_base_and_secondary_measurements(val_data)
    val_base_and_secondary_pairs_enc = encode_categories(val_base_and_secondary_pairs, fit=False,
                                                         label_encoder=label_encoder)
    fvc_true_val = val_base_and_secondary_pairs['FVCTarget']
    fvc_initial_val = val_base_and_secondary_pairs_enc['FVC']
    # prepare final training-set and labels
    x_val = define_features(val_base_and_secondary_pairs_enc)
    x_scaled_val = scale_features(x_val, scaler)
    return fvc_initial_val, fvc_true_val, x_scaled_val


def cross_validate(k, data, regressor):
    patients = data['Patient'].unique()
    kf = KFold(n_splits=k)
    num_of_confidence_levels = 10
    initial_confidence_base_line = 200 # close to optimum point
    confidence_levels = initial_confidence_base_line + np.arange(num_of_confidence_levels) * 5
    sum_score_per_conf_level = np.zeros(confidence_levels.shape[0])
    sum_training_score = 0
    for train_index, val_index in kf.split(patients):
        train_patients = patients[train_index]
        val_patients = patients[val_index]
        validation_set = data[data['Patient'].isin(val_patients)]
        training_set = data[data['Patient'].isin(train_patients)]
        predictor, label_encoder, scaler, training_score = train(training_set, regressor, verbose=0)
        sum_training_score += training_score
        fvc_initial_val, fvc_true_val, x_scaled_val = preprocess_validation_data(validation_set, scaler, label_encoder)
        # x_scaled_val = x_scaled_val[training_set.columns]
        predicted_fvc_diff = predictor.predict(x_scaled_val)
        if regressor == 'neural_net':
            predicted_fvc_diff = pd.Series(predicted_fvc_diff.flatten(), index=fvc_initial_val.index)

        fvc_predicted_val = predicted_fvc_diff + fvc_initial_val
        # plot_actual_fvc_vs_predicted_in_training_set(fvc_predicted_val, fvc_true_val)

        for i in np.arange(confidence_levels.shape[0]):
            # confidence = calc_confidence(fvc_predicted_val, confidence_levels[i]) # -6.817702
            confidence = calc_confidence(fvc_predicted_val, x_scaled_val['WeeksDiff'], 0.09, confidence_levels[i])
            score = calc_score(fvc_predicted_val, fvc_true_val, confidence)
            if i==3:
                print(score)
            sum_score_per_conf_level[i] += score

    mean_score_per_conf_level = sum_score_per_conf_level / k
    print(f'\n{k}-fold mean competition score of {regressor}, per confidence level:')
    tunning_df = pd.concat([pd.Series(confidence_levels, name='confidence_baseline'),
                            pd.Series(mean_score_per_conf_level, name='score')],
                           axis=1)
    print(tunning_df.to_string())
    best_confidence_baseline = tunning_df['confidence_baseline'][tunning_df['score'].argmax()]
    best_val_score = tunning_df['score'].max()
    print(f'\n{regressor} training score = {sum_training_score / k}')
    print(f'{regressor} best_val_confidence_baseline = {best_confidence_baseline}')
    print(f'{regressor} best_val_score = {best_val_score}\n')
    return best_confidence_baseline


def build_and_compile_nn(num_of_features, layer_size=100, dropout_rate=0.2, verbose=0):
    # Build the model
    model = keras.models.Sequential([
        keras.layers.Dense(layer_size, input_shape=[num_of_features], activation='relu', kernel_regularizer=regularizers.l1(0.001)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(int(layer_size / 2), activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1),
    ])
        #keras.layers.Dense(3, activation='relu'),
        #keras.layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis = 1))

    #     z = L.Input((nh,), name="Patient")
    # x = L.Dense(100, activation="relu", name="d1")(z)
    # x = L.Dense(100, activation="relu", name="d2")(x)
    # p1 = L.Dense(3, activation="linear", name="p1")(x)
    # p2 = L.Dense(3, activation="relu", name="p2")(x)
    # preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1),
    #                  name="preds")([p1, p2])

    if verbose > 0:
        print(model.summary())

    # Construct loss function
    #loss_fn = calc_tensor_score
    #loss_fn = mloss(0.8)
    loss_fn = 'mae'

    # compile loss function into model
    model.compile(optimizer=keras.optimizers.Adam(lr=nn_config.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=nn_config.learning_rate/10, amsgrad=False),
                  loss=loss_fn,
                  metrics=['mae'])
    return model


def train(train_meta, regressor, verbose=0):
    scaler_name = get_scaler_matched_to_regressor(regressor)

    fvc_initial_train, fvc_true_train, label_encoder, scaler, x_scaled_train, fvc_diff_train = \
        preprocess_training_data(train_meta, verbose, scaler_name)

    # Fit and summarize OLS model
    ols_model = sm.OLS(fvc_diff_train, x_scaled_train)
    ols_predictor = ols_model.fit()
    if verbose > 0:
        print(ols_predictor.summary())

    if regressor == 'random_forest':
        predictor = RandomForestRegressor(n_estimators=40, random_state=RANDOM_STATE, max_leaf_nodes=5)
        predictor.fit(x_scaled_train, fvc_diff_train)
    elif regressor == 'grad_boost':
        predictor = XGBRegressor(n_estimators=25, learning_rate=0.05, n_jobs=4, max_depth=3, random_state=RANDOM_STATE)
        predictor.fit(x_scaled_train, fvc_diff_train)
    elif regressor == 'ols':
        predictor = ols_predictor
    elif regressor == 'neural_net':
        predictor = build_and_compile_nn(x_scaled_train.shape[1], layer_size=nn_config.layer_size, dropout_rate=nn_config.dropout_rate, verbose=0)
        predictor.fit(x_scaled_train, fvc_diff_train, epochs=nn_config.epochs, batch_size=nn_config.batch_size, verbose=nn_config.fit_verbosity)
    else:
        raise RuntimeError('no such regressor: ' + regressor)

    # sanity check prediction on training-set
    predicted_fvc_diff = predictor.predict(x_scaled_train)
    if regressor == 'neural_net':
        predicted_fvc_diff = pd.Series(predicted_fvc_diff.flatten(), index=fvc_initial_train.index)

    fvc_predicted_training = predicted_fvc_diff + fvc_initial_train
    if verbose > 1:
        plot_actual_fvc_vs_predicted_in_training_set(fvc_predicted_training, fvc_true_train)
    confidence = calc_confidence(fvc_predicted_training, x_scaled_train['WeeksDiff'], 0.09, 230)
    score = calc_score(fvc_predicted_training, fvc_true_train, confidence)
    return predictor, label_encoder, scaler, score


def get_scaler_matched_to_regressor(regressor):
    if regressor == 'ols':
        scaler_name = 'min_max'
    elif regressor == 'random_forest':
        scaler_name = 'standard'
    elif regressor == 'grad_boost':
        scaler_name = 'standard'
    elif regressor == 'neural_net':
        scaler_name = 'standard'
    else:
        raise RuntimeError(f'No such regressor {regressor}')
    return scaler_name


def _predict(test_data, predictor, scaler, label_encoder, best_confidence_baseline, regressor):
    test_x, test_x_scaled, patient_week = make_test_data_to_model_compatible(test_data, label_encoder, scaler)
    print(test_x)
    fvc_diff_prediction = predictor.predict(test_x_scaled)
    if regressor == 'neural_net':
        fvc_diff_prediction = pd.Series(fvc_diff_prediction.flatten(), index=test_x_scaled.index)
    fvc_prediction = pd.Series(test_x['FVC'] + fvc_diff_prediction, name='FVC').astype(int)
    # set confidence above regression on training-set confidence interval
    confidence = calc_confidence(fvc_prediction, test_x_scaled['WeeksDiff'], 0.09, best_confidence_baseline)
    submission_df = pd.concat([patient_week, fvc_prediction, confidence], axis=1)
    return submission_df


def main():
    # Load meta-data
    train_data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
    test_data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

    cross_validate(10, train_data, 'ols')
    cross_validate(10, train_data, 'random_forest')
    cross_validate(10, train_data, 'grad_boost')
    # explore_data(train_meta)

    chosen_predictor = 'neural_net'
    best_confidence_baseline = cross_validate(10, train_data, chosen_predictor)

    predictor, label_encoder, scaler, score = train(train_data, chosen_predictor, verbose=1)
    submission_df = _predict(test_data, predictor, scaler, label_encoder, best_confidence_baseline, chosen_predictor)
    submission_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
else:
    main()
