import sys

import pandas as pd
import numpy as np
import shutil
import os
import logging

import time as time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb

COLUMNS_OF_INTEREST_NEWS = ['datetime_gmt',
                            'datetime',
                            'forecast_error',
                            'impact',
                            'new',
                            'previous_error',
                            'week',
                            'year',
                            'weekday',
                            'forecast_error_diff_deviation',
                            'forecast_error_diff_outlier_class',
                            'previous_error_diff_deviation',
                            'previous_error_diff_outlier_class'
                            ]


COLUMNS_MARKET_REACTION_BEFORE = ['volatility', 'pips_agg']

COLUMNS_MARKET_REACTION_AFTER_BASIC = ['volatility', 'pips_agg']

COLUMNS_MARKET_REACTION_AFTER_ALL = ['volatility', 'pips_agg', 'pips_candle']

COLUMNS_TO_AGG = ['forecast_error_diff_deviation',
                  'forecast_error_diff_outlier_class',
                  'previous_error_diff_deviation',
                  'previous_error_diff_outlier_class',
                  'fe_accurate',
                  'fe_better',
                  'fe_worse',
                  'pe_accurate',
                  'pe_better',
                  'pe_worse']

COLUMN_TO_PREDICT_CLF_PREFIX = 'direction_agg'
COLUMN_TO_PREDICT_REG_PREFIX = 'pips_agg'

FIELD_HIGH = 'High'
FIELD_MEDIUM = 'Medium'
FIELD_LOW = 'Low'

COLUMNS_IMPACT = [FIELD_HIGH, FIELD_MEDIUM, FIELD_LOW]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VAL = 5

THS_30 = 9
THS_60 = 11
THS_90 = 14
THS_120 = 15
THS_150 = 16
THS_180 = 17
THS_210 = 18
THS_240 = 19

PREDICTED_VALUES = [0,1,2]

OUTPUT_COLUMNS = ['model_type', 'model','sweeps_market_variables', 'sweep_news_agg','sweep_buy_sell','before_data',
                  'sweep_grid','best_score','best_params','f1_microavg', 'precision_weighted', 'precision_EUR_down',
                  'precision_EUR_same','precision_EUR_up','support_EUR_down','support_EUR_same','support_EUR_up',
                  'report','elapsed_time']


SNAPSHOT_OFFSET_BEFORE_RELEASE = 60
CANDLE_SIZE = 5

##################################################################


def set_logger(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def convert_categorical_to_numerical_news(df_news, groupby):

    # assign a unique ID to each new
    le = preprocessing.LabelEncoder()
    df_news['new_id'] = le.fit_transform(df_news['new'])
    dummy_forecast_error = pd.get_dummies(df_news['forecast_error'])
    dummy_forecast_error.columns = ['fe_accurate', 'fe_better', 'fe_worse']
    dummy_previous_error = pd.get_dummies(df_news['previous_error'])
    dummy_previous_error.columns = ['pe_accurate', 'pe_better', 'pe_worse']
    dummy_impact = pd.get_dummies(df_news['impact'])

    #df_look_up_table = df_news[['new_id', 'new']].drop_duplicates()
    #df_look_up_table.to_csv(os.path.join(base_output_path, 'look_up_table_' + groupby + '.csv'))

    df_news = df_news.drop(['new', 'datetime_gmt', 'forecast_error', 'impact', 'previous_error'], axis=1)
    df_news['new_id'] = df_news['new_id'].astype(str)

    df_news = pd.concat([df_news, dummy_forecast_error, dummy_previous_error, dummy_impact], axis=1)

    return df_news


def apply_weights(row, field_value, impact_filter, high_weight, medium_weight, low_weight):
    # values: 3_3
    # high: 0_1
    # medium: 1_0
    # low: 0_0

    if impact_filter == 'ALL':
        values = row[field_value]
        is_high = row[FIELD_HIGH]
        is_medium = row[FIELD_MEDIUM]
        is_low = row[FIELD_LOW]

        weights_high = [e * high_weight for e in is_high]
        weights_medium = [e * medium_weight for e in is_medium]
        weights_low = [e * low_weight for e in is_low]

        out = sum([a * b + a * c + a * d for a, b, c, d in zip(values, weights_high, weights_medium, weights_low)])
    else:
        values = row[field_value]
        out = sum(values)

    return out


def group_news_by_datetime(group_type, df):
    # ALL_NO_1_1_1, where
    #
    # f1: [ALL, HIGH, MEDIUM, LOW
    # f2: [YES, NO]. YES meaning just using news that were published in isolation. NO otherwise
    # f3: weight for HIGH news
    # f4: weight for MEDIUM news
    # f5: weight for LOW news
    #

    impact_filter = group_type.split('_')[0]
    standalone_filter = group_type.split('_')[1]
    high_weight = int(group_type.split('_')[2])
    medium_weight = int(group_type.split('_')[3])
    low_weight = int(group_type.split('_')[4])

    df_news = df[COLUMNS_OF_INTEREST_NEWS].copy()

    if impact_filter != 'ALL':
        df_news = df_news[df_news['impact'] == impact_filter]
        column_names_for_impact = [impact_filter]
    else:
        column_names_for_impact = COLUMNS_IMPACT

    df_news = convert_categorical_to_numerical_news(df_news, group_type)


    if impact_filter == 'ALL':
        df_news = df_news.groupby('datetime').agg({'new_id': lambda x: list(x),
                                                   'forecast_error_diff_deviation': lambda x: list(x),
                                                   'forecast_error_diff_outlier_class': lambda x: list(x),
                                                   'previous_error_diff_deviation': lambda x: list(x),
                                                   'previous_error_diff_outlier_class': lambda x: list(x),
                                                   'fe_accurate': lambda x: list(x),
                                                   'fe_better': lambda x: list(x),
                                                   'fe_worse': lambda x: list(x),
                                                   'pe_accurate': lambda x: list(x),
                                                   'pe_better': lambda x: list(x),
                                                   'pe_worse': lambda x: list(x),
                                                   'High': lambda x: list(x),
                                                   'Low': lambda x: list(x),
                                                   'Medium': lambda x: list(x),
                                                   'year': lambda x: list(x)[0],
                                                   'week': lambda x: list(x)[0],
                                                   'weekday': lambda x: list(x)[0]
                                                   }).reset_index()

    else:
        df_news = df_news.groupby('datetime').agg({'new_id': lambda x: list(x),
                                                   'forecast_error_diff_deviation': lambda x: list(x),
                                                   'forecast_error_diff_outlier_class': lambda x: list(x),
                                                   'previous_error_diff_deviation': lambda x: list(x),
                                                   'previous_error_diff_outlier_class': lambda x: list(x),
                                                   'fe_accurate': lambda x: list(x),
                                                   'fe_better': lambda x: list(x),
                                                   'fe_worse': lambda x: list(x),
                                                   'pe_accurate': lambda x: list(x),
                                                   'pe_better': lambda x: list(x),
                                                   'pe_worse': lambda x: list(x),
                                                   impact_filter: lambda x: list(x),
                                                   'year': lambda x: list(x)[0],
                                                   'week': lambda x: list(x)[0],
                                                   'weekday': lambda x: list(x)[0]
                                                   }).reset_index()

    df_news['num_news'] = df_news['new_id'].apply(lambda x: len(x))
    df_news['new_id'] = df_news['new_id'].apply(lambda x: ['_'.join(x)])
    df_news['new_id'] = df_news['new_id'].apply(lambda x: x[0])

    for field in COLUMNS_TO_AGG:
        df_news[field] = df_news.apply(lambda row: apply_weights(row, field, impact_filter, high_weight,
                                                                 medium_weight, low_weight),
                                       axis=1)

    for field in column_names_for_impact:
        df_news[field] = df_news[field].apply(lambda x: sum(x))

    if standalone_filter == 'YES':
        df_news = df_news[df_news['num_news'] == 1]

    return df_news


def model_fit_and_classify(clf_model, model_name, X_train, y_train, X_test, y_test,
                          sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                          df_results):
    time_init = time.time()

    clf_model.fit(X_train, y_train)
    y_predict = clf_model.predict(X_test)

    clf_report = classification_report(y_test, y_predict, output_dict=True)


    total_time = format((time.time() - time_init) / 60, '.2f')


    df_results = df_results.append({'model_type': 'clf',
                                    'model': model_name,
                                    'sweeps_market_variables': sweeps_market_variables,
                                    'sweep_news_agg': sweeps_new,
                                    'sweep_buy_sell': sweep_buy_sell,
                                    'before_data': before_data,
                                    'sweep_grid': sweep_grid,
                                    'best_score': format(clf_model.best_score_, '.2f'),
                                    'best_params': str(clf_model.best_params_),
                                    'f1_microavg': format(clf_report['micro avg']['f1-score'], '.2f'),
                                    'precision_weighted': format(clf_report['weighted avg']['precision'], '.2f'),
                                    'precision_EUR_down': format(clf_report['0']['precision'], '.2f'),
                                    'precision_EUR_same': format(clf_report['1']['precision'], '.2f'),
                                    'precision_EUR_up': format(clf_report['2']['precision'], '.2f'),
                                    'support_EUR_down': format(clf_report['0']['support'], '.2f'),
                                    'support_EUR_same': format(clf_report['1']['support'], '.2f'),
                                    'support_EUR_up': format(clf_report['2']['support'], '.2f'),
                                    'report': str(clf_report),
                                    'elapsed_time': total_time},
                                   ignore_index=True)

    return df_results


def model_fit_and_predict(reg_model, model_name, X_train, y_train, X_test, y_test,
                          sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                          df_results):

    time_init = time.time()

    reg_model.fit(X_train, y_train)
    y_predict = reg_model.predict(X_test)

    # MSE formula
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
    mse_test = np.sqrt(sum([diff**2 for diff in (y_test - y_predict)]) / len(y_test))


    total_time = format((time.time() - time_init) / 60, '.2f')


    df_results = df_results.append({'model_type': 'reg',
                                    'model': model_name,
                                    'sweeps_market_variables': sweeps_market_variables,
                                    'sweep_news_agg': sweeps_new,
                                    'sweep_buy_sell': sweep_buy_sell,
                                    'before_data': before_data,
                                    'sweep_grid': sweep_grid,
                                    'best_score': format(np.sqrt(-reg_model.best_score_), '.2f'),
                                    'best_params': str(reg_model.best_params_),
                                    'f1_microavg': '',
                                    'precision_weighted': '',
                                    'precision_EUR_down': '',
                                    'precision_EUR_same': '',
                                    'precision_EUR_up': '',
                                    'support_EUR_down': '',
                                    'support_EUR_same': '',
                                    'support_EUR_up': '',
                                    'report': format(mse_test, '.2f'),
                                    'elapsed_time': total_time},
                                   ignore_index=True)

    return df_results


def run_models_basic_grid(df_news_sweep, sweep_market_feature, sweep_how_agg_news, sweep_include_data_before_release,
                          sweep_when_buy_sell,columns_to_predict):

    return pd.concat([run_classification_models_basic_grid(df_news_sweep,sweep_market_feature,sweep_how_agg_news,
                                            sweep_include_data_before_release,sweep_when_buy_sell,columns_to_predict),
                      run_regression_models_basic_grid(df_news_sweep, sweep_market_feature, sweep_how_agg_news,
                                            sweep_include_data_before_release, sweep_when_buy_sell, columns_to_predict),
                      ])


def run_regression_models_basic_grid(df_news, sweep_market_variables, sweep_new, before_data, sweep_buy_sell,
                                         columns_to_predict):

    df_results = pd.DataFrame(columns = OUTPUT_COLUMNS)

    columns_of_model = list(set(df_news.columns) - set(columns_to_predict) - set(['datetime']))

    logging.info('X columns for the regression model: {}'.format(columns_of_model))

    X = df_news[columns_of_model].values
    y = df_news[columns_to_predict[1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # KNeighborsRegressor
    logging.info('Starting KNeighborsRegressor')
    reg_kn = GridSearchCV(KNeighborsRegressor(n_neighbors=1),
                          param_grid={"n_neighbors": range(1, 200, 10)},
                          scoring="neg_mean_squared_error",
                          cv=CROSS_VAL)

    df_results = model_fit_and_predict(reg_kn, 'kn', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # SVR - rbf
    logging.info('Starting SVR - rbf')
    reg_svr = GridSearchCV(SVR(kernel="rbf"),
                           param_grid={"C": [1, 3, 5], "gamma": range(1, 5)},
                           scoring="neg_mean_squared_error",
                           cv=CROSS_VAL)

    df_results = model_fit_and_predict(reg_svr, 'svr-rbf', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # DecisionTreeRegressor
    logging.info('Starting DecisionTreeRegressor')
    reg_dt = GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE),
                          param_grid={'min_samples_leaf': [10, 20, 30, 50, 100, 150, 200, 250],
                                      'max_depth': range(2, 8)},
                          scoring="neg_mean_squared_error",
                          cv=CROSS_VAL)

    df_results = model_fit_and_predict(reg_dt, 'dtree', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # RandomForest
    logging.info('Starting RandomForestRegressor')
    reg_rf = GridSearchCV(RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
                          param_grid={"min_samples_leaf": [10, 20, 30, 50, 100, 150, 200, 250],
                                      'max_depth': range(2, 8),
                                      'n_estimators': [10, 50, 100, 200]},
                          scoring="neg_mean_squared_error",
                          cv=CROSS_VAL)

    df_results = model_fit_and_predict(reg_rf, 'rforest', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # XGBoost
    logging.info('Starting XGBRegressor')
    reg_XGB = GridSearchCV(xgb.XGBRegressor( random_state=RANDOM_STATE),
                            param_grid={#'learning_rate': [0.01, 0.1, 0.5, 0.9], 'subsample': [0.3, 0.5, 0.9],
                                       #'max_depth': [3, 5, 10],
                                     'n_estimators': [10, 50, 100]},
                           scoring="neg_mean_squared_error",
                           cv=CROSS_VAL)

    df_results = model_fit_and_predict(reg_XGB, 'xgb', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # GradientBoostingRegressor
    logging.info('Starting GradientBoostingRegressor')
    reg_gb = GridSearchCV(GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
                          param_grid={'n_estimators': [10, 50, 100, 200]},
                          scoring="neg_mean_squared_error",
                          cv=CROSS_VAL)

    df_results = model_fit_and_predict(reg_gb,'gboosting', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # AdaBoost
    logging.info('Starting AdaBoostRegressor')
    reg_ada = GridSearchCV(AdaBoostRegressor(n_estimators=200, random_state=RANDOM_STATE),
                           param_grid={'n_estimators': [10, 50, 100, 200]},
                           scoring="neg_mean_squared_error",
                           cv=CROSS_VAL)

    df_results = model_fit_and_predict(reg_ada, 'ada', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    return df_results


def run_classification_models_basic_grid(df_news, sweep_market_variables, sweep_new, before_data, sweep_buy_sell,
                                         columns_to_predict):

    df_results = pd.DataFrame(columns = OUTPUT_COLUMNS)

    columns_of_model = list(set(df_news.columns) - set(columns_to_predict) - set(['datetime']))

    logging.info('X columns for the classification model: {}'.format(columns_of_model))
    logging.info('y dependent variable: {}'.format(columns_to_predict[0]))

    X = df_news[columns_of_model].values
    y = df_news[columns_to_predict[0]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # KNeighborsClassifier
    logging.info('Starting KNeighborsClassifier')
    clf_kn = GridSearchCV(KNeighborsClassifier(n_neighbors=1),
                          param_grid={"n_neighbors": range(1, 200, 10)},
                          scoring="accuracy",
                          cv=CROSS_VAL)

    df_results = model_fit_and_classify(clf_kn, 'kn', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # SVC - rbf
    logging.info('Starting SVC - rbf')
    clf_svc = GridSearchCV(SVC(kernel="rbf"),
                           param_grid={"C": [1, 3, 5], "gamma": range(1, 5)},
                           scoring="accuracy",
                           cv=CROSS_VAL)

    df_results = model_fit_and_classify(clf_svc, 'svc-rbf', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # SVC - poly
    #logging.info('Starting SVC - Poly')
    #clf_poly = GridSearchCV(SVC(kernel="poly", gamma="auto"),
                             #param_grid={"C": [1, 3, 5], "degree": [2, 3] },
    #                        param_grid={},
    #                         scoring="accuracy",
    #                         cv=CROSS_VAL)

    #df_results = model_fit_and_classify(clf_poly, 'svc-poly', X_train, y_train, X_test, y_test,
    #                                   sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
    #                                   df_results)

    # DecisionTree
    logging.info('Starting DecisionTreeClassifier')
    clf_dt = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                          param_grid={'min_samples_leaf': [10, 20, 30, 50, 100, 150, 200, 250],
                                      'max_depth': range(2, 8)},
                          scoring="accuracy",
                          cv=CROSS_VAL)

    df_results = model_fit_and_classify(clf_dt, 'dtree', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # RandomForest
    logging.info('Starting RandomForestClassifier')
    clf_rf = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
                          param_grid={"min_samples_leaf": [10, 20, 30, 50, 100, 150, 200, 250],
                                      'max_depth': range(2, 8),
                                      'n_estimators': [10, 50, 100, 200]},
                          scoring="accuracy",
                          cv=CROSS_VAL)

    df_results = model_fit_and_classify(clf_rf, 'rforest', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # XGBoost
    logging.info('Starting XGBClassifier')
    clf_XGB = GridSearchCV(xgb.XGBClassifier( random_state=RANDOM_STATE),
                            param_grid={#'learning_rate': [0.01, 0.1, 0.5, 0.9], 'subsample': [0.3, 0.5, 0.9],
                                       #'max_depth': [3, 5, 10],
                                     'n_estimators': [10, 50, 100]},
                           scoring="accuracy",
                           cv=CROSS_VAL)

    df_results = model_fit_and_classify(clf_XGB, 'xgb', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # GradientBoosting
    logging.info('Starting GradientBoostingClassifier')
    clf_gb = GridSearchCV(GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
                          param_grid={'n_estimators': [10, 50, 100, 200]},
                          scoring="accuracy",
                          cv=CROSS_VAL)

    df_results = model_fit_and_classify(clf_gb,'gboosting', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # AdaBoost
    logging.info('Starting AdaBoostClassifier')
    clf_ada = GridSearchCV(AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
                           param_grid={'n_estimators': [10, 50, 100, 200]},
                           scoring="accuracy",
                           cv=CROSS_VAL)

    df_results = model_fit_and_classify(clf_ada, 'ada', X_train, y_train, X_test, y_test,
                                       sweep_market_variables, sweep_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    return df_results


def get_dynamic_market_fields_after(candles_5m, snapshots_after, type):

    dynamic_market_fields = []

    if type == 'all':
        column_names = COLUMNS_MARKET_REACTION_AFTER_ALL
    elif type == 'basic':
        column_names = COLUMNS_MARKET_REACTION_AFTER_BASIC
    else:
        logging.error('ERROR: Bad parameters to get market features, just "all" or "basic" are supported')
        exit(-1)

    for candles in candles_5m:
        tmp = [column + '_' + str(candles - 5) + '_' + str(candles) + '_after' for column in column_names]
        dynamic_market_fields = dynamic_market_fields + tmp

    # We don´t have pips_candle and other advanced metrics in the individual snapshots
    for snapshot in snapshots_after:
        tmp = [column + '_0_' + str(snapshot) + '_after' for column in COLUMNS_MARKET_REACTION_AFTER_BASIC]
        dynamic_market_fields = dynamic_market_fields + tmp

    return dynamic_market_fields





def get_dynamic_market_fields_before():

    column_names = COLUMNS_MARKET_REACTION_BEFORE
    dynamic_market_fields = [column + '_' + str(SNAPSHOT_OFFSET_BEFORE_RELEASE) +'_0_before' for column in column_names]

    return dynamic_market_fields

########################################################################################################################
#
#   DESCRIPTION:
#
#       Function to classify the new release based on the market impact
#
#           0: USD decreases w.r.t EUR in > THs value pips
#           1: almost no impact
#           2: USD increases w.r.t EUR in < THs value pips
#
#   INPUT PARAMETERS:
#
#       num_pips:           pips difference
#       sell_after:         timestamp to sell
#
########################################################################################################################

def get_class(num_pips, sell_after):

    if sell_after == 30: threshold = THS_30
    elif sell_after == 60: threshold = THS_60
    elif sell_after == 90: threshold = THS_90
    elif sell_after == 120: threshold = THS_120
    elif sell_after == 150: threshold = THS_150
    elif sell_after == 180: threshold = THS_180
    elif sell_after == 210: threshold = THS_210
    elif sell_after == 240: threshold = THS_240

    if num_pips < threshold * -1 :
        out = PREDICTED_VALUES[0]
    elif num_pips > threshold:
        out = PREDICTED_VALUES[2]
    else: out = PREDICTED_VALUES[1]

    return out

if __name__ == '__main__':


    sweeps_when_buy_sell = list(sys.argv[1].split(','))  # e.g 15_60, predict 60 min after the release using info up to 15min
    sweeps_how_agg_news = list(sys.argv[2].split(','))  # e.g. ALL_NO_1_1_1, HIGH_YES_1_0_0
    sweep_market_features = list(sys.argv[3].split(','))
    sweeps_include_data_before_release = list(sys.argv[4].split(','))
    sweeps_grid_mode = list(sys.argv[5].split(','))

    path_to_input_df = sys.argv[6]
    candles_5m = list(sys.argv[7].split(','))
    snapshots_after = list(sys.argv[8].split(','))

    base_output_path = sys.argv[9]
    exp_name = sys.argv[10]

    candles_5m = [int(e) for e in candles_5m]
    snapshots_after = [int(e) for e in snapshots_after]

    set_logger(os.path.join(base_output_path, exp_name + '.log'))

    df = pd.read_csv(path_to_input_df)
    df_results = pd.DataFrame(columns=OUTPUT_COLUMNS)

    partial_results_path = os.path.join(base_output_path, exp_name)
    shutil.rmtree(partial_results_path, ignore_errors=True)
    os.mkdir(partial_results_path)

    for sweep_how_agg_news in sweeps_how_agg_news:

        logging.info('#########################')
        logging.info('')
        logging.info('Starting a new sweep, grouping news by: {}'.format(sweep_how_agg_news))

        df_news = group_news_by_datetime(sweep_how_agg_news, df)

        for sweep_market_feature in sweep_market_features:

            logging.info('Starting a new sweep with market features: {}'.format(sweep_market_feature))

            for sweep_include_data_before_release in sweeps_include_data_before_release:

                logging.info('Starting a new sweep, features before the publication are: {}'.format(
                                                                                sweep_include_data_before_release))

                for sweep_when_buy_sell in sweeps_when_buy_sell:

                    logging.info('Starting a new sweep, delay in between buying-selling is: {}'.format(sweep_when_buy_sell))

                    for sweep_grid in sweeps_grid_mode:

                        logging.info('Starting a new sweep for grid mode: {}'.format(sweep_grid))

                        sweep_name = sweep_how_agg_news + '-' + sweep_market_feature + '-' \
                                    + sweep_include_data_before_release + '-' \
                                     + sweep_when_buy_sell + '-' + sweep_grid

                        start_time = time.time()
                        buy_delay = int(sweep_when_buy_sell.split('_')[0])
                        sell_after = int(sweep_when_buy_sell.split('_')[1])

                        if sell_after in snapshots_after and buy_delay in snapshots_after:

                            predict_at = '_' + str(buy_delay) + '_' + str(sell_after) + '_after'

                            columns_to_predict = [column_name + predict_at for column_name in
                                                            [COLUMN_TO_PREDICT_CLF_PREFIX, COLUMN_TO_PREDICT_REG_PREFIX]]

                            candles_5m_tmp = [e for e in candles_5m if e <= buy_delay]
                            snapshots_after_tmp = [e for e in snapshots_after if e <= buy_delay]
                            market_fields_after = get_dynamic_market_fields_after(candles_5m_tmp,
                                                                                  snapshots_after_tmp,
                                                                                  sweep_market_feature)

                            if sweep_include_data_before_release == 'included':
                                market_fields_before = get_dynamic_market_fields_before()
                            else:
                                market_fields_before = []


                            market_fields = market_fields_after + market_fields_before + ['datetime'] \
                                            + ['pips_agg_0_' + str(sell_after) + '_after']

                            logging.info('List of dynamic market fields: {}'.format(market_fields))

                            df_market = df[market_fields].copy()

                            # As some news are published in bundle, we need to drop duplicates in the market df
                            df_market = df_market.drop_duplicates()

                            # We compute the pips difference in that window time
                            snapshots_buy_delay = 'pips_agg_0_' + str(buy_delay) + '_after'
                            snapshots_sell_after = 'pips_agg_0_' + str(sell_after) + '_after'
                            df_market[columns_to_predict[1]] = df_market[snapshots_sell_after] - \
                                                               df_market[snapshots_buy_delay]

                            # we create the variable holding the classification to predict
                            df_market[columns_to_predict[0]] = df_market[columns_to_predict[1]].apply(
                                                                            lambda x: get_class(x, sell_after))

                            # drop the extra column created to compute pips differences
                            df_market = df_market.drop(['pips_agg_0_' + str(sell_after) + '_after'], axis=1)

                            df_news_sweep = df_news.merge(df_market, on='datetime', how='left')

                            df_news_sweep.to_csv(os.path.join(base_output_path, sweep_name + '.csv'))

                            if sweep_grid == 'basic':
                                df_results = pd.concat([df_results, run_models_basic_grid(
                                    df_news_sweep,
                                    sweep_market_feature,
                                    sweep_how_agg_news,
                                    sweep_include_data_before_release,
                                    sweep_when_buy_sell,
                                    columns_to_predict)])


                            total_time = format((time.time() - start_time) / 60, '.2f')
                            logging.info('sweep done in: {} min'.format(total_time))

                            # Save the data computed so far in case the program crashes
                            df_results.to_csv(os.path.join(partial_results_path, sweep_name + '_summary_partial.csv'))

                        else:
                            logging.error('sweep not allowed. Buy delay: {} - Sell after: {}'.format(buy_delay, sell_after))
                            logging.error('Both values should be in the list of 30min snapshots')

    df_results.to_csv(os.path.join(base_output_path, exp_name + '_models_performance.csv'))
    shutil.rmtree(partial_results_path, ignore_errors=True)
