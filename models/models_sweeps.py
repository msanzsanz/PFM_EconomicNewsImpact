import sys

import pandas as pd
import shutil
import os
import logging

import time as time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb

COLUMNS_OF_INTEREST_NEWS = ['datetime',
                            'forecast_error',
                            'impact',
                            'new',
                            'previous_error',
                            'datetime_gmt',
                            'forecast_error_ratio',
                            'previous_error_ratio',
                            'total_error_ratio',
                            'forecast_error_ratio_zscore',
                            'total_error_ratio_zscore']

COLUMNS_MARKET_REACTION_ALL = ['close', 'low', 'high', 'volatility', 'pips_agg',
                               'pips_candle', 'direction_candle', 'direction_agg']

COLUMNS_MARKET_REACTION_BASIC = ['volatility', 'pips_agg', 'direction_agg']

COLUMNS_TO_AGG = ['forecast_error_ratio',
                  'previous_error_ratio',
                  'total_error_ratio',
                  'forecast_error_ratio_zscore',
                  'total_error_ratio_zscore',
                  'fe_accurate',
                  'fe_better',
                  'fe_worse',
                  'pe_accurate',
                  'pe_better',
                  'pe_worse']

COLUMS_TO_PREDICT_PREFIX = ['direction_agg', 'pips_agg']

FIELD_HIGH = 'High'
FIELD_MEDIUM = 'Medium'
FIELD_LOW = 'Low'

COLUMNS_IMPACT = [FIELD_HIGH, FIELD_MEDIUM, FIELD_LOW]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CROSS_VAL = 5

PREDICTED_VALUES = [-2, -1, 0, 1, 2]


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


def convert_categorical_to_numerical_news(df_news):
    # assign a unique ID to each new
    le = preprocessing.LabelEncoder()
    df_news['new_id'] = le.fit_transform(df_news['new'])
    dummy_forecast_error = pd.get_dummies(df_news['forecast_error'])
    dummy_forecast_error.columns = ['fe_accurate', 'fe_better', 'fe_worse']
    dummy_previous_error = pd.get_dummies(df_news['previous_error'])
    dummy_previous_error.columns = ['pe_accurate', 'pe_better', 'pe_worse']
    dummy_impact = pd.get_dummies(df_news['impact'])

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

    df_news = convert_categorical_to_numerical_news(df_news)

    if impact_filter == 'ALL':
        df_news = df_news.groupby('datetime').agg({'new_id': lambda x: list(x),
                                                   'forecast_error_ratio': lambda x: list(x),
                                                   'forecast_error_ratio_zscore': lambda x: list(x),
                                                   'previous_error_ratio': lambda x: list(x),
                                                   'total_error_ratio': lambda x: list(x),
                                                   'total_error_ratio_zscore': lambda x: list(x),
                                                   'fe_accurate': lambda x: list(x),
                                                   'fe_better': lambda x: list(x),
                                                   'fe_worse': lambda x: list(x),
                                                   'pe_accurate': lambda x: list(x),
                                                   'pe_better': lambda x: list(x),
                                                   'pe_worse': lambda x: list(x),
                                                   'High': lambda x: list(x),
                                                   'Low': lambda x: list(x),
                                                   'Medium': lambda x: list(x)
                                                   }).reset_index()

    else:
        df_news = df_news.groupby('datetime').agg({'new_id': lambda x: list(x),
                                                   'forecast_error_ratio': lambda x: list(x),
                                                   'forecast_error_ratio_zscore': lambda x: list(x),
                                                   'previous_error_ratio': lambda x: list(x),
                                                   'total_error_ratio': lambda x: list(x),
                                                   'total_error_ratio_zscore': lambda x: list(x),
                                                   'fe_accurate': lambda x: list(x),
                                                   'fe_better': lambda x: list(x),
                                                   'fe_worse': lambda x: list(x),
                                                   'pe_accurate': lambda x: list(x),
                                                   'pe_better': lambda x: list(x),
                                                   'pe_worse': lambda x: list(x),
                                                   impact_filter: lambda x: list(x)
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

def get_prediction_rates(conf_matrix):

    correctly_predicted = []
    total_true = []

    for predict in PREDICTED_VALUES:

        correctly_predicted = correctly_predicted.append(conf_matrix[predict][predict])
        total_true = total_true.append(conf_matrix['All'][predict])

    return [a*100/b for a,b in zip(correctly_predicted, total_true)]


def model_fit_and_predict(clf_model, model_name, X_train, y_train, X_test, y_test,
                          sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                          df_results):
    time_init = time.time()

    clf_model.fit(X_train, y_train)
    y_predict = clf_model.predict(X_test)
    acc = format(accuracy_score(y_predict, y_test), '.2f')
    total_time = format((time.time() - time_init) / 60, '.2f')

    conf_matrix = pd.crosstab(y_test, y_predict, rownames=['True'], colnames=['Predicted'],margins=True)
    per_correctly_predicted = get_prediction_rates(conf_matrix)


    df_results = df_results.append({'model_type': 'clf',
                                    'model': model_name,
                                    'sweeps_market_variables': sweeps_market_variables,
                                    'sweep_news_agg': sweeps_new,
                                    'sweep_buy_sell': sweep_buy_sell,
                                    'before_data': before_data,
                                    'sweep_grid': sweep_grid,
                                    'best_score': format(clf_model.best_score_, '.2f'),
                                    'best_params': str(clf_model.best_params_),
                                    'accuracy_test': acc,
                                    'accuracy_per_class': print(per_correctly_predicted),
                                    'elapsed_time': total_time},
                                   ignore_index=True)

    return df_results


def run_classification_models_basic_grid(df_news, sweeps_market_variables, sweeps_new, before_data, sweep_buy_sell,
                                         columns_to_predict):
    df_results = pd.DataFrame(columns=['model', 'sweeps_market_variables',
                                       'sweep_news_agg', 'sweep_buy_sell',
                                       'score', 'params'])

    columns_of_model = list(set(df_news.columns) - set(columns_to_predict) - set(['datetime']))

    logging.info('X columns for the model: {}'.format(columns_of_model))

    X = df_news[columns_of_model].values
    y = df_news[columns_to_predict[0]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # KNeighborsClassifier
    logging.info('Starting KNeighborsClassifier')
    clf_kn = GridSearchCV(KNeighborsClassifier(n_neighbors=1),
                          param_grid={"n_neighbors": range(1, 100, 10)},
                          scoring="accuracy",
                          cv=CROSS_VAL)

    df_results = model_fit_and_predict(clf_kn, 'kn', X_train, y_train, X_test, y_test,
                                       sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # SVC - rbf
    logging.info('Starting SVC - rbf')
    clf_svc = GridSearchCV(SVC(kernel="rbf"),
                           param_grid={"C": [1, 3, 5], "gamma": range(1, 5)},
                           scoring="accuracy",
                           cv=CROSS_VAL)

    df_results = model_fit_and_predict(clf_svc, 'svc-rbf', X_train, y_train, X_test, y_test,
                                       sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # SVC - poly
    logging.info('Starting SVC - Poly')
    clf_poly = GridSearchCV(SVC(kernel="poly"),
                            #param_grid={"C": [1, 3, 5], "degree": [2, 3], "gamma": 'scale'},
                            param_grid={},
                            scoring="accuracy",
                            cv=CROSS_VAL)

    df_results = model_fit_and_predict(clf_poly, 'svc-poly', X_train, y_train, X_test, y_test,
                                       sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # DecisionTree
    logging.info('Starting DecisionTreeClassifier')
    time_init = time.time()
    clf_dt = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                          param_grid={'min_samples_leaf': [10, 20, 30, 50, 100, 150, 200, 250],
                                      'max_depth': range(2, 7)},
                          scoring="accuracy",
                          cv=CROSS_VAL)

    df_results = model_fit_and_predict(clf_dt, 'dtree', X_train, y_train, X_test, y_test,
                                       sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # RandomForest
    logging.info('Starting RandomForestClassifier')
    time_init = time.time()
    clf_rf = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
                          param_grid={"min_samples_leaf": [10, 20, 30, 50, 100, 150, 200, 250],
                                      'max_depth': range(2, 7),
                                      'n_estimators': [10, 50, 100, 200]},
                          scoring="accuracy",
                          cv=CROSS_VAL)

    df_results = model_fit_and_predict(clf_rf, 'rforest', X_train, y_train, X_test, y_test,
                                       sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # XGBoost
    logging.info('Starting XGBClassifier')
    time_init = time.time()
    clf_XGB = GridSearchCV(xgb.XGBClassifier(n_estimators=200, random_state=RANDOM_STATE),
                           # param_grid={'learning_rate': [0.01, 0.1, 0.5, 0.9], 'subsample': [0.3, 0.5, 0.9],
                           #            'max_depth': [3, 5, 10]},
                           param_grid={'n_estimators': [10, 50, 100, 200]},
                           scoring="accuracy",
                           cv=CROSS_VAL)

    df_results = model_fit_and_predict(clf_XGB, 'xgb', X_train, y_train, X_test, y_test,
                                       sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # GradientBoosting
    logging.info('Starting GradientBoostingClassifier')
    time_init = time.time()
    clf_gb = GridSearchCV(GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
                          param_grid={'n_estimators': [10, 50, 100, 200]},
                          scoring="accuracy",
                          cv=CROSS_VAL)

    df_results = model_fit_and_predict(clf_gb,'gboosting', X_train, y_train, X_test, y_test,
                                       sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    # AdaBoost
    logging.info('Starting AdaBoostClassifier')
    time_init = time.time()
    clf_ada = GridSearchCV(AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
                           param_grid={'n_estimators': [10, 50, 100, 200]},
                           scoring="accuracy",
                           cv=CROSS_VAL)

    df_results = model_fit_and_predict(clf_ada, 'ada', X_train, y_train, X_test, y_test,
                                       sweeps_market_variables, sweeps_new, sweep_buy_sell, before_data, sweep_grid,
                                       df_results)

    return df_results


def get_dynamic_market_fields_after(snapshots_5m, snapshots_15m, snapshots_30m, sweeps_market_variables):
    dynamic_market_fields = []

    if sweeps_market_variables == 'all':
        column_names = COLUMNS_MARKET_REACTION_ALL
    elif sweeps_market_variables == 'basic':
        column_names = COLUMNS_MARKET_REACTION_BASIC
    else:
        logging.error('option for market fields sweep is not supported')
        exit(-1)

    for snapshot in snapshots_5m:
        tmp = [column + '_' + str(snapshot - 5) + '_' + str(snapshot) + '_after' for column in column_names]
        dynamic_market_fields = dynamic_market_fields + tmp

    for snapshot in snapshots_15m:
        tmp = [column + '_' + str(snapshot - 15) + '_' + str(snapshot) + '_after' for column in column_names]
        dynamic_market_fields = dynamic_market_fields + tmp

    for snapshot in snapshots_30m:
        tmp = [column + '_' + str(snapshot - 30) + '_' + str(snapshot) + '_after' for column in column_names]
        dynamic_market_fields = dynamic_market_fields + tmp

    return dynamic_market_fields


def get_dynamic_market_fields_before(snapshots_5m, snapshots_15m, snapshots_30m, sweeps_market_variables):
    dynamic_market_fields = []

    if sweeps_market_variables == 'all':
        column_names = COLUMNS_MARKET_REACTION_ALL
    elif sweeps_market_variables == 'basic':
        column_names = COLUMNS_MARKET_REACTION_BASIC
    else:
        logging.error('option for market fields sweep is not supported')
        exit(-1)

    for snapshot in snapshots_5m:
        tmp = [column + '_' + str(snapshot) + '_' + str(snapshot - 5) + '_before' for column in column_names]
        dynamic_market_fields = dynamic_market_fields + tmp

    for snapshot in snapshots_15m:
        tmp = [column + '_' + str(snapshot) + '_' + str(snapshot - 15) + '_before' for column in column_names]
        dynamic_market_fields = dynamic_market_fields + tmp

    for snapshot in snapshots_30m:
        tmp = [column + '_' + str(snapshot) + '_' + str(snapshot - 30) + '_before' for column in column_names]
        dynamic_market_fields = dynamic_market_fields + tmp

    return dynamic_market_fields


if __name__ == '__main__':
    '''
    Run this script using the command 'python `script_name`.py 
    '''

    sweeps_when_buy_sell = list(sys.argv[1].split(','))  # e.g 15_60, predict after 60 min using info up to 15 min
    sweeps_how_agg_news = list(sys.argv[2].split(','))  # e.g. ALL_NO_1_1_1, HIGH_YES_1_0_0
    sweeps_market_variables = list(sys.argv[3].split(','))
    sweeps_include_data_before_release = list(sys.argv[4].split(','))
    sweeps_grid_mode = list(sys.argv[5].split(','))

    data_path = sys.argv[6]
    snapshots_5m = list(sys.argv[7].split(','))
    snapshots_15m = list(sys.argv[8].split(','))
    snapshots_30m = list(sys.argv[9].split(','))

    base_output_path = sys.argv[10]
    exp_name = sys.argv[11]

    snapshots_5m = [int(e) for e in snapshots_5m]
    snapshots_15m = [int(e) for e in snapshots_15m]
    snapshots_30m = [int(e) for e in snapshots_30m]

    set_logger(os.path.join(base_output_path, exp_name + '.log'))

    df = pd.read_csv(data_path)
    df_results = pd.DataFrame(columns=['model_type', 'model', 'sweeps_market_variables', 'sweep_news_agg',
                                       'sweep_buy_sell','before_data', 'sweep_grid', 'best_score', 'best_params', 'accuracy_test',
                                       'accuracy_per_class', 'elapsed_time'])

    partial_results_path = os.path.join(base_output_path, 'partial_results')

    shutil.rmtree(partial_results_path, ignore_errors=True)
    os.mkdir(partial_results_path)

    for sweep_how_agg_news in sweeps_how_agg_news:

        logging.info('#########################')
        logging.info('')
        logging.info('Starting a new sweep grouping News: {}'.format(sweep_how_agg_news))

        df_news = group_news_by_datetime(sweep_how_agg_news, df)

        for sweep_market_variables in sweeps_market_variables:

            logging.info('Starting a new sweep for market variables: {}'.format(sweep_market_variables))

            for sweep_include_data_before_release in sweeps_include_data_before_release:
                logging.info('Starting a new sweep, features before the publication are: {}'.format(
                    sweep_include_data_before_release))

                for sweep_when_buy_sell in sweeps_when_buy_sell:

                    logging.info(
                        'Starting a new sweep for delay in between buying-selling: {}'.format(sweep_when_buy_sell))

                    for sweep_grid in sweeps_grid_mode:

                        logging.info(
                            'Starting a new sweep for grid mode: {}'.format(sweep_grid))

                        sweep_name = sweep_how_agg_news + '-' + sweep_market_variables \
                                     + '-' + sweep_include_data_before_release + '-' \
                                     + sweep_when_buy_sell + '-' + sweep_grid

                        start_time = time.time()
                        buy_delay = int(sweep_when_buy_sell.split('_')[0])
                        sell_after = int(sweep_when_buy_sell.split('_')[1])

                        if sell_after in snapshots_30m:

                            predict_at = '_' + str(sell_after - 30) + '_' + str(sell_after) + '_after'

                            columns_to_predict = [column_name + predict_at for column_name in COLUMS_TO_PREDICT_PREFIX]

                            snapshots_5m_tmp = [e for e in snapshots_5m if e <= buy_delay]
                            snapshots_15m_tmp = [e for e in snapshots_15m if e <= buy_delay]
                            snapshots_30m_tmp = [e for e in snapshots_30m if e <= buy_delay]
                            market_fields_after = get_dynamic_market_fields_after(snapshots_5m_tmp, snapshots_15m_tmp,
                                                                                  snapshots_30m_tmp, sweep_market_variables)

                            if sweep_include_data_before_release == 'included':
                                market_fields_before = get_dynamic_market_fields_before(snapshots_5m, snapshots_15m,
                                                                                        snapshots_30m,
                                                                                        sweep_market_variables)
                            else:
                                market_fields_before = []

                            market_fields = market_fields_after + market_fields_before + ['datetime'] + columns_to_predict

                            logging.info('List of dynamic market fields: {}'.format(market_fields))

                            df_market = df[market_fields].copy()
                            df_market = df_market.drop_duplicates()
                            df_news_sweep = df_news.merge(df_market, on='datetime', how='left')

                            df_news_sweep.to_csv(os.path.join(base_output_path, sweep_name + '.csv'))

                            if sweep_grid == 'basic':
                                df_results = pd.concat([df_results, run_classification_models_basic_grid(
                                    df_news_sweep,
                                    sweep_market_variables,
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
                            logging.error('Sell delay should be in the list of 30min snapshots')

    df_results.to_csv(os.path.join(base_output_path, exp_name + '_models_performance.csv'))
    #shutil.rmtree(partial_results_path, ignore_errors=True)
