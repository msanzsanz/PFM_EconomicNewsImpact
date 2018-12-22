import sys

import pandas as pd
import numpy as np

import logging

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

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


COLUMNS_MARKET_REACTION = ['close', 'low', 'high', 'volatility', 'pips_agg', 'pips_candle']



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


def apply_weights(row, field_value, high_weight, medium_weight, low_weight):

    # values: 3_3
    # high: 0_1
    # medium: 1_0
    # low: 0_0

    values = row[field_value]
    is_high = row[FIELD_HIGH]
    is_medium = row[FIELD_MEDIUM]
    is_low = row[FIELD_LOW]

    weights_high = [e * high_weight for e in is_high]
    weights_medium = [e * medium_weight for e in is_medium]
    weights_low = [e * low_weight for e in is_low]

    out = sum([a * b + a * c + a * d for a,b,c,d in zip(values, weights_high, weights_medium, weights_low)])
    return out


def apply_news_grouping(group_type, df):

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

    df_news = convert_categorical_to_numerical_news(df_news)

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


    df_news['num_news'] = df_news['new_id'].apply(lambda x: len(x))
    df_news['new_id'] = df_news['new_id'].apply(lambda x: ['_'.join(x)])
    df_news['new_id'] = df_news['new_id'].apply(lambda x: x[0])

    for field in COLUMNS_TO_AGG:
        df_news[field] = df_news.apply(lambda row: apply_weights(row, field, high_weight, medium_weight, low_weight), axis=1)

    for field in COLUMNS_IMPACT:
        df_news[field] = df_news[field].apply(lambda x: sum(x))

    if standalone_filter == 'YES':
        df_news = df_news[df_news['num_news'] == 1]

    return df_news




def get_clf_performance(df_news, sweeps_new, before_data, sweep_buy_sell, columns_to_predict):

    df_results = pd.DataFrame([])

    # remove those columns that we don´t want to use for predictions (i.e > buy_delay)
    direction = columns_to_predict[0]

    df_news[direction] = np.where(df_news[direction] == 'up', 0,
                                                    np.where(df_news[direction] == 'down', 1,
                                                    2))

    columns_of_model = list(set(df_news.columns) - set(columns_to_predict) - set(['datetime']))

    logging.info('X columns for the model: {}'.format(columns_of_model))

    X = df_news[columns_of_model].values
    y = df_news[direction].values

    # KNeighborsClassifier
    clf_kn = GridSearchCV(KNeighborsClassifier(n_neighbors=1),
                          param_grid={"n_neighbors": [5, 10, 15, 20]},
                          scoring="accuracy",
                          cv=5)
    clf_kn.fit(X, y)
    df_results = df_results.append({'model': 'clf', 'sweep_news_agg': sweeps_new, 'sweep_buy_sell': sweep_buy_sell,
                                    'before_data': before_data, 'score': clf_kn.best_score_,
                                    'params': str(clf_kn.best_params_)}, ignore_index=True)

    # DecisionTree
    clf_dt = GridSearchCV(DecisionTreeClassifier(),
                            param_grid = {'min_samples_leaf': [10,20,30,50,100], 'max_depth':range(2,7)},
                            scoring = "accuracy",
                            cv = 5)
    clf_dt.fit(X, y)
    df_results = df_results.append({'model': 'clf', 'sweep_news_agg': sweeps_new, 'sweep_buy_sell': sweep_buy_sell,
                                    'before_data': before_data, 'score': clf_kn.best_score_,
                                    'params': clf_dt.best_params_}, ignore_index=True)


    # RandomForest
    clf_rf = GridSearchCV(RandomForestClassifier(n_estimators=100, oob_score=True),
                        param_grid={"min_samples_leaf": [50, 100, 150, 200]},
                        scoring="accuracy",
                        cv=5)

    clf_rf.fit(X, y)
    df_results = df_results.append({'model': 'clf', 'sweep_news_agg': sweeps_new, 'sweep_buy_sell': sweep_buy_sell,
                                    'before_data': before_data, 'score': clf_kn.best_score_,
                                    'params': clf_rf.best_params_}, ignore_index=True)

    # XGBoost
    clf_XGB = GridSearchCV(XGBClassifier(n_estimators=100),
                          param_grid={},
                          scoring="accuracy",
                          cv=5)

    clf_XGB.fit(X, y)
    df_results = df_results.append({'model': 'clf', 'sweep_news_agg': sweeps_new, 'sweep_buy_sell': sweep_buy_sell,
                                    'before_data': before_data, 'score': clf_kn.best_score_,
                                    'params': clf_XGB.best_params_}, ignore_index=True)


    # GradientBoosting
    clf_gb = GridSearchCV(GradientBoostingClassifier(n_estimators=100),
                          param_grid={},
                          scoring="accuracy",
                          cv=5)

    clf_gb.fit(X, y)
    df_results = df_results.append({'model': 'clf', 'sweep_news_agg': sweeps_new, 'sweep_buy_sell': sweep_buy_sell,
                                    'before_data': before_data, 'score': clf_kn.best_score_,
                                    'params': clf_gb.best_params_}, ignore_index=True)

    return df_results


def get_dynamic_market_fields_after(snapshots_5m, snapshots_15m, snapshots_30m):

    dynamic_market_fields = []

    for snapshot in snapshots_5m:
        tmp = [column + '_' + str(snapshot - 5) + '_' + str(snapshot) + '_after' for column in COLUMNS_MARKET_REACTION]
        dynamic_market_fields = dynamic_market_fields + tmp

    for snapshot in snapshots_15m:
        tmp = [column + '_' + str(snapshot - 15) + '_' + str(snapshot) + '_after' for column in COLUMNS_MARKET_REACTION]
        dynamic_market_fields = dynamic_market_fields + tmp

    for snapshot in snapshots_30m:
        tmp = [column + '_' + str(snapshot - 30) + '_' + str(snapshot) + '_after' for column in COLUMNS_MARKET_REACTION]
        dynamic_market_fields = dynamic_market_fields + tmp

    return  dynamic_market_fields



def get_dynamic_market_fields_before(snapshots_5m, snapshots_15m, snapshots_30m):

    dynamic_market_fields = []

    for snapshot in snapshots_5m:
        tmp = [column + '_' + str(snapshot) + '_' + str(snapshot - 5) + '_before' for column in COLUMNS_MARKET_REACTION]
        dynamic_market_fields = dynamic_market_fields + tmp

    for snapshot in snapshots_15m:
        tmp = [column + '_' + str(snapshot) + '_' + str(snapshot - 15) + '_before' for column in COLUMNS_MARKET_REACTION]
        dynamic_market_fields = dynamic_market_fields + tmp

    for snapshot in snapshots_30m:
        tmp = [column + '_' + str(snapshot) + '_' + str(snapshot - 30) + '_before' for column in COLUMNS_MARKET_REACTION]
        dynamic_market_fields = dynamic_market_fields + tmp

    return dynamic_market_fields


if __name__ == '__main__':
    '''
    Run this script using the command 'python `script_name`.py 
    '''

    sweeps_buy_sell = list(sys.argv[1].split(',')) # e.g 15_60, meaning predict after 60 min using info up to 15 min
    sweeps_news_agg = list(sys.argv[2].split(',')) # e.g. ALL_NO_1_1_1, HIGH_YES_1_0_0

    data_path = sys.argv[3]
    log_file = sys.argv[4]
    snapshots_5m = list(sys.argv[5].split(','))
    snapshots_15m = list(sys.argv[6].split(','))
    snapshots_30m = list(sys.argv[7].split(','))

    snapshots_5m = [int(e) for e in snapshots_5m]
    snapshots_15m = [int(e) for e in snapshots_15m]
    snapshots_30m = [int(e) for e in snapshots_30m]

    set_logger(log_file)

    df = pd.read_csv(data_path)
    df_results = pd.DataFrame(columns=['model', 'sweep_news_agg', 'sweep_buy_sell', 'score', 'params'])

    for sweep_new in sweeps_news_agg:

        df_news = apply_news_grouping(sweep_new, df)

        for before_data in ['included', 'excluded']:

            for sweep_buy_sell in sweeps_buy_sell:

                buy_delay = int(sweep_buy_sell.split('_')[0])
                sell_after = int(sweep_buy_sell.split('_')[1])

                if buy_delay in snapshots_5m and sell_after in snapshots_30m:

                    predict_at = '_' + str(sell_after - 30) + '_' +  str(sell_after) + '_after'

                    columns_to_predict = [column_name + predict_at for column_name in COLUMS_TO_PREDICT_PREFIX]
                    logging.info('Sweep for buy_delay: {} sell_after: {}'.format(buy_delay, sell_after))

                    snapshots_5m_tmp = [ e for e in snapshots_5m if e <= buy_delay]
                    snapshots_15m_tmp = [e for e in snapshots_15m if e <= buy_delay]
                    snapshots_30m_tmp = [e for e in snapshots_30m if e <= buy_delay]
                    market_fields_after = get_dynamic_market_fields_after(snapshots_5m_tmp, snapshots_15m_tmp, snapshots_30m_tmp)

                    if before_data == 'included':
                        market_fields_before = get_dynamic_market_fields_before(snapshots_5m, snapshots_15m, snapshots_30m)
                    else:
                        market_fields_before = []

                    market_fields = market_fields_after + market_fields_before + ['datetime'] + columns_to_predict

                    logging.info('List of dynamic market fields: {}'.format(market_fields))

                    df_market = df[market_fields].copy()
                    df_market = df_market.drop_duplicates()
                    df_news = df_news.merge(df_market, on='datetime', how='left')

                    df_results = pd.concat([df_results, get_clf_performance(df_news,
                                                                            sweep_new,
                                                                            before_data,
                                                                            sweep_buy_sell,
                                                                            columns_to_predict)])


                else:
                    logging.error('sweep not allowed. Buy delay: {} - Sell after: {}'.format(buy_delay, sell_after))
                    logging.error('Buy delay should be within the 5 min snapshots and Sell delay should be within the 30 min snapshots')



