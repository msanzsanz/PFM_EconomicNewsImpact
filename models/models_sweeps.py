import sys, ast

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


COLUMNS_MARKET_REACTION = ['close', 'pips_agg', 'direction_agg', 'volatility', 'pips_candle']



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


def apply_weights(row, field_value, field_high, field_medium, field_low, high_weight, medium_weight, low_weight):

    # values: 3_3
    # high: 0_1
    # medium: 1_0
    # low: 0_0

    values = row[field_value]
    is_high = row[field_high]
    is_medium = row[field_medium]
    is_low = row[field_low]

    weights_high = [is_high * high_weight for e in is_high]
    weights_medium = [is_medium * medium_weight for e in is_medium]
    weights_low = [is_low * low_weight for e in is_low]

    out = [a * b + a * c + a * d for a,b,c,d in zip(values, weights_high, weights_medium, weights_low)]
    return out


def apply_news_grouping(group_type, df):

    impact_filter = group_type.split('_')[0]
    standalone_filter = group_type.split('_')[1]
    high_weight = group_type.split('_')[2]
    medium_weight = group_type.split('_')[3]
    low_weight = group_type.split('_')[4]

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
        df_news[field] = df_news.apply(lambda row: apply_weights(row, field, high_weight, medium_weight, low_weight))

    for field in ['High', 'Medium', 'Low']:
        df_news[field] = df_news[field].apply(lambda x: sum(x))

    if standalone_filter == 'YES':
        df_news = df_news[df_news['num_news'] == 1]

    return  df_news


def get_class_performance(group_type, df, buy_delay, sell_after, dynamic_market_fields):

    df_news = apply_news_grouping(group_type, df)
    df_market = df[dynamic_market_fields].copy()
    df_market = df_market.drop_duplicates()
    df_news = df_news.merge(df_market, on='datetime', how='left')

    # remove those that we don´t want to use for predictions
    columns_to_remove = get_dynamic_market_fields_after()

    columns_model = list(set(df_model_1.columns) - set(columns_to_predict) - set(['datetime']))


def get_dynamic_market_fields_after(snapshots_5m, snapshots_15m, snapshots_30m):

    for window_size in [5, 15, 30]:

        snapshots_name = snapshots + str(window_size) + 'm'
        for snapshots in snapshots_name:
            tmp = [column + '_' + str(snapshots) + '_' + str(snapshots + window_size) + '_after' for column in
                   COLUMNS_MARKET_REACTION]
            dynamic_market_fields.append(tmp)



def get_dynamic_market_fields_before(snapshots_5m, snapshots_15m, snapshots_30m):
    for window_size in [5, 15, 30]:

        snapshots_name = snapshots + str(window_size) + 'm'
        for snapshots in snapshots_name:

            tmp = [column + '_' + str(snapshots - window_size) + '_' + str(snapshots) + '_before' for column in
                   COLUMNS_MARKET_REACTION]
            dynamic_market_fields.append(tmp)

if __name__ == '__main__':
    '''
    Run this script using the command 'python `script_name`.py 
    '''

    buy_sell_sweeps = ast.literal_eval(sys.argv[1]) # e.g 15_60, meaning predict after 60 min using up to 15 min info
    news_grouping_sweeps = ast.literal_eval(sys.argv[2]) # e.g. ALL_NO_1_1_1, HIGH_YES_1_0_0
    data_path = sys.argv[3]
    log_file = sys.argv[4]
    snapshots_5m = ast.literal_eval(sys.argv[5])
    snapshots_15m = ast.literal_eval(sys.argv[6])
    snapshots_30m = ast.literal_eval(sys.argv[7])

    set_logger(log_file)

    df = pd.read_csv(data_path)
    df_results = pd.DataFrame([])
    dynamic_market_fields_after = get_dynamic_market_fields_after(snapshots_5m, snapshots_15m, snapshots_30m)
    dynamic_market_fields_before = get_dynamic_market_fields_after(snapshots_5m, snapshots_15m, snapshots_30m)
    dynamic_market_fields = dynamic_market_fields_after + dynamic_market_fields_before

    logging.info('List of dynamic market fields: {}', dynamic_market_fields)

    for new_grouping_sweep in news_grouping_sweeps:

        for pair in buy_sell_sweeps:

            buy_delay = str(pair).split('_')[0]
            sell_after = str(pair).split('_')[1]

            logging.info('Sweep for buy_delay: {} sell_after: {}'.format(buy_delay, sell_after))

            df_results.append(get_class_performance(new_grouping_sweep, df, buy_delay, sell_after, dynamic_market_fields))




