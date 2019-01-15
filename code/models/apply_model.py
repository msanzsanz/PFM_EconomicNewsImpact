
import sys
import models_sweeps
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

MARKET_FIELDS = ['volatility_0_5_after', 'pips_agg_0_5_after', 'volatility_5_10_after', 'pips_agg_5_10_after',
                 'volatility_10_15_after', 'pips_agg_10_15_after', 'volatility_15_20_after', 'pips_agg_15_20_after',
                 'volatility_20_25_after', 'pips_agg_20_25_after', 'volatility_25_30_after', 'pips_agg_25_30_after',
                 'volatility_0_30_after', 'pips_agg_0_30_after', 'volatility_0_60_after', 'pips_agg_0_60_after',
                 'volatility_60_0_before', 'pips_agg_60_0_before', 'datetime', 'pips_agg_0_120_after']

COLUMNS_TO_PREDICT = ['direction_agg_0_120_after', 'pips_agg_0_120_after']
BUY_DELAY = 0
SELL_AFTER = 30
GROUPING_STRATEGY = 'ALL_NO_1_1_1'

def prepare_df_for_model (df):

    processed_df = models_sweeps.group_news_by_datetime(GROUPING_STRATEGY, df)

    market_df = df[MARKET_FIELDS].copy()
    market_df = market_df.drop_duplicates()
    market_df[COLUMNS_TO_PREDICT[0]] = market_df[COLUMNS_TO_PREDICT[1]].apply(lambda x: models_sweeps
                                                                              .get_class(x, BUY_DELAY, SELL_AFTER))

    processed_df = processed_df.merge(market_df, on='datetime', how='left')

    return processed_df

def apply_models(train_df, test_df):

    processed_train_df = prepare_df_for_model(train_df)

    columns_of_model = list(set(processed_train_df.columns) - set(COLUMNS_TO_PREDICT) - set(['datetime']))

    # Classification model
    X_train = processed_train_df[columns_of_model].values
    y_train = processed_train_df[COLUMNS_TO_PREDICT[0]].values

    clf_kn = KNeighborsClassifier(n_neighbors=11)
    clf_kn.fit(X_train, y_train)

    processed_test_df = prepare_df_for_model(test_df)
    X_test = processed_test_df[columns_of_model].values
    y_test = processed_test_df[COLUMNS_TO_PREDICT[0]].values

    y_predict = clf_kn.predict(X_test)
    test_df['direction_agg_0_120_after_predicted'] = y_predict

    clf_report = classification_report(y_test, y_predict, output_dict=True)
    print('Classification result: {}'.format(clf_report))

    # Regression model
    y_train = processed_train_df[COLUMNS_TO_PREDICT[1]].values
    y_test = processed_test_df[COLUMNS_TO_PREDICT[1]].values

    reg_kn = GridSearchCV(KNeighborsRegressor(n_neighbors=1),
                          param_grid={"n_neighbors": range(1, 200, 10)},
                          scoring="neg_mean_squared_error",
                          cv=CROSS_VAL)
    reg_kn.fit(X_train, y_train)
    y_predict = reg_kn.predict(X_test)
    test_df['direction_agg_0_120_after_predicted'] = y_predict

    mse_test = np.sqrt(sum([diff ** 2 for diff in (y_test - y_predict)]) / len(y_test))
    print('Regression result: {}'.format(mse_test))

    test_df = test_df.filter['datetime', 'direction_agg_0_120_after_predicted', 'direction_agg_0_120_after_predicted']
    return test_df


if __name__ == '__main__':

    #
    # PRE-REQUISITES for this script is to have the dataframe with the features extracted.
    #
    # For so:
    # 1. Parse data from forexfactory
    # python forexfactory_scraper.py calendar.php?week=oct14.2006 calendar.php?week=dec23.2018 52 ../../data/demo
    #
    # 2. Download data from dukascopy
    # Go to the url: https://www.dukascopy.com/swiss/english/marketwatch/historical/
    #
    # 3. Process and merge both data sources
    # python dc_forexfactory.py  2007 2018  ../../data/raw/ forexfactory_ USD EURUSD [5,10,15,20,25,30] [0,30,60]
    # [30,60,120,180,240] ON ../../data/demo/ features dc_forexfactory.log
    #

    features_file = sys.argv[1]
    datetime_split = sys.argv[2]

    features_df = pd.read_csv(features_file)
    train_df = features_df[features_df['datetime'] < datetime_split]
    test_df = features_df[features_df['datetime'] >= datetime_split]

    df_test_with_models_predictions = apply_models(train_df,test_df)
    # This will only populate the outcome from the models for those datetimes we have predictions for
    # i.e. all the training group will have nulls
    features_df = features_df.merge(df_test_with_models_predictions, on='datetime', how='left')

    result_filename = features_file.split('.csv')[0] + '_model_predictions.csv'
    features_df.to_csv(result_filename)



