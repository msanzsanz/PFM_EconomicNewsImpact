
import sys
import models_sweeps
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

#
# ,model_type,model,sweeps_market_variables,sweep_news_agg,sweep_buy_sell,before_data,sweep_grid,best_score,best_params,f1_microavg,precision_weighted,precision_EUR_down,precision_EUR_same,precision_EUR_up,support_EUR_down,support_EUR_same,support_EUR_up,report,elapsed_time
# 0,clf,kn,basic,ALL_NO_1_1_1,60_120,included,basic,0.63,{'n_neighbors': 11},0.65,0.66,0.64,0.63,0.72,332.00,618.00,342.00,
# "{'0': {'precision': 0.6387832699619772, 'recall': 0.5060240963855421, 'f1-score': 0.5647058823529412, 'support': 332}, '1': {'precision': 0.6273525721455459, 'recall': 0.8090614886731392, 'f1-score': 0.7067137809187279, 'support': 618}, '2': {'precision': 0.7241379310344828, 'recall': 0.49122807017543857, 'f1-score': 0.5853658536585364, 'support': 342}, 'micro avg': {'precision': 0.6470588235294118, 'recall': 0.6470588235294118, 'f1-score': 0.6470588235294118, 'support': 1292}, 'macro avg': {'precision': 0.6634245910473352, 'recall': 0.6021045517447067, 'f1-score': 0.6189285056434018, 'support': 1292}, 'weighted avg': {'precision': 0.6559095260271802, 'recall': 0.6470588235294118, 'f1-score': 0.6381010770125154, 'support': 1292}}",6.19

market_fields = ['volatility_0_5_after', 'pips_agg_0_5_after', 'volatility_5_10_after', 'pips_agg_5_10_after',
                 'volatility_10_15_after', 'pips_agg_10_15_after', 'volatility_15_20_after', 'pips_agg_15_20_after',
                 'volatility_20_25_after', 'pips_agg_20_25_after', 'volatility_25_30_after', 'pips_agg_25_30_after',
                 'volatility_0_30_after', 'pips_agg_0_30_after', 'volatility_0_60_after', 'pips_agg_0_60_after',
                 'volatility_60_0_before', 'pips_agg_60_0_before', 'datetime', 'pips_agg_0_120_after']

columns_to_predict = ['direction_agg_0_120_after', 'pips_agg_0_120_after']

def prepare_df_for_model (df):

    processed_df = models_sweeps.group_news_by_datetime('ALL_NO_1_1_1', df)

    market_df = df[market_fields].copy()
    market_df = market_df.drop_duplicates()
    market_df[columns_to_predict[0]] = market_df[columns_to_predict[1]].apply(lambda x: models_sweeps.get_class(x, 60))

    processed_df = processed_df.merge(market_df, on='datetime', how='left')

    return processed_df

def apply_model_clf(train_df, test_df):

    processed_df = prepare_df_for_model(train_df)

    columns_of_model = list(set(processed_df.columns) - set(columns_to_predict) - set(['datetime']))

    X_train = processed_df[columns_of_model].values
    y_train = processed_df[columns_to_predict[0]].values

    clf_kn = KNeighborsClassifier(n_neighbors=11)
    clf_kn.fit(X_train, y_train)

    processed_df = prepare_df_for_model(test_df)
    X_test = processed_df[columns_of_model].values
    y_test = processed_df[columns_to_predict[0]].values

    y_predict = clf_kn.predict(X_test)

    clf_report = classification_report(y_test, y_predict, output_dict=True)

    processed_df['direction_agg_0_120_after_predicted'] = y_predict
    processed_df.to_csv('/Users/wola/Documents/MSS/Personales/GitRepos/PFM_EconomicNewsImpact/models/model_results.csv')

    print(clf_report)

    return 0


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
    # 3. Process data
    # python dc_forexfactory.py 2007 2018 ../../data/demo/ forexfactory_ USD EURUSD [5,10,15,20,25,30]
    # [30,60,90,120,150,180,210,240] ON ../../data/demo/ features dc_forexfactory.log
    #
    # 4. Split the data in training - test
    # split_train_test.ipynb
    #


    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    apply_model_clf(train_df,test_df)



