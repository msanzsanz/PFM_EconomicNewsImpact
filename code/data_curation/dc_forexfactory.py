import logging
import sys, ast
import pandas as pd
import numpy as np
import pytz
import datetime


def set_logger(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def compute_diff(row):
    try:

        forecasted = row['forecast']
        actual = row['actual']
        values = [forecasted, actual]

        # Eliminate non-digits at the start
        for pos, val in enumerate(values):
            i = 0
            while not val[i].isdigit() and val[i] != '-':
                i = i + 1

            values[pos] = val[i:]

        # Eliminate non-digits at the end
        for pos, val in enumerate(values):
            i = -1
            while not val[i].isdigit():
                i = i - 1

            if i != -1: values[pos] = val[0:i + 1]

        forecasted_float = float(values[0])
        actual_float = float(values[1])

        if (values[0] == values[1]):
            diff = 0

        else:
            if forecasted_float == 0:
                if actual_float == 0:
                    diff = 0
                elif abs(actual_float) >= 1:
                    diff = actual_float * 100
                elif abs(actual_float) >= 0.1:
                    diff = actual_float * 1000
                elif abs(actual_float) >= 0.01:
                    diff = actual_float * 10000
                else:
                    diff = 9999
            else:
                diff_per = abs(actual_float - forecasted_float) * 100 / abs(forecasted_float)
                sign = 1 if actual_float < forecasted_float else -1
                diff = diff_per * sign

    except:
        diff = 9999

    return diff


def compute_diff_old(row):
    try:

        forecasted = row['forecast']
        actual = row['actual']

        if (forecasted == actual):
            diff = 0
        else:
            if forecasted[0] == '-':
                start = 1
            else:
                start = 0

            starts_digit = forecasted[start].isdigit()
            ends_digit = forecasted[-1].isdigit()

            if not starts_digit:
                forecasted = forecasted[start + 1:]
                actual = actual[start + 1:]

            if not ends_digit:
                forecasted = forecasted[0:-1]
                actual = actual[0:-1]

            forecasted_float = float(forecasted)
            actual_float = float(actual)

            if forecasted_float == 0:
                if abs(actual_float) >= 1:
                    diff = actual_float * 100
                elif abs(actual_float) >= 0.1:
                    diff = actual_float * 1000
                elif abs(actual_float) >= 0.01:
                    diff = actual_float * 10000
                else:
                    diff = 9999
            else:
                diff_per = abs(actual_float - forecasted_float) * 100 / abs(forecasted_float)
                sign = 1 if actual_float >= forecasted_float else -1
                diff = diff_per * sign

    except:
        diff = 9999

    return diff


def add_dts_flag(df):
    # Create a list of start and end dates for US in each year, in UTC time
    dst_changes_utc = pytz.timezone('US/Eastern')._utc_transition_times[1:]

    # Convert to local times from UTC times and then remove timezone information
    dst_changes = [pd.Timestamp(i).tz_localize('UTC').tz_convert('US/Eastern').tz_localize(None) for i in
                   dst_changes_utc]

    flag_list = []
    for index, row in df['datetime'].iteritems():
        # Isolate the start and end dates for DST in each year
        dst_dates_in_year = [date for date in dst_changes if date.year == row.year]
        spring = dst_dates_in_year[0]
        fall = dst_dates_in_year[1]
        if (row >= spring) & (row < fall):
            flag = 1
        else:
            flag = 0
        flag_list.append(flag)

    return flag_list


def apply_dts_flag(row):
    return row['datetime'] + pd.DateOffset(hours=row['dts_flag'])


def group_forexite_by_freq(df_pair, frequency='5Min'):
    # We need to add a column with the GMT datetime, so that we can join the rate exchange with news publication
    df_pair['datetime'] = df_pair['<DTYYYYMMDD>'] + df_pair['<TIME>']
    df_pair['datetime_gmt'] = pd.to_datetime(df_pair['datetime'], format='%Y%m%d%H%M%S', errors='raise')
    df_pair['datetime_gmt'] = df_pair['datetime_gmt'].dt.tz_localize('GMT').dt.tz_convert('GMT')
    df_pair = df_pair.set_index('datetime_gmt')

    # Remove undesired columns
    df_pair = df_pair.drop(['<DTYYYYMMDD>', '<TIME>', '<VOL>', 'datetime'], axis=1)
    df_pair.columns = ['pair', 'open', 'high', 'low', 'close']

    # Group by 5-min window size
    # IMP: it´s needed to specify the variable "closed". Otherwise, the group is not taking into account the last min
    df_pair = df_pair.groupby(pd.Grouper(freq=frequency, closed='right', label='left')).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})

    # GroupBy could have created some nan rows if data from forexite contains windows GAPS
    df_pair = df_pair.dropna()

    # Replace strings by int so that we can compute pips
    for field in ['high', 'low', 'close']:
        df_pair[field] = df_pair[field].apply(lambda x: format(x, '.4f'))
        df_pair[field] = df_pair[field].str.replace('.', '')
        df_pair[field] = df_pair[field].astype(int)

    return df_pair


def fe_joined_with_dukascopy(df_features, df_pair, snapshots, freq=5):
    try:

        # Expand the forexfactory dataframe with as many snapshots as requested
        last_column_name = '_released'
        df_features[last_column_name] = df_features['datetime_gmt'] - pd.DateOffset(minutes=freq)
        df_features = df_features.set_index(last_column_name).join(df_pair)
        df_features = df_features.reset_index(drop=True)
        df_features.rename({'open': 'open' + last_column_name, \
                            'high': 'high' + last_column_name, \
                            'low': 'low' + last_column_name, \
                            'close': 'close' + last_column_name}, axis='columns', inplace='True')

        logging.info(df_features[df_features.isnull().any(1)].values)

        for snapshot in snapshots:
            column_name = '_' + str(snapshot)
            df_features[column_name] = df_features['datetime_gmt'] + pd.DateOffset(minutes=snapshot - freq)
            df_features = df_features.set_index(column_name).join(df_pair)
            df_features = df_features.reset_index(drop=True)

            df_features['volatility'] = abs(df_features['high'] - df_features['low'])
            df_features['direction'] = np.where(df_features['close'] > df_features['close' + last_column_name], 'up',
                                                'down')
            df_features['pips_agg'] = abs(df_features['close_released'] - df_features['close'])
            df_features['pips_candle'] = abs(df_features['close'] - df_features['open'])

            # Drop undesired columns
            df_features = df_features.drop(['open'], axis=1)

            df_features.rename({'close': 'close' + column_name,
                                'low': 'low' + column_name,
                                'high': 'high' + column_name, \
                                'volatility': 'volatility' + column_name, \
                                'direction': 'direction' + column_name, \
                                'pips_agg': 'pips_agg' + column_name,
                                'pips_candle': 'pips_candle' + column_name}, \
                               inplace=True, axis='columns')

            last_column_name = column_name

        return df_features

    except BaseException as e:
        logging.error('Error while extracting features from the currency pair')
        logging.error('exception: {}'.format(e))
        return pd.DataFrame()


def compute_deviation(df, size=5):
    # Unique list of news
    news_list = df['new'].unique()
    df_out = pd.DataFrame([])

    for new in news_list:
        # sort by ascending datetime
        df_temp = df[df['new'] == new].sort_values(by='datetime_gmt', ascending=True).reset_index()

        # compute mean and std of the last events
        df_temp['prediction_mean'] = df_temp['prediction_error'].rolling(window=size, min_periods=1).mean()
        df_temp['prediction_std'] = df_temp['prediction_error'].rolling(window=size, min_periods=1).std().fillna(1)
        df_temp['prediction_zscore'] = ((df_temp['prediction_error'] - df_temp['prediction_mean']) /
                                         df_temp['prediction_std']).fillna(0)


        df_out = df_out.append(df_temp)

    return df_out


def fe_forexfactory(year, ff_file, currency, freq='5min'):
    df = pd.read_csv(input_path + ff_file)
    weeks = len(df.week.unique())

    # 2018 is still not finished
    # ForexFactory does not publish the first week of 2007
    if weeks == 52 or year == 2018 or year == 2007:

        df['datetime'] = pd.to_datetime(df['datetime'])

        # Filter_macro_economic_news
        df = df[df['forecast'].notnull()]

        # Filter currency of interest
        df = df[df['country'] == currency]

        # replace nan values for categorical fields
        df['forecast_error'] = df['forecast_error'].replace(np.nan, 'accurate', regex=True)
        df['previous_error'] = df['previous_error'].replace(np.nan, 'accurate', regex=True)

        # When DST is off, we need to add +1h to forexfactory.com values during winter tz
        # We do that in 2 steps. First, compute dst flag. Second, add +1h whenever the flag is set to 1.
        df['dts_flag'] = add_dts_flag(df)
        df['datetime'] = df.apply(apply_dts_flag, axis=1)
        df['datetime_gmt'] = df['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert('GMT')

        # Some news are not published at o´clocks (i.e. neither 2:00 nor 2:30, but 1:59)
        # We rounded them to the closest 5 min candle.
        df['datetime_gmt'] = df['datetime_gmt'].dt.round(freq)

        if len(df[df.isnull().any(1)]) != 0:
            logging.info('These {} news extracted from forexfactory have nan in some columns'.format(
                len(df[df.isnull().any(1)])))
            logging.info(df[df.isnull().any(1)].values)
            df = df.dropna()

        # Compute the error, in %, between actual values and forecasted
        df['prediction_error'] = df.apply(compute_diff, axis=1)
        df['prediction_error'] = df['prediction_error'].round(2)

        # We have used 9999 to flag those times when we have not been able to compute error rate
        errors_found = len(df[df['prediction_error'] == 9999])
        if errors_found != 0:
            logging.info(
                'Unkown values appeared in the forecast - actual values: {} times.\n'.format(errors_found))
            logging.info(df[df['prediction_error'] == 9999].values)

        # Add categorical values related with datetime
        df['year'] = df['datetime'].dt.year
        df['quarter'] = df['datetime'].dt.quarter
        df['month'] = df['datetime'].dt.month
        df['day_name'] = df['datetime'].dt.day_name()

        # Drop undesired columns
        df.drop(columns=['Unnamed: 0', 'dts_flag'], axis=1, inplace=True)

        return df

    else:
        logging.info('Error {}: this dataset does not have the expected 52 weeks\n'.format(year))
        return pd.DataFrame()


def read_df_dukascopy(filename):
    df_pair = pd.read_csv(filename, header=0, sep=',')
    df_pair.columns = ['datetime_gmt', 'open', 'high', 'low', 'close', 'volume']
    df_pair['datetime_gmt'] = pd.to_datetime(df_pair['datetime_gmt'], format='%d.%m.%Y %H:%M:%S.000', errors='raise')
    df_pair['datetime_gmt'] = df_pair['datetime_gmt'].dt.tz_localize('GMT')
    df_pair = df_pair.set_index('datetime_gmt')
    df_pair = df_pair.drop('volume', axis=1)

    # Replace strings by int so that we can compute pips
    for field in ['open', 'high', 'low', 'close']:
        df_pair[field] = df_pair[field].apply(lambda x: format(x, '.4f'))
        df_pair[field] = df_pair[field].str.replace('.', '')
        df_pair[field] = df_pair[field].astype(int)

    return df_pair


if __name__ == '__main__':
    '''
    Run this using the command 'python `script_name`.py 
    '''
    # i/p
    year_start = int(sys.argv[1])
    year_end = int(sys.argv[2])
    input_path = str(sys.argv[3])
    csv_prefix_ff = sys.argv[4]
    currency_news = sys.argv[5]
    currency_pair = sys.argv[6]
    snapshots = ast.literal_eval(sys.argv[7])

    # o/p
    output_path = sys.argv[8]
    csv_prefix_out = sys.argv[9]
    log_file = sys.argv[10]

    set_logger(log_file)

    all_years_df = pd.DataFrame([])

    # Read the dataframe from dukascopy. Just one file for all years
    df_pair = read_df_dukascopy(input_path + currency_pair + '.zip')

    # Merge features
    for year in range(year_start, year_end + 1):

        logging.info('Processing year: {} for news from: {} in the pair: {}'.format(year, currency_news, currency_pair))

        # Read the dataframe as scrapped from forexfactory
        ff_file = csv_prefix_ff + str(year) + '.csv'

        # Feature extraction from forexfactory data
        df_features = fe_forexfactory(year, ff_file, currency_news)
        df_features.to_csv(output_path + csv_prefix_ff + str(year) + '_curated.csv', index=False)

        if len(df_features) != 0:

            # Feature extraction from the pair exchange value
            df_features = fe_joined_with_dukascopy(df_features, df_pair, snapshots)
            if len(df_features) != 0:

                # Save processed dataframe to disk
                # df_features = df_features.convert_objects(convert_numeric=True)
                df_features.to_csv(output_path + csv_prefix_out + '_' + str(year) + '.csv', index=False)
                all_years_df = all_years_df.append(df_features)

            else:
                logging.error('Error ocurred when extracting features from forexite')

        else:
            logging.error('Error ocurred when extracting features from forexfactory')

        logging.info('Processed year: {} for news from: {} in the pair: {}'.format(year, currency_news, currency_pair))

    logging.info('Adding z-score for deviation between forecast and actual')
    all_years_df = compute_deviation(all_years_df)

    all_years_df.to_csv(output_path + csv_prefix_out + '_' + str(year_start) + '_' + str(year_end) + '.csv',
                        index=False)

    logging.info('End')
