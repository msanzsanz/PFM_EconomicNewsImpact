import logging
import sys, ast
import pandas as pd
import numpy as np
import pytz
import math


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
    # We need to add a column with the UTC datetime, so that we can join the rate exchange with news publication
    df_pair['datetime'] = df_pair['<DTYYYYMMDD>'] + df_pair['<TIME>']
    df_pair['datetime_utc'] = pd.to_datetime(df_pair['datetime'], format='%Y%m%d%H%M%S', errors='raise', utc=True)
    df_pair = df_pair.set_index('datetime_utc')

    # Remove undesired columns
    df_pair = df_pair.drop(['<DTYYYYMMDD>', '<TIME>', '<VOL>', 'datetime'], axis=1)
    df_pair.columns = ['pair', 'open', 'high', 'low', 'close']

    # Group by 5-min window size
    df_pair = df_pair.groupby(pd.Grouper(freq=frequency)).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})

    # GroupBy could have created some nan rows if data from forexite contains windows GAPS
    df_pair = df_pair.dropna()

    # Create features of interest
    for field in ['high', 'low']:
        df_pair[field] = df_pair[field].apply(lambda x: format(x, '.4f'))
        df_pair[field] = df_pair[field].str.replace('.', '')
        df_pair[field] = df_pair[field].astype(int)

    df_pair['volatility'] = abs(df_pair['high'] - df_pair['low'])
    df_pair['direction'] = np.where(df_pair['close'] > df_pair['open'], 'up', 'down')

    # Drop undesired columns to join
    df_pair = df_pair.drop(['low', 'high'], axis=1)

    return df_pair


def fe_joined_with_forexite(df_features, df_pair, snapshots):
    try:

        # Expand the forexfactory dataframe with as many snapshots as requested
        for snapshot in snapshots:
            column_name = 'after_' + str(snapshot)
            df_features[column_name] = df_features['datetime_utc'] + pd.DateOffset(minutes=snapshot)
            df_features = df_features.set_index(column_name).join(df_pair)
            df_features = df_features.reset_index(drop=True)
            


            df_features = df_features.rename({'open': 'open_' + str(snapshot), \
                                              'close': 'close_' + str(snapshot), \
                                              'volatility': 'volatility_' + str(snapshot), \
                                              'direction': 'direction_' + str(snapshot)}, axis='columns')

        # Some news are not published on the o´clock time (i.e. neither 2:00 nor 2:30, but 1:59)
        # These are corner cases and occur < 3% of the times on low-impact news, so we are going to remove them for now.
        logging.info(
            'Unable to get currency values for {} of the news'.format(len(df_features[df_features.isnull().any(1)])))
        logging.info(df_features[df_features.isnull().any(1)].values)
        df_features.dropna(inplace=True)

        return df_features

    except BaseException as e:
        logging.error('Error while extracting features from the currency pair')
        logging.error('exception: {}'.format(e.message))
        return pd.DataFrame()


def fe_forexfactory(year, df, currency):
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
        df['datetime_utc'] = df['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert('UTC')

        # Compute the error, in %, between actual values and forecasted
        df['forecast_error_ratio'] = df.apply(compute_diff, axis=1)
        df['forecast_error_ratio'] = df['forecast_error_ratio'].round(2)

        # We have used 9999 to flag those times when we have not been able to compute error rate
        errors_found = len(df[df['forecast_error_ratio'] == 9999])
        if errors_found != 0:
            logging.info(
                'Unkown values appeared in the forecast - actual values: {} times.\n'.format(errors_found))
            logging.info(df[df['forecast_error_ratio'] == 9999].values)

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

    # Read the dataframe as scrapped from forexite. Just one file for all years
    pair_file = currency_pair + '.txt.zip'
    df_pair = pd.read_csv(input_path + pair_file, compression='zip', header=0, sep=',', \
                          dtype={'<DTYYYYMMDD>': 'str', '<TIME>': 'str'})
    df_pair = group_forexite_by_freq(df_pair)

    # Merge features
    for year in range(year_start, year_end + 1):

        logging.info('Processing year: {} for news from: {} in the pair: {}'.format(year, currency_news, currency_pair))

        # Read the dataframe as scrapped from forexfactory
        ff_file = csv_prefix_ff + str(year) + '.csv'
        df_ff = pd.read_csv(input_path + ff_file)

        # Feature extraction from forexfactory data
        df_features = fe_forexfactory(year, df_ff, currency_news)
        df_features.to_csv(output_path + csv_prefix_ff + str(year) + '_curated.csv')

        if len(df_features) != 0:

            # Feature extraction from the pair exchange value
            df_features = fe_joined_with_forexite(df_features, df_pair, snapshots)
            if len(df_features) != 0:

                # Save processed dataframe to disk
                df_features.to_csv(output_path + csv_prefix_out + '_' + str(year) + '.csv')
                all_years_df = all_years_df.append(df_features)

            else:
                logging.error('Error ocurred when extracting features from forexite')

        else:
            logging.error('Error ocurred when extracting features from forexfactory')

        logging.info('Processed year: {} for news from: {} in the pair: {}'.format(year, currency_news, currency_pair))

        all_years_df.to_csv(output_path + csv_prefix_out + '_' + str(year_start) + '_' + str(year_end) + '.csv')

        logging.info('End')
