import logging
import sys, ast
import pandas as pd
import numpy as np
import pytz


SNAPSHOT_OFFSET_BEFORE_RELEASE = 60
CANDLE_SIZE = 5

########################################################################################################################
#
#   DESCRIPTION:
#
#       Method to activate the logging mechanism
#
#   INPUT PARAMETERS:
#
#       log_file:   path + file to store program logs
#
########################################################################################################################
def set_logger(log_file):

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


########################################################################################################################
#
#   DESCRIPTION:
#
#       Method to activate the logging mechanism
#
#   INPUT PARAMETERS:
#
#       log_file:   path + file to store program logs
#
########################################################################################################################
def compute_diff(row, forecasted_field, actual_field, error_field):
    try:

        forecasted = row[forecasted_field]
        actual = row[actual_field]
        forecast_error = row[error_field]

        values = [forecasted, actual]

        # Eliminate non-digits characters at the start of the value
        for pos, val in enumerate(values):
            i = 0
            while not val[i].isdigit() and val[i] != '-':
                i = i + 1

            values[pos] = val[i:]

        # Eliminate non-digits characters at the end of the value
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
            # we cannot divide by zero, so be handle this special case
            # We consider a 100% error rate whenever:
            #   forecast is 0 and actual is 1
            #   forecast is 0.0 and actual is 0.1
            #   forecast is 0.01 and actual is 0.01
            #
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
                    # Data has been explored to ensure there are no values < 0.01
                    # Nevertheless, we log it as an ERROR if it happens
                    logging.error('Diff between actual-forecasted cannot be computed ! ')
                    diff = 9999
            else:
                diff_per = abs(actual_float - forecasted_float) * 100 / abs(forecasted_float)
                sign = 1 if actual_float < forecasted_float else -1
                diff = diff_per * sign

        # A positive diff does not mean a better value for the economy (e.g, more people unemployed is worst.
        # For making the life easier to the ML algorithms, let´s modify the variable to always consider
        # a positive diff as good to the market
        if (diff < 0 and forecast_error == 'better') or \
           (diff > 0 and forecast_error == 'worse'):
               diff = diff * -1

    except:
        diff = 9999

    return diff



########################################################################################################################
#
#   DESCRIPTION:
#
#       forexfactory.com has a interesting way to handle Daylight saving time (DST) ...
#       Basically, it seems that store the data in their servers without accounting for DST. Therefore, they adjust
#       the time on the website ONLY during visualization only when the DST flag is enabled.
#
#       When the DST flag is enabled?
#
#           1. Manually by users activating it on settings menu.
#           2. By default whenever users connect to forexfactory.com from a country that is currently on DST
#
#       Thus, if the scrapper is ran with DST = off, we need to compute that +1h before merging this data with the
#       market data
#
#   INPUT PARAMETERS:
#
#       log_file:   path + file to store program logs
#
########################################################################################################################

def add_dst_flag(df):

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


########################################################################################################################
#
#   DESCRIPTION:
#
#       Function to classify the new release based on the market impact
#
#           -2: USD decreases w.r.t EUR in > 20 pips
#           -1: USD decreases w.r.t EUR in 10-20 pips
#           0: almost no impact
#           1: USD increases w.r.t EUR in 10-20 pips
#           1: USD increases w.r.t EUR in > 20 pips
#
#   INPUT PARAMETERS:
#
#       row:                dataframe row
#       field_previous      field of the row containing previous value
#       field_current       field of the row containing current value
#
#       Note: Columns names are passed as i/p as we are going to compute the market reaction at different snapshots
#
########################################################################################################################

def compute_direction(row, field_previous, field_current):

    value_previous = row[field_previous]
    value_current = row[field_current]

    out = 0
    sign = np.where(value_current >= value_previous, 1, -1)

    if abs(value_previous - value_current) < 10:
        out = 0
    elif abs(value_previous - value_current) > 20:
        out = 2
    else:
        out = 1

    return  out * sign


########################################################################################################################
#
#   DESCRIPTION:
#
#       Function to merge both sources of data, for every snapshot of interest
#
#
#   INPUT PARAMETERS:
#
#       df_features         dataframe for forexfactory data
#       df_pair             dataframe for dukascopy data
#       snapshots           array of snapshots to loop
#       freq                window-size
#
#
########################################################################################################################
def fe_joined_with_dukascopy(df_features, df_pair, snapshots, freq):

    try:
        # Expand the forexfactory dataframe with as many snapshots as requested
        # After the publication of the new
        for snapshot in snapshots:
            offset = snapshot - freq
            column_name = '_' + str(offset) + '_' + str(snapshot) + '_after'
            df_features[column_name] = df_features['datetime_gmt'] + pd.DateOffset(minutes=offset)

            # Some news are not published at o´clocks (i.e. neither 2:00 nor 2:30, but 1:59)
            # We rounded them to the closest window
            round_freq = str(freq) + 'min'
            df_features[column_name] = df_features[column_name].dt.round(round_freq)

            df_features = df_features.set_index(column_name).join(df_pair)
            df_features = df_features.reset_index(drop=True)

            df_features['volatility'] = abs(df_features['high'] - df_features['low'])
            df_features['direction_candle'] = df_features.apply(lambda row: compute_direction(row, 'open', 'close'), axis=1)
            df_features['direction_agg'] = df_features.apply(lambda row: compute_direction(row, 'close_released', 'close'), axis=1)

            df_features['pips_agg'] = df_features['close'] - df_features['close_released']
            df_features['pips_candle'] = df_features['close'] - df_features['open']

            # Drop undesired columns
            df_features = df_features.drop(['open'], axis=1)

            df_features.rename({'close': 'close' + column_name,
                                'low': 'low' + column_name,
                                'high': 'high' + column_name, \
                                'volatility': 'volatility' + column_name, \
                                'direction_candle': 'direction_candle' + column_name, \
                                'direction_agg': 'direction_agg' + column_name, \
                                'pips_agg': 'pips_agg' + column_name,
                                'pips_candle': 'pips_candle' + column_name}, \
                               inplace=True, axis='columns')

        return df_features

    except BaseException as e:
        logging.error('Error while extracting features from the currency pair')
        logging.error('exception: {}'.format(e))
        return pd.DataFrame()

def fe_joined_with_dukascopy_old(df_features, df_pair, snapshots, freq):

    try:
        # Expand the forexfactory dataframe with as many snapshots as requested
        # After the publication of the new
        for snapshot in snapshots:
            offset = snapshot - freq
            column_name = '_' + str(offset) + '_' + str(snapshot) + '_after'
            df_features[column_name] = df_features['datetime_gmt'] + pd.DateOffset(minutes=offset)

            # Some news are not published at o´clocks (i.e. neither 2:00 nor 2:30, but 1:59)
            # We rounded them to the closest window
            round_freq = str(freq) + 'min'
            df_features[column_name] = df_features[column_name].dt.round(round_freq)

            df_features = df_features.set_index(column_name).join(df_pair)
            df_features = df_features.reset_index(drop=True)

            df_features['volatility'] = abs(df_features['high'] - df_features['low'])
            df_features['direction_candle'] = df_features.apply(lambda row: compute_direction(row, 'open', 'close'), axis=1)
            df_features['direction_agg'] = df_features.apply(lambda row: compute_direction(row, 'close_released', 'close'), axis=1)

            df_features['pips_agg'] = df_features['close'] - df_features['close_released']
            df_features['pips_candle'] = df_features['close'] - df_features['open']

            # Drop undesired columns
            df_features = df_features.drop(['open'], axis=1)

            df_features.rename({'close': 'close' + column_name,
                                'low': 'low' + column_name,
                                'high': 'high' + column_name, \
                                'volatility': 'volatility' + column_name, \
                                'direction_candle': 'direction_candle' + column_name, \
                                'direction_agg': 'direction_agg' + column_name, \
                                'pips_agg': 'pips_agg' + column_name,
                                'pips_candle': 'pips_candle' + column_name}, \
                               inplace=True, axis='columns')

        # Before the publication of the new
        for snapshot in snapshots:
            offset = snapshot - freq
            column_name = '_' + str(snapshot) + '_' + str(offset) +'_before'
            df_features[column_name] = df_features['datetime_gmt'] - pd.DateOffset(minutes=snapshot)

            # Some news are not published at o´clocks (i.e. neither 2:00 nor 2:30, but 1:59)
            # We rounded them to the closest 15 min candle.
            round_freq = str(freq) + 'min'
            df_features[column_name] = df_features[column_name].dt.round(round_freq)

            df_features = df_features.set_index(column_name).join(df_pair)
            df_features = df_features.reset_index(drop=True)

            df_features['volatility'] = abs(df_features['high'] - df_features['low'])
            df_features['direction_candle'] = df_features.apply(lambda row: compute_direction(row, 'open', 'close'), axis=1)
            df_features['direction_agg'] = df_features.apply(lambda row: compute_direction(row, 'close', 'close_released'), axis=1)

            df_features['pips_candle'] = df_features['close'] - df_features['open']
            df_features['pips_agg'] =  df_features['close_released'] - df_features['close']

            # Drop undesired columns
            df_features = df_features.drop(['open'], axis=1)

            df_features.rename({'close': 'close' + column_name,
                                'low': 'low' + column_name,
                                'high': 'high' + column_name, \
                                'volatility': 'volatility' + column_name, \
                                'direction_candle': 'direction_candle' + column_name, \
                                'direction_agg': 'direction_agg' + column_name, \
                                'pips_agg': 'pips_agg' + column_name,
                                'pips_candle': 'pips_candle' + column_name}, \
                               inplace=True, axis='columns')


        return df_features

    except BaseException as e:
        logging.error('Error while extracting features from the currency pair')
        logging.error('exception: {}'.format(e))
        return pd.DataFrame()

########################################################################################################################
#
#   DESCRIPTION:
#
#       Whenever a new is published, governments can also correct previous publications.
#       We need to take this into account, as we could have 2 deviations:
#
#           1. Deviation from the forecasted value
#           2. Plus positive/negative correction of the previous data
#
#
#   INPUT PARAMETERS:
#
#       df         dataframe for forexfactory data
#
#
########################################################################################################################
def compute_previous_error_ratio(df):

    # Unique list of news
    news_list = df['new'].unique()
    df_out = pd.DataFrame([])

    for new in news_list:
        # sort by ascending datetime
        df_temp = df[df['new'] == new].sort_values(by='datetime_gmt', ascending=True).reset_index()
        df_temp.drop(columns=['index'], axis=1, inplace=True)
        df_temp['previous_value'] = df_temp['actual'].shift().fillna(df_temp['previous'])
        df_temp['previous_error_ratio'] = df_temp.apply(lambda row: compute_diff(row, 'previous_value',
                                                                                 'previous', 'previous_error'), axis=1)
        df_temp['previous_error_ratio'] = df_temp['previous_error_ratio'].round(2)

        # We have used 9999 to flag those times when we have not been able to compute error rate
        errors_found = len(df_temp[df_temp['previous_error_ratio'] == 9999])
        if errors_found != 0:
            logging.error(
                'Unkown values appeared when computing previous error ratio values: {} times.\n'.format(errors_found))
            logging.error(df[df['previous_error_ratio'] == 9999].values)

        df_temp['total_error_ratio'] = df_temp['forecast_error_ratio'] + df_temp ['previous_error_ratio']

        df_out = df_out.append(df_temp)

    return df_out


########################################################################################################################
#
#   DESCRIPTION:
#
#       Function to normalize deviation errors based on the previous 5 releases
#
#   INPUT PARAMETERS:
#
#       df         dataframe for forexfactory data
#
#
########################################################################################################################
def compute_deviation(df, field_name, size=5):
    # Unique list of news
    news_list = df['new'].unique()
    df_out = pd.DataFrame([])

    for new in news_list:
        # sort by ascending datetime
        df_temp = df[df['new'] == new].sort_values(by='datetime_gmt', ascending=True).reset_index()
        df_temp.drop(columns=['index'], axis=1, inplace=True)
        field_mean = field_name + '_mean'
        field_std = field_name + '_std'
        field_zscore = field_name + '_zscore'

        # compute mean and std of the last events
        df_temp[field_mean] = df_temp[field_name].rolling(window=size, min_periods=1).mean()
        df_temp[field_std] = df_temp[field_name].rolling(window=size, min_periods=1).std().fillna(1)
        df_temp[field_zscore] = ((df_temp[field_name] - df_temp[field_mean]) /
                                         df_temp[field_std]).fillna(0)

        df_temp[field_zscore] = df_temp[field_zscore].apply(lambda x: format(x,'.2f'))


        df_out = df_out.append(df_temp)

    return df_out


########################################################################################################################
#
#   DESCRIPTION: 
#
#       Front-end to clean the raw data scrapped from forexfactory.com
#
#   INPUT PARAMETERS:
#
#       year:           year of the scrapped data
#       ff_file:        path of the file containing the scrapped data
#       currency:       three-letter abbreviation for the news of interest. E.g. USD for United States dollar news
#       dst_correction: "ON" if the scrapper was run when DST was off
#                       "OFF" otherwise
#       freq:           [optional] window-size of how the market data was downloaded from dukascopy
#
########################################################################################################################

def fe_forexfactory(year, ff_file, currency, dst_correction, freq='5min'):
    
    df = pd.read_csv(input_path + ff_file)
    weeks = len(df.week.unique())

    # Check that the raw data contains 52 weeks.
    # Two exceptions:
    #   - By the time this code was written, 2018 is still not finished
    #   - ForexFactory does not publish the first week of 2007
    #
    if weeks == 52 or year == 2018 or year == 2007:

        df['datetime'] = pd.to_datetime(df['datetime'])

        # Filter by macro-economic newss
        df = df[df['forecast'].notnull()]

        # Filter currency of interest
        df = df[df['country'] == currency]

        # replace nan values for categorical fields
        df['forecast_error'] = df['forecast_error'].replace(np.nan, 'accurate', regex=True)
        df['previous_error'] = df['previous_error'].replace(np.nan, 'accurate', regex=True)

        # When DST is off, we need to add +1h to forexfactory.com values during winter tz
        # We do that in 2 steps. First, compute dst flag. Second, add +1h whenever the flag is set to 1.
        if dst_correction == 'ON':
            df['dts_flag'] = add_dst_flag(df)
            df['datetime'] = df.apply(apply_dts_flag, axis=1)

        df['datetime_gmt'] = df['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert('GMT')

        # Some news are not published at o´clocks (i.e. neither 2:00 nor 2:30, but 1:59)
        # We rounded them to the closest 5 min candle.
        df['datetime_gmt'] = df['datetime_gmt'].dt.round(freq)

        # Log raws with missing data
        if len(df[df.isnull().any(1)]) != 0:
            logging.error('These {} news extracted from forexfactory have nan in some columns'.format(
                len(df[df.isnull().any(1)])))
            logging.error(df[df.isnull().any(1)].values)
            
            df = df.dropna()
            logging.error('Removed these rows from the dataframe')

        # Compute the error, in %, between actual values and the forecasted ones
        df['forecast_error_ratio'] = df.apply(lambda row: compute_diff(row, 'forecast', 'actual', 'forecast_error'), axis=1)
        df['forecast_error_ratio'] = df['forecast_error_ratio'].round(2)

        # We have used the encoding 9999 to flag whenever we have not been able to compute error rate
        errors_found = len(df[df['forecast_error_ratio'] == 9999])
        if errors_found != 0:
            logging.error(
                'Unkown values appeared in the forecast - actual values: {} times.\n'.format(errors_found))
            logging.error(df[df['forecast_error_ratio'] == 9999].values)

        # Add categorical values related with datetime
        df['year'] = df['datetime'].dt.year
        df['quarter'] = df['datetime'].dt.quarter
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday

        # Drop undesired columns
        df.drop(columns=['Unnamed: 0', 'dts_flag'], axis=1, inplace=True)

        return df

    else:
        logging.error('Error {}: this dataset does not have the expected 52 weeks\n'.format(year))
        return pd.DataFrame()


def get_market_information_after(df, snapshot_at):

    df_local = df.copy()

    sufix = '_0_' + str(snapshot_at) + '_after'
    window_size = snapshot_at // CANDLE_SIZE

    column_high = 'high' + sufix
    column_low = 'low' + sufix
    column_open = 'open' + sufix
    column_volatility = 'volatility' + sufix
    column_pips = 'pips_agg' + sufix
    column_direction = 'direction_agg' + sufix

    # rolling function counts for the current row. Shift jumps as many rows as indicated
    df_local[column_high] = df_local['high'].rolling(window=window_size, min_periods=1).max()
    df_local[column_low] = df_local['low'].rolling(window=window_size, min_periods=1).min()

    df_local[column_volatility] = df_local[column_high] - df_local[column_low]
    df_local[column_volatility] = df_local[column_volatility].astype(int)

    df_local[column_open] = df_local['open'].shift(window_size - 1).fillna(df_local['open'])
    df_local[column_pips] = df_local[column_open] - df_local['open']
    df_local[column_pips] = df_local[column_pips].astype(int)

    #df_local[column_direction] = df_local.apply(lambda row: compute_direction(row, 'open', column_open), axis=1)

    # Drop undesired columns
    df_local = df_local.drop([column_high, column_low, column_open, 'open', 'high', 'low', 'close'], axis=1)

    return df_local

def get_market_information_before (df_pair, snapshot_at):

    df = df_pair.copy()
    sufix = '_' + str(snapshot_at) + '_0_before'
    window_size = snapshot_at // CANDLE_SIZE

    column_high = 'high' + sufix
    column_low = 'low' + sufix
    column_open = 'open' + sufix
    column_volatility = 'volatility' + sufix
    column_pips = 'pips_agg' + sufix
    column_direction = 'direction_agg' + sufix

    # rolling function counts for the current row. Shift jumps as many rows as indicated
    df[column_high] = df['high'].rolling(window=window_size, min_periods=1).max()
    df[column_low] = df['low'].rolling(window=window_size, min_periods=1).min()
    df[column_volatility] = df[column_high] - df[column_low]
    df[column_volatility] = df[column_volatility].astype(int)

    df[column_open] = df['open'].shift(window_size-1).fillna(df['open'])
    df[column_pips] = df['open'] - df[column_open]
    df[column_pips] = df[column_pips].astype(int)

    #df[column_direction] = df.apply(lambda row: compute_direction(row, column_open, 'open'), axis=1)

    # Drop undesired columns
    df = df.drop([column_high, column_low, column_open, 'open', 'high', 'low', 'close'], axis=1)

    return df

########################################################################################################################
#
#   DESCRIPTION: 
#
#       Front-end to clean the raw data downloaded from dukascopy.com
#
#   INPUT PARAMETERS:
#
#       filename:   path + filename of the raw data file
#
########################################################################################################################

def fe_dukascopy(filename):

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


def add_features_from_snapshots(df_features, df_pair):

    # Create a copy of the dataframe reversed
    df_pair_reverse = df_pair[::-1].copy()

    # We  extract market information some time before the news are released.
    # Why?, To validate our intuition that high volatility before the release of news could be a useful
    # feature to predict market impact
    df_pair_before_release = get_market_information_before(df_pair, SNAPSHOT_OFFSET_BEFORE_RELEASE)
    df_features = df_features.set_index('datetime_gmt').join(df_pair_before_release)
    df_features = df_features.reset_index(drop=False)

    for snapshot in snapshots_after:
        df_pair_snapshot = get_market_information_after(df_pair_reverse, snapshot)
        df_features = df_features.set_index('datetime_gmt').join(df_pair_snapshot)
        df_features = df_features.reset_index(drop=False)

    return df_features

########################################################################################################################
#
#   SCOPE:
#
#       This script processes the data scrapped from forexfactory.com and downloaded from dukascopy.com
#
#   INPUT PARAMETERS:
#
#       year_start:     first year to process
#       year_end:       last year to process
#       input_path:     path containing the raw data scrapped from Forexfactory and downloaded from Dukascopy
#       csv_prefix_ff:  prefix for Forexfactory raw data files
#                       Naming convention for the scrapper: <csv_prefix_ff>_<year>.csv
#
#       currency_news:  three-letter abbreviation for the news of interest. E.g. USD for United States dollar news
#       currency_pair:  six-letter abbreviation for the pair of interest. E.f. EURUSD for euro-american dollar
#       candles_5m:   array of required snapshots for 5min candles
#       snapshots_15m:  array of required snapshots for 15min candles
#       snapshots_30m:  array of required snapshots for 30min candles
#       dst_correction: "ON" if the scrapper was run when DST was off
#                       "OFF" otherwise
#       output_path:    path to store the dataframe with the processed data
#       csv_prefix_out: prefix to use for naming the processed data
#       log_file:       path to the log file, including filename
#
#
#   INVOCATION EXAMPLE:
#
#       python 2007 2018 ./../data/raw/ forexfactory USD EURUSD [5,10,15,20,25,30] [45,60] [60,90,120,180,210,240] \
#               ../../data/curated/ features dc_forexfactory.log
#
#
########################################################################################################################

if __name__ == '__main__':


    # Read i/p
    year_start = int(sys.argv[1])
    year_end = int(sys.argv[2])
    input_path = str(sys.argv[3])
    csv_prefix_ff = sys.argv[4]
    currency_news = sys.argv[5]
    currency_pair = sys.argv[6]
    candles_5m = ast.literal_eval(sys.argv[7])
    snapshots_after = ast.literal_eval(sys.argv[8])
    dst_correction = str(sys.argv[9])
    output_path = sys.argv[10]
    csv_prefix_out = sys.argv[11]
    log_file = sys.argv[12]

    # Create log file
    set_logger(log_file)

    # Create an empty dataframe for the processed data
    all_years_df = pd.DataFrame([])

    # Read the dataframe from dukascopy. Just one file for all years
    logging.info('Reading dukascopy file...')
    df_pair = fe_dukascopy(input_path + currency_pair + '.zip')

    # Group the historical data from dukascopy into 15-min windows
    # df_pair_15Min = df_pair.groupby(pd.Grouper(freq='15Min', closed='right', label='left')).agg(
    #                                                     {'open': 'first',
    #                                                      'high': 'max',
    #                                                      'low': 'min',
    #                                                      'close': 'last'})
    # 
    # # GroupBy could have created some nan rows if data contains windows GAPS, so we eliminate them
    # df_pair_15Min = df_pair_15Min.dropna()


    # Merge features
    for year in range(year_start, year_end + 1):

        logging.info('Processing year: {} for news from: {} in the pair: {}'.format(year, currency_news, currency_pair))

        # Read the dataframe as scrapped from forexfactory
        ff_file = csv_prefix_ff + str(year) + '.csv'

        # Feature extraction from forexfactory data
        df_features = fe_forexfactory(year, ff_file, currency_news, dst_correction)

        # Add market information for the requested snapshots
        df_features = add_features_from_snapshots(df_features, df_pair)

        if len(df_features) != 0:

            # Expand forexfactory´s dataframe with the information of exchange rate for when each new is released
            # Note that each candle from dukascopy is right handled closed so, for getting the status of the market
            # at 09:00 am, we need to merge the dataframe with the candle at 08:55
            last_column_name = '_released'
            df_features[last_column_name] = df_features['datetime_gmt'] - pd.DateOffset(minutes=5)
            df_features = df_features.set_index(last_column_name).join(df_pair)
            df_features = df_features.reset_index(drop=True)
            df_features.rename({'open': 'open' + last_column_name, \
                                'high': 'high' + last_column_name, \
                                'low': 'low' + last_column_name, \
                                'close': 'close' + last_column_name}, axis='columns', inplace='True')

            # We log any occurence when we don´t have market data 
            if len(df_features[df_features.isnull().any(1)]) > 0:
                logging.error('Rows with nan fields when getting market data when the news were released')
                logging.error(df_features[df_features.isnull().any(1)].values)

            df_features = fe_joined_with_dukascopy(df_features, df_pair, candles_5m, 5)
            
            if len(df_features) != 0:

                # Append procesed data 
                all_years_df = all_years_df.append(df_features)

            else:
                logging.error('Error ocurred when extracting features from dukascopy')

        else:
            logging.error('Error ocurred when extracting features from forexfactory')

        logging.info('Processed year: {} for news from: {} in the pair: {}'.format(year, currency_news, currency_pair))

    logging.info('Compute deviation with actual values from the previous publication')
    all_years_df = compute_previous_error_ratio(all_years_df)

    logging.info('Adding z-score for deviation between forecast and actual')
    all_years_df = compute_deviation(all_years_df, 'forecast_error_ratio')

    logging.info('Adding z-score for deviation between forecast and actual, including previous values')
    all_years_df = compute_deviation(all_years_df, 'total_error_ratio')

    all_years_df.to_csv(output_path + csv_prefix_out + \
                        '_news_' + currency_news + '_pair_' + \
                        currency_pair + '_' + \
                        str(year_start) + '_' + str(year_end) + '.csv',
                        index=False)

    logging.info('End')
