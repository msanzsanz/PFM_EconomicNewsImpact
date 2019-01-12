MARKET REACTION TO NEWS EVENTS
---

## Table of contents
1. [Abstract](#abstract)
2. [Data sources](#data)
3. [Methodology](#methodology)
4. [Summary of results](#results)
5. [Dashboard](#dashboard)
6. [Conclusions & future Research](#conclusions)
7. [References](#references)

## Abstract <a name="abstract"> </a>
The reason we want to use the Forex Factory calendar is to know when market-moving news are expected and thereby avoid or prepare for periods of high volatility.

## Data sources <a name="data"></a>
Two publicly available data sources have been used as sources of raw data:
- **Macro-economic news events, scrapped from [forexfactory](https://www.forexfactory.com/calendar.php)**

    Forex Factory calendar is one of the most accurate calendars to keep track of Forex-related news events.
Unfortunatelly, forexfactory does not facilitate any mechanism to download this historical data from their website, so a dedicated scrapper had to be developed for such  purpose.

    Once ran the scrapper, a file is created in the desired path with the following information:
    
    | Field | Description | Format
    | ------ | ------ | -------
    | country | Country that publishes the new. | 3-letter abbreviation of the country, e.g., USD
    | datetime | Date and time of publication. | yyyy-mm-dd hh:mm:ss, e.g., 2018-10-21 00:00:00
    | impact | Forexfactory classification for the expected impact of this release in the market. | <High\|Low\|Medium>
    | new | Name of the new. | String, e.g., Unemployment Rate
    | forecast | Forecasted value agreed by the economic experts and analytics. | 5%
    | actual | Actual value officially published by the goverment. It´s worth mentioning that each new could have a different range of values. For instance, unemployment rates are published in %, e.g., 5%. However, other news are published in thousands or millions, as 'Unemployment Claims', which could have a value of 210k. Thus, this field will require some pre-procesing as explained later.<br>Lower or higher actual values do not mean better or worse. If we look at GDP numbers, for example, a higher number is generally seen as more positive for an economy and, by so, the strength of its currency. If we look at unemployment data, a lower number is positively interpreted as it shows the economy of the country is improving.| 4% or 210K or 30B or 0.1
    | forecast_error | Flag that indicates whether the actual value was better or worse for the economy than the forecasted one.   | <better\|worse\|accurate>
    | previous | Actual value of the previous release for this same new.| e.g. 3%
    | previous_error | Flag that indicates whether the previous value was corrected on this release or not. If corrected, the flag indicates whether the corrected value was better or worse for the economy than the forecasted one.  | <better\|worse\|accurate>
    | week | week of the year. | From 1 to 52

- **Historical EUR-USD exchange rate, downloaded from [dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/)**

    The exchange price for the EUR-USD pair was downloaded from dukascopy, in candlesticks of 5 minutes.

    | Field | Description | Format
    | ------ | ------ | -------
    | Gmt time | timestamp in GMT | dd.mm.yyyy hh:mm:ss.000, e.g. 01.01.2007 00:00:00.000
    | Open | enchange rate when the candle opened | e.g. 1.31908
    | High | highest exchange rate for that candle | e.g. 1.31961
    | Low | lowest exchange rate for that candle  | e.g. 1.31896
    | Close | enchange rate when the candle closed | e.g. 1.31947
    | Volume | Volume of transactions done on that period | e.g. 5268.6


## Methodology <a name="methodology"></a>
The picture below illustrates the methodology followed for this project:

Each piece of code is commented, so that readers can understand what it does / how to invoke it.

- **Phase-1: Obtain raw data from publicly available sources**

    As explained above, data from forexfactory was obtained using our own scrapper whilst data from dukascopy was downloaded from their website.   
    Both pieces of raw data were uploaded to the repo, to /data/raw/   
    Data from forexfactory was obtained running the scrapper with the following arguments:

    ```sh
    $ cd code/scrappers
    $ python forexfactory_scraper.py calendar.php?week=oct14.2006 calendar.php?week=dec23.2018 52 ../../data/raw/
    ```
- **Phase-2: Get familiar with raw data**   

    Before extracting features from the raw data we did an exploratory analysis to understand it better: **/code/data_curation/data_familiarity.ipynb**.   
Main observations would be:
    * Measure unit for some news is % (as unemployment rate), whilst others use  Millions, Billions, Thousands, etc. We would need to normalize that.
    * For USD, news impact is more or less evenly distributed between High, Medium and Low. For instance, for 2017, 22 High news were published, 27 Medium and 29 Low.
    * There are news published as having "High" impact sometimes and "Medium" impact others, like "Unemployment Claims".
    * forexfactory.com publishes non-accurate forecasts around 2/3 of the times.
    * There are news published in 'bundle', this is, at exactly the same date and time. This occurs about 30% of the times. This is pretty relevant as the market could react based on the combination of them.      

- **Phase-3: Feature engineer**

    Raw data was processed to generate features of interest for the models.   
In essence, our models aim to predict the market reaction to news events, so we needed to incorporate market features to our dataframe:

    For that:
    * Both dataframes were converted to the same timezone   
Forexfactory was scrapped in US/Eastern with DST off and dukascopy provides its data in GMT, already accounting for DST.
    * Handle Daylight Saving Times (DST)   
Forexfactory stores the data without accounting for DST (as explained in the code), so we needed to add +1h whenever applicable.
    * A significant amount of time was spent to sanity check a proper merge of both dataframes by datetime.   
This could sound like an easy task, but it was very time-consuming. We needed to ensure that dukascopy was alligned with forexfactory so that the data going into the models reflected the same market reaction as the one observed in the forexfactory website. The goal of our project is to predict short-term impact, so a deviation in minutes (as those originated by DST ) could ruin our models.   
   
        *As a curiosity, our first data source for exchange rates was forexite.com, but we had to change to dukascopy exactly because of this same reason*
    
        ```sh
        $ cd code/data_curation
        $ python dc_forexfactory.py 2007 2018  ../../data/raw/ forexfactory_ USD EURUSD [5,10,15,20,25,30] [0,30,60] [30,60,120,180,240] ON ../../data/curated/ features dc_forexfactory.log
        ```
    
    Once ran the code, a new set of features are created: 
    
    | Field | Description | Format
    | ------ | ------ | -------
    |year, quarter, month, weekday | raw datetime is decomposed into individual fields. For the models, we will use year, week and weekday. | yyyy, e.g. 2018
    |forecast_error_diff| difference between actual and forecasted values. <br> We have aligned the sign with the usual market effect, i.e., positive diffs are good for the currency. Negative otherwise. This value could be pretty small for some news (e.g. 0.2% - 0.1% = 0.1 for CPI), or pretty large for others (e.g. 312K - 179K = 133 for Non-Farm employment change). <br> Thus, we would need to transform it to the same scale before using it on the modeling phase.    | float
    |forecast_error_diff_deviation| ratio between *forecast_error_diff* and the standard deviation of *forecast_error_diff* for the previous **5 events**. <br> This field tells you if an economic data was un-expected compared to its recent history. A surprise could generate a bigger impact (more price movement) in the markets. | float
    |forecast_error_diff_outlier_class| numeric field indicating whether *forecast_error_diff* was an outlier **for the entire history** of this new or not.  | 0: within Q1 - Q3 <br> 1: Out of the Q1 - Q3 range but still not an outlier <br> 2: Outlier <br> 3: Extreme Outlier
    |previous_error_diff| difference between previous published value and corrected one (if corrected). As it happens for *forecast_error_diff*, this value could be pretty small for some news and pretty large for others, so we need to transform it to the same scale.| float
    |previous_error_diff_deviation| ratio between *previous_error_diff* and the standard deviation of previous_error_diff for the previous **5 events**. Deviations in forecasted values + corrections on the previous release could generate bigger impacts in the market. | float
    |previous_error_diff_outlier_class| numeric field indicating whether *previous_error_diff* was an outlier or not.  | 0: within Q1 <br> Q3, 1: Out of the Q1 - Q3 range but still not an outlier <br> 2: Outlier <br> 3: Extreme Outlier
    |volatility_60_0_**before**| volatility, i.e. (high - low), one hour before the publication of the new. | integer 
    |pips_agg_60_0_**before**| pips variation in the  exchange pair, calculated as (close - open), one hour before the publication of the new | integer 
    |volatility_X_Y_**after**| volatility, i.e. (high - low), of the window X-Y, when X and Y represent minutes after the release of the new. <br> E.g. volatility_0_30_after holds the volatility 30 minutes after the new is released, volatility_30_60_after holds the volatility between 30 and 60 minutes, etc. <br> In this table, we will use *X_Y* convention for the sake of simplicity, as the dataframe will contain as many windows as requested by the user when running the code.| integer 
    |pips_candle_X_Y_**after**| pips variation in the exchange pair, calculated as (close - open), of the *X_Y* window. | integer 
    |pips_agg_X_Y_**after**| Aggregated pips variation in the exchange pair, calculated as (close - open), being open the exchange value when the new was released, i.e. the *0_Y* window | integer 
    |pips_candle_max_X_Y_**after**| maximum pips variation in the exchange pair, calculated as (high - open), of *X_Y* window.| integer 
    |pips_candle_min_X_Y_**after**| minimum pips variation in the exchange pair, calculated as (low - open), of *X_Y* window. | integer 
    
- **Phase-4: Feature and model selection**   
    Before creating models we did an exploratory analysis to understand our features better: **/code/models/data_familiarity.ipynb**.   
Main goals were:   
    * Ensure there was no missing data.
    * Locate outliers and decide what to do with them.
    * Review features correlations.
    * Decide which models would make more sense for our prediction objective.
    
    Details could be found on the notebook itself. Key points would be:
    
    *  
   


## Summary of results <a name="results"></a>

## Dashboard <a name="dashboard"></a>

## Conclusions & future Research <a name="conclusions"></a>

## References <a name="references"></a>