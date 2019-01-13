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
Unfortunatelly, Forex Factory does not facilitate any mechanism to download this historical data from their website, so a dedicated scrapper had to be developed for such  purpose.

    Once ran the scrapper, a file is created in the desired path with the following information:
    
    | Field | Description | Format
    | ------ | ------ | -------
    | country | Country that publishes the new. | 3-letter abbreviation of the country, e.g., USD
    | datetime | Date and time of publication. | yyyy-mm-dd hh:mm:ss, e.g., 2018-10-21 00:00:00
    | impact | Forex Factory classification for the expected impact of this release in the market. | <High\|Low\|Medium>
    | new | Name of the new. | String, e.g., Unemployment Rate
    | forecast | Forecasted value agreed by the economic experts and analytics. | 5%
    | actual | Actual value officially published by the goverment. It´s worth mentioning that each new could have a different range of values. For instance, unemployment rates are published in %, e.g., 5%. However, other news are published in thousands or millions, as 'Unemployment Claims', which can have a value of 210k. Thus, to compute deviation from the forecasted value we will require some post-processing during the feature engineer phase, as explained later.| 4% or 210K or 30B or 0.1
    | forecast_error | Flag that indicates whether the actual value was better or worse for the economy than the forecasted one.   | <better\|worse\|accurate>
    | previous | Governments can amend the released values if needed. If that happens, Forex Factory make that correction public in their economic calendar by mean of this field.| e.g. 3%
    | previous_error | Flag that indicates whether the previous value was corrected on this release or not. If corrected, the flag indicates whether the corrected value was better or worse for the economy w.r.t the previous one.  | <better\|worse\|accurate>
    | week | week of the year. | From 1 to 52

- **Historical EUR-USD exchange rate, downloaded from [dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/)**

    The exchange price for the EUR-USD pair was downloaded from dukascopy, in candlesticks of 5 minutes. Each candle includes this information:

    | Field | Description | Format
    | ------ | ------ | -------
    | Gmt time | Timestamp in GMT | dd.mm.yyyy hh:mm:ss.000, e.g. 01.01.2007 00:00:00.000
    | Open | Exchange rate when the candle opened | e.g. 1.31908
    | High | Highest exchange rate for that candle | e.g. 1.31961
    | Low | Lowest exchange rate for that candle  | e.g. 1.31896
    | Close | Exchange rate when the candle closed | e.g. 1.31947
    | Volume | Volume of transactions done on during this candle  | e.g. 5268.6


## Methodology <a name="methodology"></a>
The picture below illustrates the methodology followed for this project:

Each piece of code is commented, so that readers can understand what it does / how to invoke it.

- **Phase-1: Obtain raw data from publicly available sources**

    As explained above, data from Forex Factory was obtained using our own scrapper whilst data from dukascopy was downloaded from their website.   
    Both pieces of raw data were uploaded to the repo, available in /data/raw/   
    Data from Forex Factory was obtained running the scrapper with the following arguments:

    ```sh
    $ cd code/scrappers
    $ python forexfactory_scraper.py calendar.php?week=oct14.2006 calendar.php?week=dec23.2018 52 ../../data/raw/
    ```
- **Phase-2: Raw data exploration**   

    Before extracting features from the raw data we did an exploratory analysis to understand it better: **/code/data_curation/data_exploration.ipynb**.   
Main observations would be:
    * Measure unit for some news is % (as unemployment rate), whilst others use  Millions, Billions, Thousands, etc. We would need to normalize that.
    * For USD, the forecasted impact published by Forex Factory is more or less evenly distributed between High, Medium and Low. For instance, for 2017, 22 High news were published, 27 Medium and 29 Low.
    * There are news published as having "High" impact sometimes and "Medium" impact others, like "Unemployment Claims".
    * Forex Factory publishes non-accurate forecasts around 2/3 of the times.
    * 30% of the times news published in 'bundle', this is, at the same date and time. This is pretty relevant as the market could react based on the combination of them.      

- **Phase-3: Feature engineer**

    Raw data was processed to generate features of interest for the models.   
In essence, our models aim to predict the market reaction to news events, so we needed to incorporate market features to our dataframe.

    For that:
    * Both dataframes were converted to the same timezone   
Forex Factory was scrapped in US/Eastern (DST off) and dukascopy provides its data in GMT (DST on).
    * Handle Daylight Saving Times (DST)   
Forex Factory stores the data without accounting for DST (as explained in the code), so we needed to add +1h whenever applicable.
    * A significant amount of time was spent to sanity check a proper merge of both dataframes by datetime.   
This could sound like an easy task, but it was very time-consuming. We needed to ensure that dukascopy was alligned with Forex Factory so that the data going into the models reflected the same market reaction as the one observed in the Forex Factory website. The goal of our project is to predict short-term impact based on news events, so a deviation in minutes (as those originated by DST ) could ruin our models.   
   
        *As a curiosity, our first data source for exchange rates was forexite.com, but we had to change to dukascopy exactly because of this same reason. Not sure how dukascopy handles DST but we were unable to validate the merge.*
        
        To extract features from the raw data, run:
        
        ```sh
        $ cd code/data_curation
        $ python dc_forexfactory.py 2007 2018  ../../data/raw/ forexfactory_ USD EURUSD [5,10,15,20,25,30] [0,30,60] [30,60,120,180,240] ON ../../data/curated/ features dc_forexfactory.log
        ```
    
    Once ran the code, a new set of features is created: 
    
    | Field | Description | Format
    | ------ | ------ | -------
    |year, quarter, month, weekday | Raw datetime is decomposed into individual fields. For the models, we will use year, week and weekday. | yyyy, e.g. 2018
    |forecast_error_diff| Difference between actual and forecasted values. <br>Lower or higher *actual* values do not mean better or worse. <br> If we look at GDP numbers, for example, a higher number is generally seen as more positive for an economy and, by so, the strength of its currency. If we look at unemployment data, a lower number is positively interpreted as it shows the economy of the country is improving. <br> Thus, we ensure that the difference is always negative whenever *forecast_error* is equal to *"worse"* and positive whenever is equal to *"better"*. <br> That will make the life easier to models and for visualization dashboards. | float
    |forecast_error_diff_deviation| *forecast_error_diff* can be pretty small for some news (e.g. 0.2% - 0.1% = 0.1 for CPI) or pretty large for others (e.g. 312K - 179K = 133 for Non-Farm employment change). <br> Therefore, we need to transform it to a common scale across news. <br> *forecast_error_diff_deviation* is the ratio between *forecast_error_diff* and the standard deviation of *forecast_error_diff* for the previous **5 events**. <br> This field tells you if an economic data was un-expected compared to its recent history. Bigger deviations could generate bigger surprises in the markets followed by, hopefully, more price movement. | float
    |forecast_error_diff_outlier_class| numeric field indicating whether *forecast_error_diff* was an outlier **for the entire history** of this new or not.  | 0: within Q1 - Q3 <br> 1: Out of the Q1 - Q3 range but still not an outlier <br> 2: Outlier <br> 3: Extreme Outlier
    |previous_error_diff| difference between the previous published value and the corrected one. <br> If the government does not correct the value, this field will be equal to 0. <br> As it happens for *forecast_error_diff*, this value could be pretty small for some news and pretty large for others, so we need to transform it to the same scale.| float
    |previous_error_diff_deviation| ratio between *previous_error_diff* and the standard deviation of previous_error_diff for the previous **5 events**. Deviations in the forecasted value + corrections on the previous release could generate bigger impacts in the market. | float
    |previous_error_diff_outlier_class| numeric field indicating whether *previous_error_diff* was an outlier or not.  | 0: within Q1 <br> Q3, 1: Out of the Q1 - Q3 range but still not an outlier <br> 2: Outlier <br> 3: Extreme Outlier
    |volatility_60_0_**before**| volatility, i.e. (high - low), one hour before the publication of the new. | integer 
    |pips_agg_60_0_**before**| pips variation in the  exchange pair, calculated as (close - open), one hour before the publication of the new | integer 
    |volatility_X_Y_**after**| volatility, i.e. (high - low), of the window X-Y, when X and Y represent minutes after the release of the new. <br> E.g. volatility_0_30_after holds the volatility 30 minutes after the new is released, volatility_30_60_after holds the volatility between 30 and 60 minutes, and so on. <br> In this table, we will use the *X_Y* convention for the sake of simplicity, as the dataframe will contain as many windows as requested by the user when running the code.| integer 
    |pips_candle_X_Y_**after**| pips variation in the exchange pair, calculated as (close - open), of the *X_Y* window. | integer 
    |pips_agg_X_Y_**after**| Aggregated pips variation in the exchange pair, calculated as (close - open), being open the exchange value when the new was released, i.e. the *0_Y* window | integer 
    |pips_candle_max_X_Y_**after**| maximum pips variation in the exchange pair, calculated as (high - open), of *X_Y* window.| integer 
    |pips_candle_min_X_Y_**after**| minimum pips variation in the exchange pair, calculated as (low - open), of *X_Y* window. | integer 
    
- **Phase-4: Feature selection**   
    Before creating any model we did an exploratory analysis to understand our features better: **/code/models/data_exploration.ipynb**.   
Main goals were:   
    * Ensure there were no NaN or nulls.
    * Locate outliers and decide what to do with them.
    * Review features correlations.
    * Decide which models would make more sense for our prediction objective.
    
    Low-level details could be found in the notebook itself. Key relevant points would be:
    
    * No featured extracted Forex Factory seems to have high correlation with the reaction of the market, which is surprising and disappointing.   
     My expectation was to see a linear-ish relationship between *forecast_error_diff_deviation* and the corresponding market impact. I.e., surprises on news events should generally generate a
bigger impact (more price movement) in the markets. However, it does not seems to work like that. Sometimes big deviations do not move the market at all and the other way around... :-(
    * The highest correlations are seen on features that hold market reaction with those same features from the short-term past. I.e. *pips_candle_max_30_240_after* is highly correlated with *pips_candle_max_30_180_after* , as well as *pips_candle_max_30_120*. <br> It´s still correlated with *pips_candle_max_30_60*, but with less intensity. 
   
- **Phase-5: Models**

    Due to the lack of linear relationships between variables, linear regression models are likely to work poorly for our purpose.   
    Other techniques like KNeighbors, decision trees, random forest, boosting classifiers, etc. are expected to behave better, capturing non-linearity in the data by dividing the space into smaller sub-spaces.
    
    But, how to start? We needed to decide a bunch of things:
    
    1. **How to group Forex Factory data whenever there are several news published at the same date and time.**   
    Basically, we could apply several approaches:   
        * Option A: sum up all deviations to compute one unique value by each datetime, giving the same weight(=1) to each new, regardless its type.
        * Option B: assuming that news classified by Forex Factory as having a 'High' impact will, in fact, move the market more heavily than those classified as 'Low' impact, it seems sensible to apply different weights when adding up deviations (3 for 'High', 2 for 'Medium', 1 for 'Low').
        * Option C: consider in our models just news published in isolation, regardless their impact classification. All news will have the same weight.
        * Option D: consider in our models just news published in isolation, regardless their impact classification, but weighted by the Forex Factory impact classification.
        * Option E: consider in our models just news published in isolation, and just those classified as having a 'High' impact.
  
    2. **Once published the new, how much time do we wait for running our model?**   
    As explained before, during phase-4 was observed that the highest correlations was seen on features holding market reaction, with those same features from the short-term past.   
    Thus, perhaps we will have higher accuracy predicting  market reaction after 2 hours if we feed the model with  market reaction after 30 min.   
    But, how much time to wait? 15 min? 30 min? difficult to say at this time.   

    3. **Do we want to feed the model with market status before the release of the new?**   
    It seems sensible to think that low volatility before news events could mean that the market is waiting for the release announcement in order to take a path (either going UP or DOWN). Thus, it feels like an interesting feature to consider in our models.
    
    We can generate a lot of models, one per each combination of these variables.    
    Creating individual Jupyter notebooks was not a practical approach to tests all these combinations, so we decided to create a mini-tool to do sweeps based on the user choice.
    
    ```sh
    $ cd code/models
    $ nohup python models_sweeps.py 0_30 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/ sweeps_baseline_0_30 &
    $ nohup python models_sweeps.py 0_60 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_0_60 &
    $ nohup python models_sweeps.py 0_120 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_0_120 &
    $ nohup python models_sweeps.py 0_180 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/ sweeps_baseline_0_180 &
    $ nohup python models_sweeps.py 0_240 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_0_240 &
    $ nohup python models_sweeps.py 30_60 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_30_60 &
    $ nohup python models_sweeps.py 30_120 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_30_120 &
    $ nohup python models_sweeps.py 30_180 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_30_180 &
    $ nohup python models_sweeps.py 30_240 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/ sweeps_baseline_30_240 &
    $ nohup python models_sweeps.py 60_120 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_60_120 &       
    $ nohup python models_sweeps.py 60_180 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_60_180 &
    $ nohup python models_sweeps.py 60_240 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 0,30,60 30,60,120,180,240 ../../data/models_results/  sweeps_baseline_60_240 &

    ```
      
          
## Summary of results <a name="results"></a>

## Dashboard <a name="dashboard"></a>

## Conclusions & future Research <a name="conclusions"></a>

## References <a name="references"></a>


Visualización para ver si hay alguna noticia para la que exista correlación en deviation --> market impact
