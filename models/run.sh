nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_NO_1_1_1 market_basic excluded ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_NO_1_1_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_NO_3_2_1 market_basic excluded ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_NO_3_2_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_YES_1_1_1 market_basic excluded ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_YES_1_1_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_YES_3_2_1 market_basic excluded ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_YES_3_2_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 High_NO_1_1_1 market_basic excluded ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_High_NO_1_1_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 High_YES_1_1_1 market_basic excluded ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_High_YES_1_1_1 &

nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_NO_1_1_1 market_all included ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ all_included_ALL_NO_1_1_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_NO_3_2_1 market_all included ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ all_included_ALL_NO_3_2_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_YES_1_1_1 market_all included ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ all_included_ALL_YES_1_1_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_YES_3_2_1 market_all included ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ all_included_ALL_YES_3_2_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 High_NO_1_1_1 market_all included ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ all_included_High_NO_1_1_1 &
nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 High_YES_1_1_1 market_all included ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ all_included_High_YES_1_1_1 &



One single sweep to check for performance
==========
python models_sweeps.py 15_60 High_NO_1_1_1 market_basic excluded ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_High_NO_1_1_1


