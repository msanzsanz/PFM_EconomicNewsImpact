>> BASIC SWEEPS IN GRID OPTIONS

    >>> Individual jobs - using basic market features and excluding *_before features (AWS)
    
        nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_NO_1_1_1 basic excluded basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_NO_1_1_1 &
        nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_NO_3_2_1 basic excluded basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_NO_3_2_1 &
        nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_YES_1_1_1 basic excluded basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_YES_1_1_1 &
        nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 ALL_YES_3_2_1 basic excluded basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_YES_3_2_1 &
        nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 High_NO_1_1_1 basic excluded basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_High_NO_1_1_1 &
        nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 High_YES_1_1_1 basic excluded basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv 5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_High_YES_1_1_1 &
    
    >> Individual jobs - using all market features and including *_before features (myLaptop)

        nohup python models_sweeps.py 0_60 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_0_60 &
        nohup python models_sweeps.py 30_60 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_30_60 &
        nohup python models_sweeps.py 30_90 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_30_90 &
        nohup python models_sweeps.py 60_120 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_60_120 &
        nohup python models_sweeps.py 60_180 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_60_180 &
        nohup python models_sweeps.py 60_240 ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_60_240 &

        nohup python models_sweeps.py 0_60 ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_0_60 &
        nohup python models_sweeps.py 30_60 ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_30_60 &
        nohup python models_sweeps.py 30_90 ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_30_90 &
        nohup python models_sweeps.py 60_120 ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_60_120 &
        nohup python models_sweeps.py 60_180 ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_60_180 &
        nohup python models_sweeps.py 60_240 ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1 basic,all included basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv  5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/ sweeps_baseline_60_240 &


    >> Just one command, to do all the sweeps

        nohup python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 \
        ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1  \
        basic excluded basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv \
        5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ basic_excluded_ALL_NO_1_1_1 &

        python models_sweeps.py 15_60,30_60,30_90,30_120,60_120,60_240 \
        ALL_NO_1_1_1,ALL_NO_3_2_1,ALL_YES_1_1_1,ALL_YES_3_2_1,High_NO_1_1_1,High_YES_1_1_1  \
        all included basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv \
        5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ sweeps_all_features_basic_grips





