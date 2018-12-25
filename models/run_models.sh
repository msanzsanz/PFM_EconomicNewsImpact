#!/bin/bash

for time in 15_60,30_60,30_90,30_120,60_120,60_240; do

    for type in ALL_NO_1_1_1 ALL_NO_3_2_1 ALL_YES_3_2_1 ALL_YES_1_1_1 High_NO_1_1_1 High_YES_1_1_1; do

        python models_sweeps.py $time $type \
        all included basic ../data/curated/features_rounded_news_USD_pair_EURUSD_2007_2018.csv \
        5,10,15,20,25,30 45,60 60,90,120,150,180,210,240 ../data/curated/ all-included-${type}

    done
done
