#!/bin/bash

for time in 0_60,15_60,30_60,30_90,30_120,30_120,30_240; do

    for type in ALL_NO_1_1_1 ALL_NO_3_2_1 ALL_YES_3_2_1 ALL_YES_1_1_1 High_NO_1_1_1 High_YES_1_1_1; do

        nohup python models_sweeps.py $time $type \
        all,basic included,excluded basic ../data/curated/features_news_USD_pair_EURUSD_2007_2018.csv \
        5,10,15,20,25,30 45,60,90,120,150,180,210,240 ./models_out/  ${time}-${type} &

    done
done
