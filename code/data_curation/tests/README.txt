###########################################################################################################
###
### Manual tests done to ensure that the data join between news releases and pair exchange is well-done
### Lot of effort was put here, as it´s key to ensure that market movements were due to macro-economic data
### releases.
### Worth to remind that we are analysing short-term impact, so a deviation in minutes (as those originated by
### DTS ) could ruin our models.
###
#########

Entire dataframe exported to: 20181107_macroeconomic_news_2007_2018.csv

#########

MANUAL CHECK WINTER_TIME, DTS ON ---

Manual check of market reaction to the "Unemployment claims" release on 25 Oct 2018, 8:30 (US/Eastern), which
corresponds to 12:30 GMT with DTS on.

"Ground truth": https://www.forexfactory.com/market.php
window-size: 5 minutes

 ET     GMT        close
(7:55)  11:55       1.1411
        12:25       1.1415
        12:30       1.1418
        12:35       1.1422
        12:55       1.1430 --
        13:55       1.1380 --
        14:55       1.1374


My processed data:
source: dukascopy
window-size: 5 minutes

            close
11:55       1.1411
12:25       1.1414
12:30       1.1417
12:35       1.1422
12:40       1.1411
12:55       1.1430 --
13:55       1.1380 --
14:55       1.1379


MANUAL CHECK SUMMER_TIME, DTS OFF ---

Manual check of market reaction to the "Unemployment Rate " release on 02 Feb 2018, 8:30 (US/Eastern), which
corresponds to 13:30 UCT (GMT with DTS off)

"Ground truth" ----------
source: https://www.forexfactory.com/market.php
window-size: 1 hour (5min aggregations are not provided for that long ago)

            close
ET          GMT         close
6-7           11-12            1.2490
7-8           12-13            1.2491
8-9           13-14            1.2445
9-10          14-15            1.2428


My processed data:
source: dukascopy
window-size: 1 hour

            close
11:55       1.2490
12:55       1.2490
13:25       1.2491
13:55       1.2445
14:55       1.2427



############################################################################################################