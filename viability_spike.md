# INTRODUCTION

Before kicking-off the project, I need to nail down which questions I´d like to answer, in which order, and what data is available for the analysis. 
Therefore, the scope for this spike is to do an initial reseach to clarify these three things.

## QUESTIONS I WOULD LIKE TO ANSWER

1. Is it possible to predict, with > 70% confidence level, the price of a pair (e.g. EUR /USD) one hour after releasing a major economic new (e.g. unemployment rate in USA)?

2. Is it possible to forecast the value of an economic new based on previous data for that same new and previous data from other news?

3. Which are the bigger correlations in between news?

## DATA SOURCES

### Which Forex Calendar to use for data acquisition?

There are several websites that offer economic calendars and I have one main question at the beginning of the project: do they all provide the same forecast values?
I found this great article comparing the top 10 websites which solved my question: *"Quite often a trader will come across differences in the event forecast data released by two different calendars"*


Let´s see how the project evolves but, at least for now, I think having multiple forescasts will help me for modeling my own forecast.

For those interested in knowing more about economic calendars and what information is publically available in these websites, I would encourage to read the entire article: https://www.earnforex.com/blog/top-10-forex-calendars/ 
It´s a great in-deep analysis of the pros and cons of all top Forex Calendars available out there.

### From where I can get historical currency rates for my pairs of interest?

After some research on the web, I found this place: https://www.forextester.com/data/datasources
The source for this data is https://www.forexite.com/traderoom/default.asp, which it´s supposed to be quite accurate, based on forum's reviews 
