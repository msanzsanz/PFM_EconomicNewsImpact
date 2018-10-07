from bs4 import BeautifulSoup
import requests
from datetime import datetime
import logging
import csv
import pandas as pd

def setLogger():

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='forexfactory.log',
                    filemode='w')
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def getEconomicCalendar(startlink,endlink):

    d = []

    while (startlink != endlink):

        # write to console current status
        logging.info('Scraping data for link: {}'.format(endlink))

        # get the page and make the soup
        baseURL = 'https://www.forexfactory.com/'
        r = requests.get(baseURL + endlink)
        data = r.text
        soup = BeautifulSoup(data, 'lxml')

        # get and parse table data, ignoring details and graph
        table = soup.find('table', class_='calendar__table')


        # do not use the '.calendar__row--grey' css selector (reserved for historical data)
        trs = table.select('tr.calendar__row.calendar_row')
        fields = ['date','time','currency','impact','event','actual','forecast','previous']

        # some rows do not have a date (cells merged)
        curr_year = endlink[-4:]
        curr_date = ''
        curr_time = ''
        for tr in trs:

            # Some weeks have events, so we skip them
            if tr['data-eventid'] != '':


                # fields may mess up sometimes, see Tue Sep 25 2:45AM French Consumer Spending
                # in that case we append to errors.csv the date time where the error is
                try:
                    for field in fields:
                        data = tr.select('td.calendar__cell.calendar__{}.{}'.format(field,field))[0]

                        if field=='date' and data.text.strip()!='':
                            curr_date = data.text.strip()

                        elif field=='time' and data.text.strip()!='':
                            # time is sometimes 'All Day' or 'Day X' (eg. WEF Annual Meetings)
                            if data.text.strip().find('Day')!=-1:
                                curr_time = '12:00am'
                            else:
                                curr_time = data.text.strip()

                        elif field=='currency':
                            currency = data.text.strip()

                        elif field=='impact':
                            # when impact says 'Non-Economic' on mouseover, the relevant
                            # class name is 'Holiday', thus we do not use the classname
                            impact = data.find('span')['title'].split()[0]

                        elif field=='event':
                            event = data.text.strip()

                        elif field=='actual':
                            actual = data.text.strip()
                            forecast_error = ''
                            if data.find('span'):
                                forecast_error = data.find('span')['class'][0]


                        elif field=='forecast':
                            forecast = data.text.strip()

                        elif field=='previous':
                            previous = data.text.strip()
                            previous_error = ''
                            if data.find('span'):
                                if len(data.find('span')['class']) > 1:
                                    previous_error = data.find('span')['class'][1]

                    dt = datetime.strptime(curr_year + curr_date, '%Y%a%b %d')

                    #print(','.join([dt.strftime('%A, %B %d, %Y'),curr_time, currency,impact,event,actual,forecast,previous, forecast_error, previous_error]))

                    d.append({'date': str(dt), 'time': curr_time, 'new': event, 'country': currency, \
                              'impact': impact, 'actual': actual, 'forecast_error': forecast_error, \
                              'forecast': forecast, 'previous': previous, 'previous_error': previous_error})

                except:

                    with open('errors.csv','a') as f:
                        csv.writer(f).writerow([curr_year,curr_date,curr_time, currency,impact,event,actual,forecast,previous, forecast_error, previous_error])

        # get the link for the previous week
        endlink = soup.select('a.calendar__pagination.calendar__pagination--prev.prev')


        try:
            endlink = endlink[0]['href']
            getEconomicCalendar(startlink, endlink)

        except:
            logging.info('No more data available in the server')
            calendar_df = pd.DataFrame(d)
            calendar_df.to_csv('forexfactory.csv')

    # exit recursion when last available link has reached
    logging.info('Successfully retrieved data')
    calendar_df = pd.DataFrame(d)
    calendar_df.to_csv('forexfactory.csv')

    return 0


if __name__ == '__main__':

    '''
    Run this using the command 'python `script_name`.py >> `output_name`.csv'
    '''

    setLogger()
    getEconomicCalendar('calendar.php?week=jan7.1990','calendar.php?week=sep30.2018')


