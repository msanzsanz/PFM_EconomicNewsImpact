from bs4 import BeautifulSoup
import requests
from datetime import datetime
import logging
import pandas as pd
import sys


def set_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='forexfactory.log',
                        filemode='w')

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_economic_calendar(start_link, end_link, week_number, ouput_path):

    d = []
    d_new_year = []
    curr_year = end_link[-4:]

    while start_link != end_link:

        # write to console current status
        logging.info('Scraping data for link: {}'.format(end_link))

        baseURL = 'https://www.forexfactory.com/'
        r = requests.get(baseURL + end_link)

        if r.url == 'https://www.forexfactory.com/calendar.php':
            logging.info('No more data available in the server')
            calendar_df = pd.DataFrame(d)
            calendar_df.to_csv(output_path + 'forexfactory_' + curr_year + '.csv')
            return 0

        data = r.text
        soup = BeautifulSoup(data, 'lxml')

        # Get the week coverage to check whether there is a change in the year
        week = soup.find('li', class_= "calendar__options left" )
        week_text = week.text.strip()

        year_start = week_text.split('-')[0].rstrip()[-4:]
        year_end = week_text.split('-')[1].rstrip()[-4:]
        week_of_year_change = year_start != year_end

        # get and parse table data, ignoring details and graph
        table = soup.find('table', class_='calendar__table')

        trs = table.select('tr.calendar__row.calendar_row')
        fields = ['time', 'currency', 'impact', 'event', 'actual', 'forecast', 'previous']

        for tr in trs:

            # Get date
            data = tr.select('td.calendar__cell.calendar__{}.{}'.format('date', 'date'))[0]
            if data.text.strip() != '' : curr_date = data.text.strip()

            # Some days have events, so we skip them
            if tr['data-eventid'] != '':

                try:
                    for field in fields:

                        data = tr.select('td.calendar__cell.calendar__{}.{}'.format(field, field))[0]

                        if field == 'time' and data.text.strip() != '':
                            # time is sometimes 'All Day' or 'Day X' (eg. WEF Annual Meetings)
                            if data.text.strip().find('am') == -1 and data.text.strip().find('pm') == -1:
                                curr_time = '12:00am'
                            else:
                                curr_time = data.text.strip()

                        elif field == 'currency':
                            currency = data.text.strip()

                        elif field == 'impact':
                            # when impact says 'Non-Economic' on mouseover, the relevant
                            # class name is 'Holiday', thus we do not use the classname
                            impact = data.find('span')['title'].split()[0]

                        elif field == 'event':
                            event = data.text.strip()

                        elif field == 'actual':
                            actual = data.text.strip()
                            forecast_error = ''
                            if data.find('span'):
                                forecast_error = data.find('span')['class'][0]


                        elif field == 'forecast':
                            forecast = data.text.strip()

                        elif field == 'previous':
                            previous = data.text.strip()
                            previous_error = ''
                            if data.find('span'):
                                if len(data.find('span')['class']) > 1:
                                    previous_error = data.find('span')['class'][1]

                    is_january = curr_date[-5:][:3] == "Jan"
                    year = curr_year
                    data_array = d

                    if week_of_year_change:
                        if not is_january:
                            # A new dataframe is needed for the new year
                            year = str(int(curr_year) -1)
                            data_array = d_new_year
                            week_number = 52
                        else:
                            week_number = 1


                    dt = datetime.strptime(year + ' ' + curr_date + ' '+ curr_time, '%Y %a%b %d %I:%M%p')
                    data_array.append({'datetime': str(dt), 'week': week_number, 'new': event, 'country': currency, \
                              'impact': impact, 'actual': actual, 'forecast_error': forecast_error, \
                              'forecast': forecast, 'previous': previous,
                              'previous_error': previous_error})


                except:

                    with open('errors_forexfactory.csv', 'a') as f:
                        f.write(end_link + '\n')
                        f.write(','.join([dt.strftime('%A, %B %d, %Y'), curr_time, currency, impact, event, \
                                          actual, forecast, previous, forecast_error, previous_error]))

        # We save the year csv
        if week_of_year_change or curr_date[-5:] == "Jan 7":
            # save events for that year
            logging.info('Successfully retrieved data for year: ' + curr_year)
            calendar_df = pd.DataFrame(d)
            calendar_df.to_csv(output_path + 'forexfactory_' + curr_year + '.csv')
            del (calendar_df)

            curr_year = str(int(curr_year) - 1)
            d = d_new_year
            d_new_year = []
            week_number = 53


        # get the link for the previous week
        end_link = soup.select('a.calendar__pagination.calendar__pagination--prev.prev')
        end_link = end_link[0]['href']
        week_number = week_number -1

    logging.info('End of the program')
    calendar_df = pd.DataFrame(d)
    calendar_df.to_csv(output_path + 'forexfactory_' + curr_year + '.csv')

    return 0


if __name__ == '__main__':
    '''
    Run this using the command 'python `script_name`.py 
    example: python forexfactory_scraper.py calendar.php?week=dec31.1900 calendar.php?week=oct28.2018
    '''

    set_logger()
    start_week = sys.argv[1]
    end_week = sys.argv[2]
    week_number = int(sys.argv[3])
    output_path = sys.argv[4]


    get_economic_calendar(start_week, end_week, week_number, output_path)
