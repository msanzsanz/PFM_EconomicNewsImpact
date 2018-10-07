from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import logging

def setLogger():

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='dailyfx.log',
                    filemode='w')

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def getEconomicCalendar(num_weeks):

    driver = webdriver.Firefox()
    driver.get("https://www.dailyfx.com/calendar")

    id_of_week = 'daily-cal'
    week_days = [id_of_week + str(i) for i in range(7)]

    d = []
    year = ''

    try:

        for week in range(num_weeks):

            for day in week_days:

                try:
                    day_rows = driver.find_element_by_id(day)

                    if len(day_rows) > 0:

                        news = day_rows.find_elements(By.TAG_NAME, 'tr')
                        date = news[0].text

                        if date.split(',')[2] != year & len(d) != '':

                            calendar_df = pd.DataFrame(d)
                            calendar_df.to_csv('dailyfx_' + year + '.csv')
                            d = []
                            del (calendar_df)
                            year = date.split(',')[2]

                        for new in news[1:]:

                            fields = new.find_elements(By.TAG_NAME, 'td')

                            if len(fields) > 8:
                                time = fields[0].text
                                new = fields[3].text
                                country = new.split()[0]
                                impact = fields[4].text
                                actual = fields[5].text
                                forecast = fields[6].text
                                previous = fields[7].text

                                forecast_error = ''
                                correction = fields[5].get_attribute('outerHTML').split('class="')
                                forecast_error = correction[1][0]

                                previous_error = ''
                                correction = fields[7].get_attribute('outerHTML').split('class="')
                                previous_error = correction[1][0]

                                # print("time: " + time + ", new: " + new + ", impact:" + impact +", actual:" \
                                #      + actual + ", forecast_error" + forecast_error + ", forecast:" + forecast \
                                #      + ", previous" + previous + ', previous_error: ' + previous_error )

                                d.append({'date': date, 'time': time, 'new': new, 'country': country, \
                                          'impact': impact, 'actual': actual, 'forecast_error': forecast_error, \
                                          'forecast': forecast, 'previous': previous, 'previous_error': previous_error})

                except:
                    with open('errors_dailyfx.csv', 'a') as f:
                        f.write('Week: ' + str(week) + ', Day: ' +  day )



            print('End of week: ', str(week))

            buttons = driver.find_element_by_class_name('grid-prev')
            buttons.click()

        return 0

    except:

        print('End of the history or some error')
        calendar_df = pd.DataFrame(d)
        calendar_df.to_csv('dailyfx_' + year + '.csv')


if __name__ == '__main__':

    '''
    Run this using the command 'python `script_name`.py 
    '''
    setLogger()

    # Let's aim for more than 20 years
    getEconomicCalendar(1500)

