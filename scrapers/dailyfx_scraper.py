from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import logging
import time as t


def setLogger():

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='dailyfx.log',
                    filemode='w')

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def rewind_weeks(driver, num_weeks):

    for i in range(num_weeks):
        buttons = driver.find_element_by_class_name('grid-prev')
        buttons.click()
        t.sleep(1.5)

    return 0


def getEconomicCalendar(num_weeks, start_week, chunk_size):

    d = []
    year = '2018'

    id_of_week = 'daily-cal'
    week_days = [id_of_week + str(i) for i in range(7)]
    week_days.reverse()

    f = open('errors_dailyfx.csv', 'a')

    i = 0
    driver = webdriver.Firefox()
    driver.get("https://www.dailyfx.com/calendar")

    rewind_weeks(driver, start_week)

    try:

        for week in range(num_weeks):

            if i > chunk_size:
                driver.quit()
                driver = webdriver.Firefox()
                driver.get("https://www.dailyfx.com/calendar")
                i = 0
                start_week += chunk_size + 1
                rewind_weeks(driver, start_week)

            else:

                i += 1

            for day in week_days:

                try:
                    day_rows = driver.find_element_by_id(day)

                    if day_rows.text != '':

                        news = day_rows.find_elements(By.TAG_NAME, 'tr')
                        date = news[0].text
                        logging.info('Day: {} '.format(date))
                        year_row = date.split(',')[2].split()[0]

                        if ((year_row != year) & (len(d) != 0)):

                            logging.info('End of year: {} '.format(year))
                            calendar_df = pd.DataFrame(d)
                            calendar_df.to_csv('dailyfx_' + year + '.csv')
                            d = []
                            del (calendar_df)
                            year = year_row

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


                                d.append({'date': date, 'time': time, 'new': new, 'country': country, \
                                          'impact': impact, 'actual': actual, 'forecast_error': forecast_error, \
                                          'forecast': forecast, 'previous': previous, 'previous_error': previous_error})

                except:
                    f.write('week: ' + str(week) + ', day: ' + str(day) + '\n')


            logging.info('End of week: {} '.format(week))


            buttons = driver.find_element_by_class_name('grid-prev')
            buttons.click()



        return 0

    except:

        logging.info('End of the history or some error')
        calendar_df = pd.DataFrame(d)
        calendar_df.to_csv('dailyfx_' + year + '.csv')


if __name__ == '__main__':

    '''
    Run this using the command 'python `script_name`.py 
    '''
    setLogger()

    # Let's aim for more than 20 years
    getEconomicCalendar(1500, 342, 10)

