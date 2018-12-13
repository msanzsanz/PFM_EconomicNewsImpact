




ff_2017['forecast_error'] = ff_2017['forecast_error'].replace(np.nan, 'accurate', regex=True)
ff_2017['previous_error'] = ff_2017['previous_error'].replace(np.nan, 'accurate', regex=True)


def get_type(value):
    if value.find('K') != -1:
        return 'K'
    elif value.find('M') != -1:
        return 'M'
    elif value.find('%') != -1:
        return '%'
    elif value.find('<') != -1:
        return '<'
    else:
        return 'F'


def compute_error(forecasted, actual, fftype):
    magnitude = 1
    if fftype == 'K':
        magnitude = 1000
    elif fftype == 'M':
        magnitude = 1000000

    f = float(forecasted.split(fftype)[0]) * magnitude
    a = float(actual.split(fftype)[0]) * magnitude
    diff_per = abs(a - f) * 100 / abs(f)

    sign = 1 if f >= a else -1

    return diff_per * sign

compute_error('0.8%', '0.4%', '%')