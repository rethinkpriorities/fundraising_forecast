import random

import numpy as np
import pandas as pd

from pprint import pprint


N_SCENARIOS = 20000
MU = 0.15
SIGMA = 0.35
OFFSET = 0.85
VERBOSE = False
raw_data = pd.read_csv('forecast.csv')


def parse_currency(currency):
    if currency == ' $ -   ':
        return 0
    elif isinstance(currency, str):
        currency = currency.replace('$', '').replace(' ', '').replace(',', '')
        return float(currency)
    else:
        return 0


def print_money(money):
    return '${:,.2f}'.format(money)


# Either round to nearest hundred if <1000 or nearest thousand if >= 1000
def round_to_nearest(num):
    n_digits = len(str(num))
    n_digits = n_digits - 1
    if n_digits > 3:
        n_digits = 3
    return np.round(num, -n_digits)


def parse_percent(percent):
    if isinstance(percent, str):
        return float(percent.replace('%', '')) / 100
    else:
        return 0


def lognormal_sample(low, high, mu, sigma, offset):
    if (low > high) or (high < low):
        raise ValueError
    if low == high:
        return low
    else:
        r = np.random.lognormal(mu, sigma) - offset
        if r > 1:
            r = 1
        if r < 0:
            r = 0
        return low * (1 - r) + high * r


raw_data = raw_data[raw_data['Donor'] != 'Current Donors'][raw_data['Donor'] != 'Possible Donors']
raw_data = raw_data.drop([' 2018 Gift Amount ',
                          ' 2019 Gift Amount ',
                          ' 2020 Gift Amount ',
                          ' 2021 Gift Amount '], axis=1)
fundraising_data = {}
for index, row in raw_data.iterrows():
    donor = row['Donor']
    if donor and isinstance(donor, str):
        y2020_low = parse_currency(row[' 2020 Gift Potential - Low '])
        y2020_high = parse_currency(row[' 2020 Gift Potential - High '])
        y2020_prob = parse_percent(row['2020 Likelihood of Gift'])
        y2020_logmean = np.mean([lognormal_sample(y2020_low, y2020_high, MU, SIGMA, OFFSET) for _ in range(int(N_SCENARIOS / 10))])
        y2020_naive_ev = y2020_logmean * y2020_prob
        y2021_low = parse_currency(row[' 2021 Gift Potential - Low '])
        y2021_high = parse_currency(row[' 2021 Gift Potential - High '])
        y2021_prob = parse_percent(row['2021 Likelihood of Gift'])
        y2021_logmean = np.mean([lognormal_sample(y2021_low, y2021_high, MU, SIGMA, OFFSET) for _ in range(int(N_SCENARIOS / 10))])

        y2021_naive_ev = y2021_logmean * y2021_prob

        fundraising_data[donor] = {'2020': {'low': y2020_low,
                                            'high': y2020_high,
                                            'logmean': y2020_logmean,
                                            'prob': y2020_prob,
                                            'naive_ev': y2020_naive_ev},
                                   '2021': {'low': y2021_low,
                                            'high': y2021_high,
                                            'logmean': y2021_logmean,
                                            'prob': y2021_prob,
                                            'naive_ev': y2021_naive_ev}}

print('-')
print('### FUNDRAISING DATA ###')
pprint(sorted(list(fundraising_data.items())))
print('-')
print('-')

y2020_all_scenario_totals = []
y2021_all_scenario_totals = []
import pdb
pdb.set_trace()

for s in range(N_SCENARIOS):
    if VERBOSE:
        print('-')
        print('### SCENARIO {} ###'.format(s + 1))
    elif s % 100 == 0:
        print('### SCENARIO {} ###'.format(s + 1))
    y2020_fundraising_totals = {}
    y2021_fundraising_totals = {}

    for donor, donation in fundraising_data.items():
        if random.random() <= donation['2020']['prob']:
            y2020_donation = lognormal_sample(low=donation['2020']['low'],
                                              high=donation['2020']['high'],
                                              mu=MU, sigma=SIGMA, offset=OFFSET)
        else:
            y2020_donation = 0

        if random.random() <= donation['2021']['prob']:
            y2021_donation = lognormal_sample(low=donation['2021']['low'],
                                              high=donation['2021']['high'],
                                              mu=MU, sigma=SIGMA, offset=OFFSET)
        else:
            y2021_donation = 0

        y2020_donation = round_to_nearest(y2020_donation)
        y2021_donation = round_to_nearest(y2021_donation)

        if VERBOSE:
            print('{} gives {} in 2020 and {} in 2021'.format(donor,
                                                              print_money(y2020_donation),
                                                              print_money(y2021_donation)))
        y2020_fundraising_totals[donor] = y2020_donation
        y2021_fundraising_totals[donor] = y2021_donation

    y2020_total_raised = sum(y2020_fundraising_totals.values())
    y2021_total_raised = sum(y2021_fundraising_totals.values())
    if VERBOSE:
        print('TOTAL RAISED IN 2020: {}'.format(print_money(y2020_total_raised)))
        print('TOTAL RAISED IN 2021: {}'.format(print_money(y2021_total_raised)))
    y2020_all_scenario_totals.append(y2020_total_raised)
    y2021_all_scenario_totals.append(y2021_total_raised)

if VERBOSE:
    print('-')
    print('-')
    print('-')
joint_scenarios = np.array(y2020_all_scenario_totals) + np.array(y2021_all_scenario_totals)
y2020_percentiles = np.percentile(y2020_all_scenario_totals, [10, 20, 50, 80, 90])
y2021_percentiles = np.percentile(y2021_all_scenario_totals, [10, 20, 50, 80, 90])
joint_percentiles = np.percentile(joint_scenarios, [10, 20, 50, 80, 90])

y2020_percentiles = [print_money(x) for x in y2020_percentiles]
print('SCENARIO 2020 -- 10% {} - 20% {} - 50% {} - 80% {} - 90% {}'.format(*y2020_percentiles))

y2021_percentiles = [print_money(x) for x in y2021_percentiles]
print('SCENARIO 2021 -- 10% {} - 20% {} - 50% {} - 80% {} - 90% {}'.format(*y2021_percentiles))

joint_percentiles = [print_money(x) for x in joint_percentiles]
print('SCENARIO 2020+2021 -- 10% {} - 20% {} - 50% {} - 80% {} - 90% {}'.format(*joint_percentiles))

print('-')
print('-')
print('-')
print('## 2020 Detailed Breakdown ##')
print(list(zip([100 - z for z in range(100)],
               [print_money(y) for y in np.percentile(y2020_all_scenario_totals, range(100))])))

print('-')
print('-')
print('-')
print('## 2020-2021 Detailed Breakdown ##')
print(list(zip([100 - z for z in range(100)],
               [print_money(y) for y in np.percentile(joint_scenarios, range(100))])))

import pdb
pdb.set_trace()
