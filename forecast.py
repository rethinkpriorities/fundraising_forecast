import pickle
import random

import numpy as np
import pandas as pd

from scipy import stats
from pprint import pprint
from functools import partial
from collections import defaultdict


N_SCENARIOS = 40000

CREDIBLE_INTERVAL = 0.8  # 80% chance of donation falling within range, conditional on giving
# TODO: Should we jitter the chance of giving as well, to account for correlated errors / make the interval wider?

VERBOSE = False          # Change this to True for scenario-specific output
SCENARIO_RANGES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]


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


def lognormal_sample(low, high, interval):
    if (low > high) or (high < low):
        raise ValueError
    if low == high:
        return low
    else:
        log_low = np.log(low)
        log_high = np.log(high)
        mu = (log_high + log_low) / 2
        cdf_value = 0.5 + 0.5 * interval
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (log_high - mu) / normed_sigma
        return np.random.lognormal(mu, sigma)


raw_data = pd.read_csv('forecast.csv')
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
        y2021_low = parse_currency(row[' 2021 Gift Potential - Low '])
        y2021_high = parse_currency(row[' 2021 Gift Potential - High '])
        y2021_prob = parse_percent(row['2021 Likelihood of Gift'])

        fundraising_data[donor] = {'2020': {'low': y2020_low,
                                            'high': y2020_high,
                                            'prob': y2020_prob},
                                   '2021': {'low': y2021_low,
                                            'high': y2021_high,
                                            'prob': y2021_prob}}

y2020_all_scenario_totals = []
y2021_all_scenario_totals = []
y2020_fundraising_totals = defaultdict(partial(np.array, 0))
y2021_fundraising_totals = defaultdict(partial(np.array, 0))
joint_fundraising_totals = defaultdict(partial(np.array, 0))

for s in range(N_SCENARIOS):
    if s % 100 == 0:
        if VERBOSE:
            print('-')
            print('### SCENARIO {} ###'.format(s + 1))
        else:
            print('... Completed {}/{}'.format(s + 1, N_SCENARIOS))

    y2020_donations = []
    y2021_donations = []

    for donor, donation in fundraising_data.items():
        if random.random() <= donation['2020']['prob']:
            y2020_donation = lognormal_sample(low=donation['2020']['low'],
                                              high=donation['2020']['high'],
                                              interval=CREDIBLE_INTERVAL)
        else:
            y2020_donation = 0

        if random.random() <= donation['2021']['prob']:
            y2021_donation = lognormal_sample(low=donation['2021']['low'],
                                              high=donation['2021']['high'],
                                              interval=CREDIBLE_INTERVAL)
        else:
            y2021_donation = 0

        y2020_donation = round_to_nearest(y2020_donation)
        y2021_donation = round_to_nearest(y2021_donation)

        if s % 100 == 0 and VERBOSE:
            print('{} gives {} in 2020 and {} in 2021'.format(donor,
                                                              print_money(y2020_donation),
                                                              print_money(y2021_donation)))
        y2020_donations.append(y2020_donation)
        y2021_donations.append(y2021_donation)
        y2020_fundraising_totals[donor] = np.append(y2020_fundraising_totals[donor], y2020_donation)
        y2021_fundraising_totals[donor] = np.append(y2021_fundraising_totals[donor], y2021_donation)
        joint_fundraising_totals[donor] = np.append(joint_fundraising_totals[donor], y2020_donation + y2021_donation)

    y2020_total_raised = sum(y2020_donations)
    y2021_total_raised = sum(y2021_donations)
    if s % 100 == 0 and VERBOSE:
        print('TOTAL RAISED IN 2020: {}'.format(print_money(y2020_total_raised)))
        print('TOTAL RAISED IN 2021: {}'.format(print_money(y2021_total_raised)))
    y2020_all_scenario_totals.append(y2020_total_raised)
    y2021_all_scenario_totals.append(y2021_total_raised)

if VERBOSE:
    print('-')
    print('-')
    print('-')
joint_scenarios = np.array(y2020_all_scenario_totals) + np.array(y2021_all_scenario_totals)
y2020_percentiles = np.percentile(y2020_all_scenario_totals, SCENARIO_RANGES)
y2021_percentiles = np.percentile(y2021_all_scenario_totals, SCENARIO_RANGES)
joint_percentiles = np.percentile(joint_scenarios, SCENARIO_RANGES)

print('SCENARIO 2020 -- {}'.format(' -- '.join(['{}%: {}'.format(s, print_money(y2020_percentiles[i])) for i, s in enumerate(SCENARIO_RANGES)])))
print('SCENARIO 2021 -- {}'.format(' -- '.join(['{}%: {}'.format(s, print_money(y2021_percentiles[i])) for i, s in enumerate(SCENARIO_RANGES)])))
print('SCENARIO -- {}'.format(' -- '.join(['{}%: {}'.format(s, print_money(joint_percentiles[i])) for i, s in enumerate(SCENARIO_RANGES)])))

print('... Saving 1/7')
pickle.dump(fundraising_data, open('fundraising_data.p', 'wb'))
print('... Saving 2/7')
pickle.dump(y2020_fundraising_totals, open('y2020_fundraising_totals.p', 'wb'))
print('... Saving 3/7')
pickle.dump(y2021_fundraising_totals, open('y2021_fundraising_totals.p', 'wb'))
print('... Saving 4/7')
pickle.dump(joint_fundraising_totals, open('joint_fundraising_totals.p', 'wb'))
print('... Saving 5/7')
pickle.dump(y2020_all_scenario_totals, open('y2020_all_scenario_totals.p', 'wb'))
print('... Saving 6/7')
pickle.dump(y2021_all_scenario_totals, open('y2021_all_scenario_totals.p', 'wb'))
print('... Saving 7/7')
pickle.dump(joint_scenarios, open('joint_scenarios.p', 'wb'))

import pdb
pdb.set_trace()
