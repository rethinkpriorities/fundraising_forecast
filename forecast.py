import pickle
import random
import argparse

import numpy as np
import pandas as pd
import squigglepy as sq

from scipy import stats
from pprint import pprint
from functools import partial
from collections import defaultdict


parser = argparse.ArgumentParser(description='Forecast fundraising')
parser.add_argument('--scenarios', type=int, help='How many Monte Carlo simulations to run?', default=10000)
parser.add_argument('--interval', type=float, help='Width of the credible interval for the gift, conditional on giving', default=0.9)
# TODO: Should we jitter the chance of giving as well, to account for correlated errors / make the interval wider?

error_mean_doc = ('Specify an additional constant error term that is normally distributed with an interval specified by `CREDIBLE_INTERVAL`,'
                  'a mean specified `CONSTANT_ERROR_MEAN`, and a width of the interval specified by `CONSTANT_ERROR_WIDTH` (width is on both sides of the mean).'
                  'Set `CONSTANT_ERROR_WIDTH` to 0 to ignore.')
parser.add_argument('--constant_error_mean',type=float, help=error_mean_doc, default=0)
parser.add_argument('--constant_error_width', type=float, help='Width of the interval for the constant error term', default=0)
parser.add_argument('--csv', type=str, help='Define the relative path to the CSV with the donor information', default='forecast-example.csv')
parser.add_argument('--path', type=str, help='Define a custom path for the saved model outputs', default='')
parser.add_argument('--save', type=bool, help='Set to False to not save (overwrite) model outputs', default=True)
parser.add_argument('--update_every', type=int, help='How many steps to run before updating?', default=1000)
parser.add_argument('--verbose', type=bool, help='Set to True to get scenario-specific output', default=False)
args = parser.parse_args()

N_SCENARIOS = args.scenarios
CREDIBLE_INTERVAL = args.interval
CONSTANT_ERROR_MEAN = args.constant_error_mean
CONSTANT_ERROR_WIDTH = args.constant_error_width
VERBOSE = args.verbose
CSV = args.csv
SAVE = args.save
PATH = args.path
update_every = args.update_every

SCENARIO_RANGES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]  # Generate printed output for these percentiles


def parse_currency(currency):
    if isinstance(currency, int) or isinstance(currency, float):
        return currency
    if currency == ' $ -   ':
        return 0
    if isinstance(currency, str):
        currency = currency.replace('$', '').replace(' ', '').replace(',', '')
        return float(currency)
    else:
        return 0


def print_money(money):
    if money >= 0:
        return '${:,.2f}'.format(money)
    else:
        return '-${:,.2f}'.format(-money)


# Either round to nearest hundred if <1000 or nearest thousand if >= 1000
def round_to_nearest(num):
    n_digits = len(str(num))
    n_digits = n_digits - 1
    if n_digits > 3:
        n_digits = 3
    return np.round(num, -n_digits)


def parse_percent(percent):
    if isinstance(percent, int) or isinstance(percent, float):
        return percent
    if isinstance(percent, str):
        return float(percent.replace('%', '')) / 100
    else:
        return 0


raw_data = pd.read_csv(CSV)
raw_data = raw_data[['Donor', '2021 Gift Potential - Low', '2021 Gift Potential - High',
                     '2021 Likelihood of Gift', '2022 Gift Potential - Low',
                     '2022 Gift Potential - High', '2022 Likelihood of Gift']]
raw_data = raw_data[raw_data['Donor'] != 'Current Donors'][raw_data['Donor'] != 'Possible Donors']
fundraising_data = {}
for index, row in raw_data.iterrows():
    donor = row['Donor']
    if donor and isinstance(donor, str):
        y2021_low = parse_currency(row['2021 Gift Potential - Low'])
        y2021_high = parse_currency(row['2021 Gift Potential - High'])
        y2021_prob = parse_percent(row['2021 Likelihood of Gift'])
        y2022_low = parse_currency(row['2022 Gift Potential - Low'])
        y2022_high = parse_currency(row['2022 Gift Potential - High'])
        y2022_prob = parse_percent(row['2022 Likelihood of Gift'])

        fundraising_data[donor] = {'2021': {'low': y2021_low,
                                            'high': y2021_high,
                                            'prob': y2021_prob},
                                   '2022': {'low': y2022_low,
                                            'high': y2022_high,
                                            'prob': y2022_prob}}

y2021_all_scenario_totals = []
y2022_all_scenario_totals = []
y2021_fundraising_totals = defaultdict(partial(np.array, 0))
y2022_fundraising_totals = defaultdict(partial(np.array, 0))
joint_fundraising_totals = defaultdict(partial(np.array, 0))
constant_errors = []

for s in range(N_SCENARIOS):
    if s % update_every == 0:
        if VERBOSE:
            print('-')
            print('### SCENARIO {} ###'.format(s + 1))
        else:
            print('... Completed {}/{}'.format(s + 1, N_SCENARIOS))

    y2021_donations = []
    y2022_donations = []

    for donor, donation in fundraising_data.items():
        if random.random() <= donation['2021']['prob']:
            y2021_donation = sq.sample(sq.lognorm(donation['2021']['low'],
                                                  donation['2021']['high']),
                                       credibility=CREDIBLE_INTERVAL)
        else:
            y2021_donation = 0

        if random.random() <= donation['2022']['prob']:
            y2022_donation = sq.sample(sq.lognorm(donation['2022']['low'],
                                                  donation['2022']['high']),
                                       credibility=CREDIBLE_INTERVAL)
        else:
            y2022_donation = 0

        y2021_donation = round_to_nearest(y2021_donation)
        y2022_donation = round_to_nearest(y2022_donation)

        if s % update_every == 0 and VERBOSE:
            print('{} gives {} in 2021 and {} in 2022'.format(donor,
                                                              print_money(y2021_donation),
                                                              print_money(y2022_donation)))
        y2021_donations.append(y2021_donation)
        y2022_donations.append(y2022_donation)
        y2021_fundraising_totals[donor] = np.append(y2021_fundraising_totals[donor], y2021_donation)
        y2022_fundraising_totals[donor] = np.append(y2022_fundraising_totals[donor], y2022_donation)
        joint_fundraising_totals[donor] = np.append(joint_fundraising_totals[donor], y2021_donation + y2022_donation)

    y2021_total_raised = sum(y2021_donations)
    if CONSTANT_ERROR_WIDTH > 0:
        scenario_constant_error = sq.sample(sq.norm(-CONSTANT_ERROR_WIDTH/2.0 + CONSTANT_ERROR_MEAN,
                                                    CONSTANT_ERROR_WIDTH/2.0 + CONSTANT_ERROR_MEAN),
                                            credibility=CREDIBLE_INTERVAL)
    else:
        scenario_constant_error = 0

    y2022_total_raised = sum(y2022_donations) + scenario_constant_error

    if s % update_every == 0 and VERBOSE:
        print('CONSTANT ERROR TERM: {}'.format(print_money(scenario_constant_error)))
        print('TOTAL RAISED IN 2021: {}'.format(print_money(y2021_total_raised)))
        print('TOTAL RAISED IN 2022: {}'.format(print_money(y2022_total_raised)))

    y2021_all_scenario_totals.append(y2021_total_raised)
    y2022_all_scenario_totals.append(y2022_total_raised)
    constant_errors.append(scenario_constant_error)

if VERBOSE:
    print('-')
    print('-')
    print('-')


joint_scenarios = np.array(y2021_all_scenario_totals) + np.array(y2022_all_scenario_totals)
y2021_percentiles = np.percentile(y2021_all_scenario_totals, SCENARIO_RANGES)
y2022_percentiles = np.percentile(y2022_all_scenario_totals, SCENARIO_RANGES)
joint_percentiles = np.percentile(joint_scenarios, SCENARIO_RANGES)

print('SCENARIO 2021 -- {}'.format(' -- '.join(['{}%: {}'.format(s, print_money(y2021_percentiles[i])) for i, s in enumerate(SCENARIO_RANGES)])))
print('SCENARIO 2022 -- {}'.format(' -- '.join(['{}%: {}'.format(s, print_money(y2022_percentiles[i])) for i, s in enumerate(SCENARIO_RANGES)])))
print('SCENARIO 2021+2022 -- {}'.format(' -- '.join(['{}%: {}'.format(s, print_money(joint_percentiles[i])) for i, s in enumerate(SCENARIO_RANGES)])))

if SAVE:
    print('... Saving 1/8')
    pickle.dump(fundraising_data, open('{}fundraising_data.p'.format(PATH), 'wb'))
    print('... Saving 2/8')
    pickle.dump(y2021_fundraising_totals, open('{}y2021_fundraising_totals.p'.format(PATH), 'wb'))
    print('... Saving 3/8')
    pickle.dump(y2022_fundraising_totals, open('{}y2022_fundraising_totals.p'.format(PATH), 'wb'))
    print('... Saving 4/8')
    pickle.dump(joint_fundraising_totals, open('{}joint_fundraising_totals.p'.format(PATH), 'wb'))
    print('... Saving 5/8')
    pickle.dump(y2021_all_scenario_totals, open('{}y2021_all_scenario_totals.p'.format(PATH), 'wb'))
    print('... Saving 6/8')
    pickle.dump(y2022_all_scenario_totals, open('{}y2022_all_scenario_totals.p'.format(PATH), 'wb'))
    print('... Saving 7/8')
    pickle.dump(joint_scenarios, open('{}joint_scenarios.p'.format(PATH), 'wb'))
    print('... Saving 8/8')
    pickle.dump(constant_errors, open('{}constant_errors.p'.format(PATH), 'wb'))

