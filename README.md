## Fundraising Forecast

We use this to forecast our expected fundraising. Using distributions and a Monte Carlo simulation, we arrive at the probabilities of raising certain amounts of money over a one or two year time horizon.

The forecast is created in [a Google Sheet like this](https://docs.google.com/spreadsheets/d/1FxsakQIAikj3jEnMYRRNgNUE5VtqQoWpm7N9ciNg-RU/edit?usp=sharing), itemizing each donor, a chance that donor will donate for the year, and a conditional forecast of the range of donations expected conditional on the donor donating for that year. This sheet can then be downloaded as a CSV, moved into this repo as `forecast.csv`, and then `forecast.py` can be run (see `forecast-example.csv` in the repo for an example of what the CSV might look like).
