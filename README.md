## Fundraising Forecast

We use this to forecast our expected fundraising. Using distributions and a Monte Carlo simulation, we arrive at the probabilities of raising certain amounts of money over a one or two year time horizon.

The forecast is created in [a Google Sheet like this](https://docs.google.com/spreadsheets/d/1FxsakQIAikj3jEnMYRRNgNUE5VtqQoWpm7N9ciNg-RU/edit?usp=sharing), itemizing each donor, a chance that donor will donate for the year, and a conditional forecast of the range of donations expected conditional on the donor donating for that year. This sheet can then be downloaded as a CSV.

To run the forecast, use that CSV via `forecast.py`.

For example, `python3 forecast.py --csv forecast-example.csv`.

Look to `python3 forecast.py --help` for more details on what to do.
