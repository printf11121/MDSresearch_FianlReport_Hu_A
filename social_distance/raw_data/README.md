# Raw data

The raw datasets are not committed because of their size and redistribution
considerations. Place the following files in this directory before rebuilding
the cleaned datasets:

- `australia.csv`: Imperial College London YouGov COVID-19 Behaviour Tracker
  Australian survey data. See the [YouGov COVID-19 tracker repository](https://github.com/YouGov-Data/covid-19-tracker).
- `OxCGRT_AUS_latest.csv`: Australian national and subnational policy records
  from the Oxford COVID-19 Government Response Tracker. See the
  [OxCGRT repository](https://github.com/OxCGRT/covid-policy-dataset).

The project includes derived data in `../data/` so that the reported modelling
results can be inspected without committing the large raw files.
