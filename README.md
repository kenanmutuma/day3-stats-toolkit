# Day 3: Statistical Analysis Toolkit (Wakala Plus)

## Approach
Wakala Plus wants a small statistics toolkit that its risk team can read line by line. Every calculation in `stats_toolkit.py` is written in plain Python: mean, median, mode, variance, standard deviation, quartiles and IQR outlier detection. The `statistics` module is not used, and NumPy is imported in one place only, `compare_with_numpy()`, to verify the results afterwards.

The column `daily_deposits_kes` of `data/agent_deposits.csv` is read with the `csv` module and cleaned by `load_column()`. The cleaned values are wrapped in a `StatisticsAnalyzer` class that sorts the data once, produces a summary, returns a new analyzer without the outliers, and prints the report. The report is printed and saved to `output/stats_report.txt`.

## How to run
1. Install NumPy (needed only for the verification step): `pip install -r requirements.txt`
2. Run the report from the project folder: `python stats_toolkit.py`
3. Run the unit tests: `python -m unittest -v` (pytest also works: `pytest`)

The report also runs the unit tests and includes their result at the bottom.

## My choices
**Sample or population variance.** The report uses the sample formula (n - 1). The 40 days are a sample of an ongoing process, so the data does not cover every day the agents will ever trade, and dividing by n - 1 avoids understating the spread. `variance()` and `std_dev()` also support the population formula with `sample=False`.

**Quartile method.** Linear interpolation between the closest ranks (Hyndman and Fan type 7). It is the default of NumPy (`method="linear"`) and of Excel's `QUARTILE.INC`, and the NumPy comparison uses the same method. Outliers are values below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR.

**Mode.** Returns every value that occurs most often, as a list. When every value occurs only once there is no mode worth reporting, so an empty list is returned and the report shows `none`.

**What counts as invalid.** A value is rejected, with a reason, when it is blank (missing value), a placeholder such as `-` or `n/a`, text or a number that cannot be read exactly such as `12,40O` (not a valid number), `nan` or `inf`, or negative. Nothing is guessed. The one repair is thousands separators that are written correctly, such as `1,250.50`. Zero is accepted, because a day with no deposits can be real. The row number is the line number in the file, where the header is line 1.

## Results for this dataset
- 40 rows read, 35 valid values and 5 rejected: row 7 (missing), row 10 (`-`), row 14 (`n/a`), row 25 (`12,40O`) and row 29 (missing).
- Four outliers lie outside the fences (lower 8,530.19, upper 28,445.41): 96,500.00 (2026-09-20, AG-104), 1,200.00 (2026-10-02, AG-104), 88,400.00 (2026-10-06, AG-104) and 29,060.99 (2026-10-07, AG-101).
- The mean falls from 22,400.00 with all data to 18,349.65 without the outliers, and the sample standard deviation falls from 18,111.32 to 3,180.63, so a few extreme days account for most of the spread. The median barely moves (18,805.47 to 18,506.72).
- No value occurs twice, so there is no mode in either version of the data.

## NumPy verification
Mean, median, variance (ddof=1), standard deviation and Q1 and Q3 (method=linear) all report MATCH. Comparisons use `math.isclose` with a tolerance, not `==`. A mismatch would most likely come from using the population formula (ddof=0) or a different quartile method.

## Assumptions
- Each row is one agent's deposit total for one day, and `daily_deposits_kes` is the only column analysed.
- A blank or placeholder means the figure was not entered, so the day is excluded and not treated as zero.
- Outliers are flagged for review and not assumed to be errors. The report lists the date and agent of each so the risk team can check them.
- The 29,060.99 outlier is only just above the upper fence and is the weakest of the four.