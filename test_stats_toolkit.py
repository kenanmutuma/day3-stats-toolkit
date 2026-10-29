"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""
import math
import os
import tempfile
import unittest
from pathlib import Path

from stats_toolkit import (
    StatisticsAnalyzer,
    iqr_outliers,
    load_column,
    mean,
    median,
    mode,
    quartiles,
    std_dev,
    variance,
)


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        self.assertAlmostEqual(mean([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(mean([10]), 10.0)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([3, 1, 2]), 2)          # odd: the middle value
        self.assertEqual(median([4, 1, 3, 2]), 2.5)     # even: average of 2 and 3

    def test_mode_with_ties(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1, 2])  # two values tie
        self.assertEqual(mode([5, 5, 5, 1]), [5])

    def test_mode_when_every_value_appears_once(self):
        self.assertEqual(mode([1, 2, 3]), [])            # documented: no mode

    def test_empty_input_raises(self):
        for function in (mean, median, mode, variance, std_dev, quartiles, iqr_outliers):
            with self.assertRaises(ValueError):
                function([])


class TestSpread(unittest.TestCase):
    data = [2, 4, 4, 4, 5, 5, 7, 9]

    def test_sample_vs_population_variance(self):
        self.assertAlmostEqual(variance(self.data, sample=False), 4.0)
        self.assertAlmostEqual(variance(self.data, sample=True), 32 / 7)
        self.assertAlmostEqual(std_dev(self.data, sample=False), 2.0)

    def test_std_dev_single_value(self):
        self.assertEqual(std_dev([7], sample=False), 0.0)
        with self.assertRaises(ValueError):
            std_dev([7], sample=True)                    # n - 1 would be zero


class TestQuartiles(unittest.TestCase):
    def test_quartiles_use_linear_interpolation(self):
        self.assertEqual(quartiles([1, 2, 3, 4, 5]), (2, 3, 4))
        self.assertEqual(quartiles([1, 2, 3, 4]), (1.75, 2.5, 3.25))


class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers([10, 10, 10, 10, 10])
        self.assertEqual(result["outliers"], [])
        self.assertEqual(result["iqr"], 0)

    def test_detects_obvious_outlier(self):
        result = iqr_outliers([10, 11, 12, 13, 14, 100])
        self.assertAlmostEqual(result["q1"], 11.25)
        self.assertAlmostEqual(result["q3"], 13.75)
        self.assertAlmostEqual(result["upper_fence"], 17.5)
        self.assertEqual(result["outliers"], [100])

    def test_detects_low_outlier(self):
        result = iqr_outliers([1, 50, 51, 52, 53, 54])
        self.assertEqual(result["outliers"], [1])


class TestLoading(unittest.TestCase):
    def test_invalid_and_missing_values_are_rejected(self):
        rows = ["date,amount", "d1,10", "d2,", "d3,-", "d4,n/a", "d5,abc",
                "d6,\"1,250.50\"", "d7,-5", "d8,nan", "d9,\"12,40O\""]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            values, rejected = load_column(path, "amount")
        self.assertEqual(values, [10.0, 1250.5])
        rejected_rows = [row_number for row_number, _ in rejected]
        self.assertEqual(rejected_rows, [3, 4, 5, 6, 8, 9, 10])


class TestAnalyzer(unittest.TestCase):
    def test_without_outliers_returns_new_analyzer(self):
        analyzer = StatisticsAnalyzer([10, 11, 12, 13, 14, 100], name="demo")
        cleaned = analyzer.without_outliers()
        self.assertIsNot(cleaned, analyzer)
        self.assertEqual(cleaned.summary()["count"], 5)
        self.assertEqual(analyzer.summary()["count"], 6)   # original unchanged
        self.assertEqual(cleaned.summary()["max"], 14)

    def test_data_is_sorted_once(self):
        analyzer = StatisticsAnalyzer([3, 1, 2])
        before = analyzer._sorted
        analyzer.summary()
        analyzer.report()
        self.assertIs(analyzer._sorted, before)

    def test_summary_values(self):
        summary = StatisticsAnalyzer([1, 2, 3, 4, 5]).summary()
        self.assertEqual(summary["mean"], 3)
        self.assertEqual(summary["median"], 3)
        self.assertEqual(summary["q1"], 2)
        self.assertEqual(summary["q3"], 4)
        self.assertTrue(math.isclose(summary["variance"], 2.5))

    def test_empty_analyzer_raises(self):
        with self.assertRaises(ValueError):
            StatisticsAnalyzer([])


if __name__ == "__main__":
    unittest.main()