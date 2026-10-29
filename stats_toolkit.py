"""
Day 3 - Statistical Analysis Toolkit

RULE: every calculation is implemented with plain Python.
NumPy is imported ONLY in compare_with_numpy(), to verify the results.
The statistics module is not used.

Choices made in this toolkit
- Variance and standard deviation: the sample formula (n - 1) is the default,
  because the 40 days are a sample of an ongoing process, not every day ever.
- Quartiles: linear interpolation between closest ranks (Hyndman and Fan
  type 7). This is the default of NumPy and of Excel's QUARTILE.INC.
- Mode: every value that occurs most often. When every value occurs once
  there is no mode worth reporting, so an empty list is returned.
- Invalid values: blanks, placeholders such as '-' and 'n/a', text, numbers
  that cannot be read exactly, nan/inf and negative deposits are rejected
  with a reason. Nothing is guessed or repaired, except thousands
  separators that are written correctly, such as '1,250.50'.
"""
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"
REPORT_PATH = Path("output") / "stats_report.txt"
COLUMN = "daily_deposits_kes"

NUMPY_QUARTILE_METHOD = "linear"
NUMPY_DDOF = 1

PLACEHOLDERS = {"-", "--", "n/a", "na", "null", "none", "nil", "tbc", "?"}
THOUSANDS_PATTERN = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def parse_value(raw: str) -> tuple[float | None, str | None]:
    """Return (number, None) for a trustworthy value, or (None, reason) if rejected."""
    text = (raw or "").strip()
    if text == "":
        return None, "missing value"
    if text.lower() in PLACEHOLDERS:
        return None, "placeholder, not a number"
    if THOUSANDS_PATTERN.match(text):
        text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None, "not a valid number"
    if not math.isfinite(number):
        return None, "not a finite number"
    if number < 0:
        return None, "negative deposit"
    return number, None


def _find_column(fieldnames: list[str], column: str) -> str:
    """Match a column name ignoring case, spaces and a hidden BOM character."""
    wanted = column.strip().lower()
    for name in fieldnames:
        if name is not None and name.replace("\ufeff", "").strip().lower() == wanted:
            return name
    raise KeyError("Column '" + column + "' not found. Columns found: " + repr(fieldnames))


def load_records(path: Path, column: str) -> tuple[list[dict], list[tuple[int, str, str]], int]:
    """Read the CSV. Return (valid_records, rejected, rows_read).

    A valid record holds the value and a label (date, agent_id) for reports.
    row_number is the line number in the file, where the header is line 1.
    The delimiter (comma, semicolon, tab) is detected, and a BOM is ignored.
    """
    records = []
    rejected = []
    rows_read = 0
    with open(path, newline="", encoding="utf-8-sig") as file:
        sample = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample.split("\n")[0], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(file, dialect=dialect)
        if reader.fieldnames is None:
            raise KeyError("The file " + str(path) + " is empty")
        value_key = _find_column(reader.fieldnames, column)
        keys = {name.replace("\ufeff", "").strip().lower(): name for name in reader.fieldnames}
        for row_number, row in enumerate(reader, start=2):
            rows_read += 1
            raw = row[value_key]
            number, reason = parse_value(raw)
            if number is None:
                rejected.append((row_number, raw or "", reason))
            else:
                date = row.get(keys.get("date", ""), "") or ""
                agent = row.get(keys.get("agent_id", ""), "") or ""
                records.append({"value": number, "label": (date + ", " + agent).strip(", ")})
    return records, rejected, rows_read


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
    records, rejected, _ = load_records(path, column)
    values = [record["value"] for record in records]
    return values, [(row_number, raw) for row_number, raw, _ in rejected]


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def _require_values(values: list[float], name: str) -> None:
    if len(values) == 0:
        raise ValueError(name + "() needs at least one value, got an empty list")


def mean(values: list[float]) -> float:
    """Arithmetic mean."""
    _require_values(values, "mean")
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def _median_sorted(ordered: list[float]) -> float:
    size = len(ordered)
    middle = size // 2
    if size % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def median(values: list[float]) -> float:
    """Middle value, or the average of the two middle values for an even count."""
    _require_values(values, "median")
    return _median_sorted(sorted(values))


def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often, smallest first.

    If every value occurs only once there is no mode worth reporting, so the
    result is an empty list.
    """
    _require_values(values, "mode")
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    highest = 0
    for count in counts.values():
        if count > highest:
            highest = count
    if highest == 1:
        return []
    return sorted(value for value, count in counts.items() if count == highest)


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    _require_values(values, "variance")
    n = len(values)
    if sample and n < 2:
        raise ValueError("sample variance needs at least two values")
    centre = mean(values)
    squared_gaps = 0.0
    for value in values:
        squared_gaps += (value - centre) ** 2
    return squared_gaps / (n - 1 if sample else n)


def std_dev(values: list[float], sample: bool = True) -> float:
    """Square root of the variance, with the same sample/population choice."""
    return math.sqrt(variance(values, sample))


def _percentile_sorted(ordered: list[float], p: float) -> float:
    """Percentile by linear interpolation between closest ranks (type 7)."""
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _quartiles_sorted(ordered: list[float]) -> tuple[float, float, float]:
    return (
        _percentile_sorted(ordered, 0.25),
        _median_sorted(ordered),
        _percentile_sorted(ordered, 0.75),
    )


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3) using linear interpolation (Hyndman and Fan type 7)."""
    _require_values(values, "quartiles")
    return _quartiles_sorted(sorted(values))


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, both fences and the values outside them (in original order)."""
    _require_values(values, "iqr_outliers")
    if k < 0:
        raise ValueError("k must not be negative")
    q1, _, q3 = quartiles(values)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    indices = [i for i, value in enumerate(values) if value < lower or value > upper]
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "outliers": [values[i] for i in indices],
        "outlier_indices": indices,
    }


# ---------------------------------------------------------------------------
# The analyzer class
# ---------------------------------------------------------------------------
class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values: list[float], name: str = "values",
                 labels: list[str] | None = None) -> None:
        if len(values) == 0:
            raise ValueError("StatisticsAnalyzer needs at least one value")
        if labels is not None and len(labels) != len(values):
            raise ValueError("labels must have the same length as values")
        self.name = name
        self._values = [float(v) for v in values]
        self._labels = list(labels) if labels is not None else [""] * len(values)
        self._sorted = sorted(self._values)  # sorted once, reused by every method
        self._outlier_cache = {}
        self.rows_read = len(values)
        self.rejected = []

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        """Load and clean one column of a CSV file and wrap it in an analyzer."""
        records, rejected, rows_read = load_records(path, column)
        analyzer = cls(
            [r["value"] for r in records],
            name=column,
            labels=[r["label"] for r in records],
        )
        analyzer.rows_read = rows_read
        analyzer.rejected = rejected
        return analyzer

    # -- raw access ----------------------------------------------------------
    @property
    def values(self) -> list[float]:
        return list(self._values)

    def outlier_info(self, k: float = 1.5) -> dict:
        """IQR outlier details, calculated once per value of k."""
        if k not in self._outlier_cache:
            self._outlier_cache[k] = iqr_outliers(self._values, k)
        return self._outlier_cache[k]

    def outlier_labels(self, k: float = 1.5) -> list[tuple[float, str]]:
        info = self.outlier_info(k)
        return [(self._values[i], self._labels[i]) for i in info["outlier_indices"]]

    # -- statistics ----------------------------------------------------------
    def summary(self) -> dict:
        """Every statistic in one dict. Sample variance is None for a single value."""
        ordered = self._sorted
        q1, q2, q3 = _quartiles_sorted(ordered)
        n = len(ordered)
        var = variance(ordered, sample=True) if n >= 2 else None
        return {
            "count": n,
            "mean": mean(ordered),
            "median": q2,
            "mode": mode(ordered),
            "variance": var,
            "std_dev": math.sqrt(var) if var is not None else None,
            "min": ordered[0],
            "max": ordered[-1],
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
        }

    def without_outliers(self, k: float = 1.5) -> "StatisticsAnalyzer":
        """Return a NEW analyzer without the IQR outliers. This one is unchanged."""
        info = self.outlier_info(k)
        drop = set(info["outlier_indices"])
        keep = [i for i in range(len(self._values)) if i not in drop]
        return StatisticsAnalyzer(
            [self._values[i] for i in keep],
            name=self.name + " (outliers removed)",
            labels=[self._labels[i] for i in keep],
        )

    # -- reporting -----------------------------------------------------------
    @staticmethod
    def _number(value: float | None) -> str:
        return "n/a" if value is None else format(value, ",.2f")

    def report(self) -> str:
        """The statistics report: all data and outliers removed, side by side."""
        full = self.summary()
        clean_analyzer = self.without_outliers()
        clean = clean_analyzer.summary()
        info = self.outlier_info()
        num = self._number

        def modes(summary: dict) -> str:
            return ", ".join(num(m) for m in summary["mode"]) if summary["mode"] else "none"

        def row(label: str, left: str, right: str) -> str:
            return label.ljust(20) + left.ljust(22) + right

        lines = [
            "=== Statistics report: " + self.name + " ===",
            "Rows read         : " + str(self.rows_read),
            "Valid values      : " + str(len(self._values)),
            "Rejected values   : " + str(len(self.rejected)),
        ]
        for row_number, raw, reason in self.rejected:
            lines.append("  row " + str(row_number) + ": '" + raw + "' (" + reason + ")")
        lines += [
            "",
            row("", "all data", "outliers removed"),
            row("count", str(full["count"]), str(clean["count"])),
            row("mean", num(full["mean"]), num(clean["mean"])),
            row("median", num(full["median"]), num(clean["median"])),
            row("mode", modes(full), modes(clean)),
            row("variance (sample)", num(full["variance"]), num(clean["variance"])),
            row("std dev (sample)", num(full["std_dev"]), num(clean["std_dev"])),
            row("min / max",
                num(full["min"]) + " / " + num(full["max"]),
                num(clean["min"]) + " / " + num(clean["max"])),
            row("Q1 / Q3",
                num(full["q1"]) + " / " + num(full["q3"]),
                num(clean["q1"]) + " / " + num(clean["q3"])),
            row("IQR", num(full["iqr"]), num(clean["iqr"])),
            "",
            "Outlier fences      : lower " + num(info["lower_fence"])
            + ", upper " + num(info["upper_fence"]),
        ]
        found = self.outlier_labels()
        if found:
            parts = [num(value) + " (" + label + ")" for value, label in found]
            lines.append("Outliers            : " + ", ".join(parts))
        else:
            lines.append("Outliers            : none")
        return "\n".join(lines)

    def risk_note(self) -> str:
        """Three sentences for the risk team about what the outliers might mean."""
        info = self.outlier_info()
        found = self.outlier_labels()
        if not found:
            return ("No daily total fell outside the IQR fences, so no day needs "
                    "special review. The spread of the deposits looks consistent "
                    "across agents. The risk team can rely on the ordinary "
                    "controls for this period.")
        high = [item for item in found if item[0] > info["upper_fence"]]
        low = [item for item in found if item[0] < info["lower_fence"]]
        per_agent = {}
        for _, label in found:
            if label:
                agent = label.split(", ")[-1]
                per_agent[agent] = per_agent.get(agent, 0) + 1
        first = (str(len(found)) + " daily totals fall outside the fences ("
                 + str(len(high)) + " above " + self._number(info["upper_fence"])
                 + " KES and " + str(len(low)) + " below "
                 + self._number(info["lower_fence"]) + " KES).")
        if len(per_agent) == 1:
            agent = next(iter(per_agent))
            second = ("All of them belong to one agent, " + agent + ", which points "
                      "to a pattern for that agent and not to random noise.")
        elif per_agent:
            ranked = sorted(per_agent.items(), key=lambda item: (-item[1], item[0]))
            shares = ", ".join(agent + " accounts for " + str(count) for agent, count in ranked)
            second = ("The outliers are spread across agents (" + shares + "), so "
                      "the agent with the most of them should be reviewed first.")
        else:
            second = "The agents involved could not be identified from the data."
        third = ("Very high days could be genuine large transactions, structuring "
                 "or a data-entry error with an extra digit, while very low days "
                 "could be a float shortage or a missing digit, so each one should "
                 "be checked against the transaction count and the agent's records "
                 "before any action is taken.")
        return " ".join([first, second, third])


# ---------------------------------------------------------------------------
# Verification with NumPy
# ---------------------------------------------------------------------------
def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np

    values = np.array(analyzer.values, dtype=float)
    mine = analyzer.summary()

    def close(a: float, b: float) -> bool:
        return math.isclose(a, float(b), rel_tol=1e-9, abs_tol=1e-9)

    try:
        q1, q3 = np.percentile(values, [25, 75], method=NUMPY_QUARTILE_METHOD)
    except TypeError:  # NumPy older than 1.22 used the name 'interpolation'
        q1, q3 = np.percentile(values, [25, 75], interpolation=NUMPY_QUARTILE_METHOD)

    return {
        "mean": close(mine["mean"], np.mean(values)),
        "median": close(mine["median"], np.median(values)),
        "variance": close(mine["variance"], np.var(values, ddof=NUMPY_DDOF)),
        "std dev": close(mine["std_dev"], np.std(values, ddof=NUMPY_DDOF)),
        "Q1 / Q3": close(mine["q1"], q1) and close(mine["q3"], q3),
    }


def numpy_report(analyzer: StatisticsAnalyzer) -> str:
    """Format the NumPy comparison as MATCH or MISMATCH lines."""
    try:
        results = compare_with_numpy(analyzer)
    except ImportError:
        return "=== NumPy verification ===\nNumPy is not installed, so the check was skipped."
    notes = {
        "variance": "  (ddof=" + str(NUMPY_DDOF) + ")",
        "Q1 / Q3": "  (method=" + NUMPY_QUARTILE_METHOD + ")",
    }
    lines = ["=== NumPy verification ==="]
    for name, ok in results.items():
        verdict = "MATCH" if ok else "MISMATCH"
        lines.append(name.ljust(10) + verdict + notes.get(name, ""))
    return "\n".join(lines)


def run_unit_tests() -> str:
    """Run the unit tests in a separate process and return the Ran/OK summary."""
    folder = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "test_stats_toolkit"],
            cwd=folder, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return "=== Unit tests ===\nCould not run the tests: " + str(error)
    output = result.stderr.strip().splitlines()
    ran = [line for line in output if line.startswith("Ran ")]
    status = output[-1] if output else "no output"
    return "=== Unit tests ===\n" + (ran[0] if ran else "Ran ? tests") + "\n" + status


def build_report(path: Path = DATA_PATH, column: str = COLUMN) -> str:
    analyzer = StatisticsAnalyzer.from_csv(path, column)
    parts = [
        analyzer.report(),
        "",
        "Note for the risk team: " + analyzer.risk_note(),
        "",
        numpy_report(analyzer),
        "",
        run_unit_tests(),
    ]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    try:
        report_text = build_report()
    except FileNotFoundError:
        raise SystemExit("File not found: " + str(DATA_PATH))
    except KeyError as error:
        raise SystemExit(str(error))
    print(report_text)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print("Report saved to " + REPORT_PATH.as_posix())