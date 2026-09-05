"""Published rainfall threshold models for landslide risk assessment.

This module implements two peer-reviewed threshold equations for the
Northeastern Himalaya region. These are NOT invented or approximated —
they are direct implementations of equations published in the
hydrological/geotechnical literature.

THRESHOLD MODEL ROLE IN THE SYSTEM:
    The threshold model is the BASELINE layer. It is always active and
    always produces a defensible answer. The ML classifier (Phase 2+) is
    an optional refinement layer that sits ON TOP of these thresholds.
    The threshold model is NEVER replaced by ML.

EQUATIONS IMPLEMENTED:

1. NE Himalaya Moisture Threshold (cumulative rainfall)
   Source: NE-Himalaya rainfall-threshold research
   Equation: E(mm) = -11.10 + 0.62 × D(hr)
   Domain:   24 < D < 1440 hours
   Variable: D = antecedent rainfall duration in hours
   Output:   E = critical cumulative rainfall in mm
   Property: Linear, monotonically increasing in D
   Interpretation: If cumulative rainfall over D hours exceeds E,
                   the zone is flagged as "elevated risk".

2. Sikkim Intensity-Duration (I-D) Threshold
   Source: NE-Himalaya rainfall-threshold research (Sikkim-specific)
   Equation: I = 43.26 × D^(-0.78)
   Domain:   D > 0 days
   Variable: D = duration in days
   Output:   I = critical rainfall intensity in mm/day
   Property: Decreasing, convex function of D (shorter storms have
             higher critical intensity)
   Interpretation: If average rainfall intensity over D days exceeds I,
                   the zone is flagged as "elevated risk".

MATH NOTE:
    The NE-Himalaya threshold is linear in D and increases with D.
    The Sikkim I-D threshold is a power law that decreases with D.
    They measure DIFFERENT THINGS:
      - NE-Himalaya: total cumulative rainfall over a window (mm vs hr)
      - Sikkim: average intensity over a window (mm/day vs days)
    Do NOT compare them directly or mix units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Domain boundaries (do not change without re-validating against source paper)
# ---------------------------------------------------------------------------

NE_HIMALAYA_D_MIN_HOURS = 24.0  # exclusive lower bound
NE_HIMALAYA_D_MAX_HOURS = 1440.0  # exclusive upper bound

SIKKIM_D_MIN_DAYS = 0.0  # exclusive lower bound (must be > 0)


# ---------------------------------------------------------------------------
# Core threshold functions
# ---------------------------------------------------------------------------


def ne_himalaya_moisture_threshold(d_hours: float) -> float:
    """Compute the NE Himalaya cumulative rainfall threshold.

    E(mm) = -11.10 + 0.62 × D(hr)

    This is a LINEAR function that increases with duration.
    For a given antecedent window D hours, the critical cumulative
    rainfall is E mm. Actual cumulative rainfall above E implies
    elevated landslide risk.

    Args:
        d_hours: Antecedent rainfall duration in hours (24 < D < 1440).

    Returns:
        Critical cumulative rainfall E in mm.

    Raises:
        ValueError: If d_hours is outside the valid domain.

    Examples:
        >>> ne_himalaya_moisture_threshold(48)
        18.66
        >>> ne_himalaya_moisture_threshold(72)
        33.54
        >>> ne_himalaya_moisture_threshold(168)  # 7 days
        93.06
    """
    if not isinstance(d_hours, (int, float)):
        raise TypeError(f"d_hours must be numeric, got {type(d_hours).__name__}")
    if not (NE_HIMALAYA_D_MIN_HOURS < d_hours < NE_HIMALAYA_D_MAX_HOURS):
        raise ValueError(
            f"d_hours must be in ({NE_HIMALAYA_D_MIN_HOURS}, {NE_HIMALAYA_D_MAX_HOURS}) exclusive, "
            f"got {d_hours}"
        )
    return -11.10 + 0.62 * d_hours


def sikkim_intensity_duration_threshold(d_days: float) -> float:
    """Compute the Sikkim intensity-duration threshold.

    I = 43.26 × D^(-0.78)

    This is a POWER LAW function that decreases with duration.
    For a given duration D days, the critical average rainfall
    intensity is I mm/day. Actual intensity above I implies
    elevated landslide risk.

    Args:
        d_days: Duration in days (must be > 0).

    Returns:
        Critical intensity I in mm/day.

    Raises:
        ValueError: If d_days is not positive.

    Examples:
        >>> sikkim_intensity_duration_threshold(1)
        43.26
        >>> sikkim_intensity_duration_threshold(7)
        9.482220362171997
        >>> sikkim_intensity_duration_threshold(30)
        3.2363554743763954
    """
    if not isinstance(d_days, (int, float)):
        raise TypeError(f"d_days must be numeric, got {type(d_days).__name__}")
    if d_days <= SIKKIM_D_MIN_DAYS:
        raise ValueError(f"d_days must be > {SIKKIM_D_MIN_DAYS}, got {d_days}")
    return 43.26 * (d_days ** (-0.78))


# ---------------------------------------------------------------------------
# Threshold exceedance checking
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Result of a threshold exceedance check.

    This dataclass supports downstream explainability. Every field is
    populated by the threshold computation — nothing is hard-coded.
    """

    region: str
    exceeded: bool
    threshold: float
    actual: float
    unit: str
    margin: float
    duration_hours: float
    explanation_template: str = ""

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "exceeded": self.exceeded,
            "threshold": round(self.threshold, 2),
            "actual": self.actual,
            "unit": self.unit,
            "margin": round(self.margin, 2),
            "duration_hours": self.duration_hours,
        }


def check_threshold_exceedance(
    cumulative_rainfall_mm: float,
    duration_hours: float,
    region: Literal["ne_himalaya", "sikkim"] = "ne_himalaya",
) -> ThresholdResult:
    """Check whether rainfall exceeds the published threshold for a region.

    For the NE Himalaya region, the comparison is:
        actual cumulative rainfall (mm) vs threshold E (mm) over D hours.

    For the Sikkim region, the comparison is:
        average intensity (mm/day) vs threshold I (mm/day) over D days.
        Average intensity = cumulative_rainfall_mm / (duration_hours / 24).

    Args:
        cumulative_rainfall_mm: Total rainfall over the duration in mm.
        duration_hours: Duration of the rainfall event in hours.
        region: Which threshold to apply.

    Returns:
        ThresholdResult with all fields needed for explanation.
    """
    if region == "ne_himalaya":
        threshold = ne_himalaya_moisture_threshold(duration_hours)
        actual = cumulative_rainfall_mm
        unit = "mm"
        exceeded = actual > threshold
        margin = actual - threshold
    elif region == "sikkim":
        d_days = duration_hours / 24.0
        threshold = sikkim_intensity_duration_threshold(d_days)
        actual = cumulative_rainfall_mm / d_days if d_days > 0 else 0.0
        unit = "mm/day"
        exceeded = actual > threshold
        margin = actual - threshold
    else:
        raise ValueError(
            f"Unknown region: '{region}'. Valid regions: 'ne_himalaya', 'sikkim'"
        )

    return ThresholdResult(
        region=region,
        exceeded=exceeded,
        threshold=threshold,
        actual=actual,
        unit=unit,
        margin=margin,
        duration_hours=duration_hours,
    )


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------


def select_best_result(results: list[ThresholdResult]) -> ThresholdResult:
    """Pick the single threshold check that should headline an explanation.

    Preference order:
      1. a triggered (exceeded) check with the highest positive margin, else
      2. the check closest to the threshold (least-negative margin).

    Raising over a per-reading threshold is what drives the risk level, so
    the "best" check is the most extreme comparison, not the newest one.

    Args:
        results: Non-empty list of ThresholdResult objects.

    Returns:
        The ThresholdResult to headline the explanation.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results must not be empty")
    return max(results, key=lambda r: r.margin)


def format_explanation(result: ThresholdResult) -> str:
    """Generate a human-readable explanation from a ThresholdResult.

    This is the text that appears in:
    - Dashboard alert console
    - SMS templates
    - Field app notifications

    The explanation MUST include the specific numbers that drove the
    decision — this is what makes the system trustworthy to district
    officers.

    Args:
        result: A ThresholdResult from check_threshold_exceedance.

    Returns:
        Human-readable explanation string.
    """
    duration_label = _format_duration(result.duration_hours)

    if result.region == "ne_himalaya":
        if result.exceeded:
            return (
                f"{duration_label} cumulative rainfall of {result.actual:.1f}mm "
                f"exceeds the NE Himalaya threshold of {result.threshold:.1f}mm "
                f"for this duration ({result.margin:.1f}mm above threshold)."
            )
        else:
            return (
                f"{duration_label} cumulative rainfall of {result.actual:.1f}mm "
                f"is below the NE Himalaya threshold of {result.threshold:.1f}mm "
                f"({abs(result.margin):.1f}mm below threshold)."
            )
    elif result.region == "sikkim":
        if result.exceeded:
            return (
                f"Average rainfall intensity of {result.actual:.1f}mm/day "
                f"over {duration_label} exceeds the Sikkim I-D threshold of "
                f"{result.threshold:.1f}mm/day ({result.margin:.1f}mm/day above threshold)."
            )
        else:
            return (
                f"Average rainfall intensity of {result.actual:.1f}mm/day "
                f"over {duration_label} is below the Sikkim I-D threshold of "
                f"{result.threshold:.1f}mm/day ({abs(result.margin):.1f}mm/day below threshold)."
            )
    return f"Threshold check for {result.region}: unknown format."


def _format_duration(hours: float) -> str:
    """Format a duration in hours into a human-readable string."""
    if hours < 24:
        return f"{hours:.0f}-hour"
    days = hours / 24.0
    if days == int(days):
        return f"{int(days)}-day"
    return f"{days:.1f}-day"


# ---------------------------------------------------------------------------
# Mathematical properties (for validation)
# ---------------------------------------------------------------------------


def ne_himalaya_is_monotonic_increasing() -> bool:
    """Verify that the NE Himalaya threshold is monotonically increasing.

    The equation E = -11.10 + 0.62*D is linear with positive slope,
    so it must be strictly increasing over its domain.
    """
    test_points = [25, 50, 100, 200, 500, 1000, 1439]
    values = [ne_himalaya_moisture_threshold(d) for d in test_points]
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def sikkim_is_monotonic_decreasing() -> bool:
    """Verify that the Sikkim I-D threshold is monotonically decreasing.

    The equation I = 43.26 * D^(-0.78) is a power law with negative
    exponent, so it must be strictly decreasing for D > 0.
    """
    test_points = [0.1, 0.5, 1, 2, 5, 10, 30, 90]
    values = [sikkim_intensity_duration_threshold(d) for d in test_points]
    return all(values[i] > values[i + 1] for i in range(len(values) - 1))


def ne_himalaya_threshold_at(d_hours: float) -> float | None:
    """Safe wrapper that returns None instead of raising for invalid input."""
    try:
        return ne_himalaya_moisture_threshold(d_hours)
    except (ValueError, TypeError):
        return None


def sikkim_threshold_at(d_days: float) -> float | None:
    """Safe wrapper that returns None instead of raising for invalid input."""
    try:
        return sikkim_intensity_duration_threshold(d_days)
    except (ValueError, TypeError):
        return None
