"""Comprehensive tests for published rainfall threshold models.

All expected values in this file are computed directly from the published
equations. No values are invented or approximated from external sources.

Test categories:
    1. NE Himalaya moisture threshold (linear)
    2. Sikkim intensity-duration threshold (power law)
    3. Threshold exceedance checking
    4. Explanation generation
    5. Mathematical properties (monotonicity)
    6. Edge cases and invalid inputs
    7. Type safety
"""

import pytest

from apps.ml_bridge.ml.threshold_model import (
    NE_HIMALAYA_D_MAX_HOURS,
    NE_HIMALAYA_D_MIN_HOURS,
    SIKKIM_D_MIN_DAYS,
    ThresholdResult,
    check_threshold_exceedance,
    format_explanation,
    ne_himalaya_is_monotonic_increasing,
    ne_himalaya_moisture_threshold,
    ne_himalaya_threshold_at,
    sikkim_intensity_duration_threshold,
    sikkim_is_monotonic_decreasing,
    sikkim_threshold_at,
)


# ============================================================================
# 1. NE Himalaya Moisture Threshold: E = -11.10 + 0.62 * D
# ============================================================================


class TestNeHimalayaMoistureThreshold:
    """Tests for the NE Himalaya cumulative rainfall threshold.

    Equation: E(mm) = -11.10 + 0.62 * D(hr)
    Valid for: 24 < D < 1440 hours
    """

    def test_equation_at_48_hours(self):
        # E = -11.10 + 0.62 * 48 = -11.10 + 29.76 = 18.66
        assert ne_himalaya_moisture_threshold(48) == pytest.approx(18.66, abs=0.01)

    def test_equation_at_72_hours(self):
        # E = -11.10 + 0.62 * 72 = -11.10 + 44.64 = 33.54
        assert ne_himalaya_moisture_threshold(72) == pytest.approx(33.54, abs=0.01)

    def test_equation_at_168_hours(self):
        # E = -11.10 + 0.62 * 168 = -11.10 + 104.16 = 93.06
        assert ne_himalaya_moisture_threshold(168) == pytest.approx(93.06, abs=0.01)

    def test_equation_at_100_hours(self):
        # E = -11.10 + 0.62 * 100 = -11.10 + 62.0 = 50.90
        assert ne_himalaya_moisture_threshold(100) == pytest.approx(50.90, abs=0.01)

    def test_equation_at_50_hours(self):
        # E = -11.10 + 0.62 * 50 = -11.10 + 31.0 = 19.90
        assert ne_himalaya_moisture_threshold(50) == pytest.approx(19.90, abs=0.01)

    def test_equation_at_25_hours(self):
        # E = -11.10 + 0.62 * 25 = -11.10 + 15.5 = 4.40
        assert ne_himalaya_moisture_threshold(25) == pytest.approx(4.40, abs=0.01)

    def test_equation_at_1439_hours(self):
        # E = -11.10 + 0.62 * 1439 = -11.10 + 892.18 = 881.08
        assert ne_himalaya_moisture_threshold(1439) == pytest.approx(881.08, abs=0.01)

    def test_equation_at_1000_hours(self):
        # E = -11.10 + 0.62 * 1000 = -11.10 + 620.0 = 608.90
        assert ne_himalaya_moisture_threshold(1000) == pytest.approx(608.90, abs=0.01)

    def test_boundary_exactly_24_raises(self):
        with pytest.raises(ValueError, match="d_hours must be in"):
            ne_himalaya_moisture_threshold(24)

    def test_boundary_exactly_1440_raises(self):
        with pytest.raises(ValueError, match="d_hours must be in"):
            ne_himalaya_moisture_threshold(1440)

    def test_below_24_raises(self):
        with pytest.raises(ValueError):
            ne_himalaya_moisture_threshold(23.99)

    def test_above_1440_raises(self):
        with pytest.raises(ValueError):
            ne_himalaya_moisture_threshold(1440.01)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            ne_himalaya_moisture_threshold(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            ne_himalaya_moisture_threshold(-10)

    def test_very_large_raises(self):
        with pytest.raises(ValueError):
            ne_himalaya_moisture_threshold(10000)

    def test_float_boundary_just_above_24(self):
        # 24.001 is valid (just above the exclusive boundary)
        result = ne_himalaya_moisture_threshold(24.001)
        assert result == pytest.approx(-11.10 + 0.62 * 24.001, abs=0.01)

    def test_float_boundary_just_below_1440(self):
        # 1439.999 is valid (just below the exclusive boundary)
        result = ne_himalaya_moisture_threshold(1439.999)
        assert result == pytest.approx(-11.10 + 0.62 * 1439.999, abs=0.01)


# ============================================================================
# 2. Sikkim Intensity-Duration Threshold: I = 43.26 * D^(-0.78)
# ============================================================================


class TestSikkimIntensityDurationThreshold:
    """Tests for the Sikkim I-D threshold.

    Equation: I = 43.26 * D^(-0.78)
    Valid for: D > 0 days
    Output: I in mm/day
    """

    def test_equation_at_1_day(self):
        # I = 43.26 * 1^(-0.78) = 43.26
        assert sikkim_intensity_duration_threshold(1) == pytest.approx(43.26, abs=0.01)

    def test_equation_at_7_days(self):
        # I = 43.26 * 7^(-0.78)
        expected = 43.26 * (7 ** (-0.78))
        assert sikkim_intensity_duration_threshold(7) == pytest.approx(
            expected, abs=0.01
        )

    def test_equation_at_30_days(self):
        # I = 43.26 * 30^(-0.78)
        expected = 43.26 * (30 ** (-0.78))
        assert sikkim_intensity_duration_threshold(30) == pytest.approx(
            expected, abs=0.01
        )

    def test_equation_at_05_days(self):
        # I = 43.26 * 0.5^(-0.78)
        expected = 43.26 * (0.5 ** (-0.78))
        assert sikkim_intensity_duration_threshold(0.5) == pytest.approx(
            expected, abs=0.01
        )

    def test_equation_at_90_days(self):
        # I = 43.26 * 90^(-0.78)
        expected = 43.26 * (90 ** (-0.78))
        assert sikkim_intensity_duration_threshold(90) == pytest.approx(
            expected, abs=0.01
        )

    def test_equation_at_01_days(self):
        # I = 43.26 * 0.1^(-0.78)
        expected = 43.26 * (0.1 ** (-0.78))
        assert sikkim_intensity_duration_threshold(0.1) == pytest.approx(
            expected, abs=0.01
        )

    def test_boundary_zero_raises(self):
        with pytest.raises(ValueError, match="d_days must be >"):
            sikkim_intensity_duration_threshold(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="d_days must be >"):
            sikkim_intensity_duration_threshold(-1)

    def test_very_small_positive(self):
        # 0.001 days = ~1.44 minutes — technically valid but unrealistic
        result = sikkim_intensity_duration_threshold(0.001)
        assert result > 0
        # Should be very large (short duration = high critical intensity)
        assert result > 1000


# ============================================================================
# 3. Threshold Exceedance Checking
# ============================================================================


class TestCheckThresholdExceedance:
    """Tests for the unified threshold exceedance checker."""

    # --- NE Himalaya region ---

    def test_ne_himalaya_exceeds_at_72h(self):
        # Threshold at 72h = 33.54mm. Rainfall of 40mm exceeds it.
        result = check_threshold_exceedance(40.0, 72, region="ne_himalaya")
        assert result.exceeded is True
        assert result.threshold == pytest.approx(33.54, abs=0.01)
        assert result.margin == pytest.approx(40.0 - 33.54, abs=0.01)
        assert result.unit == "mm"

    def test_ne_himalaya_does_not_exceed_at_72h(self):
        # Threshold at 72h = 33.54mm. Rainfall of 20mm does not exceed.
        result = check_threshold_exceedance(20.0, 72, region="ne_himalaya")
        assert result.exceeded is False
        assert result.margin < 0

    def test_ne_himalaya_exactly_at_threshold(self):
        # Threshold at 72h = 33.54mm. Rainfall of exactly 33.54mm: NOT exceeded.
        # The equation says > (strict), so equal is not exceeded.
        result = check_threshold_exceedance(33.54, 72, region="ne_himalaya")
        assert result.exceeded is False
        assert result.margin == pytest.approx(0.0, abs=0.01)

    def test_ne_himalaya_just_above_threshold(self):
        # Threshold at 72h = 33.54mm. Rainfall of 33.55mm exceeds.
        result = check_threshold_exceedance(33.55, 72, region="ne_himalaya")
        assert result.exceeded is True

    def test_ne_himalaya_duration_48h(self):
        # Threshold at 48h = 18.66mm
        result = check_threshold_exceedance(25.0, 48, region="ne_himalaya")
        assert result.exceeded is True
        assert result.threshold == pytest.approx(18.66, abs=0.01)

    def test_ne_himalaya_large_rainfall(self):
        # 200mm over 100h: threshold = 50.9mm, clearly exceeds
        result = check_threshold_exceedance(200.0, 100, region="ne_himalaya")
        assert result.exceeded is True
        assert result.margin > 100

    def test_ne_himalaya_zero_rainfall(self):
        result = check_threshold_exceedance(0.0, 72, region="ne_himalaya")
        assert result.exceeded is False
        assert result.margin < 0

    # --- Sikkim region ---

    def test_sikkim_exceeds_at_1_day(self):
        # Threshold at 1 day = 43.26mm/day. Intensity of 50mm/day exceeds.
        # cumulative = 50mm over 1 day = 24h
        result = check_threshold_exceedance(50.0, 24, region="sikkim")
        assert result.exceeded is True
        assert result.unit == "mm/day"
        assert result.actual == pytest.approx(50.0, abs=0.01)

    def test_sikkim_does_not_exceed_at_1_day(self):
        # cumulative = 30mm over 1 day = 30mm/day < 43.26mm/day
        result = check_threshold_exceedance(30.0, 24, region="sikkim")
        assert result.exceeded is False

    def test_sikkim_exceeds_at_7_days(self):
        # Threshold at 7 days = 9.48mm/day. Intensity of 15mm/day exceeds.
        # cumulative = 15 * 7 = 105mm over 168h
        result = check_threshold_exceedance(105.0, 168, region="sikkim")
        assert result.exceeded is True
        assert result.actual == pytest.approx(15.0, abs=0.01)

    def test_sikkim_does_not_exceed_at_7_days(self):
        # cumulative = 50mm over 168h = 50/7 = 7.14mm/day < 9.48mm/day
        result = check_threshold_exceedance(50.0, 168, region="sikkim")
        assert result.exceeded is False

    def test_sikkim_exact_threshold_not_exceeded(self):
        # Threshold at 1 day = 43.26mm/day. Exactly 43.26mm/day: NOT exceeded (> only)
        result = check_threshold_exceedance(43.26, 24, region="sikkim")
        assert result.exceeded is False

    # --- Region validation ---

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="Unknown region"):
            check_threshold_exceedance(50.0, 24, region="unknown")

    def test_ne_himalaya_invalid_duration_raises(self):
        with pytest.raises(ValueError):
            check_threshold_exceedance(50.0, 24, region="ne_himalaya")

    # --- Result structure ---

    def test_result_is_threshold_result(self):
        result = check_threshold_exceedance(40.0, 72, region="ne_himalaya")
        assert isinstance(result, ThresholdResult)

    def test_result_has_all_fields(self):
        result = check_threshold_exceedance(40.0, 72, region="ne_himalaya")
        assert hasattr(result, "region")
        assert hasattr(result, "exceeded")
        assert hasattr(result, "threshold")
        assert hasattr(result, "actual")
        assert hasattr(result, "unit")
        assert hasattr(result, "margin")
        assert hasattr(result, "duration_hours")

    def test_result_to_dict(self):
        result = check_threshold_exceedance(40.0, 72, region="ne_himalaya")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "exceeded" in d
        assert "threshold" in d
        assert "margin" in d


# ============================================================================
# 4. Explanation Generation
# ============================================================================


class TestFormatExplanation:
    """Tests for human-readable explanation output."""

    def test_ne_himalaya_exceeded_explanation(self):
        result = check_threshold_exceedance(60.0, 72, region="ne_himalaya")
        explanation = format_explanation(result)
        # 72h = 3 days exactly, so _format_duration returns "3-day"
        assert "3-day" in explanation
        assert "60.0mm" in explanation
        assert "33.5mm" in explanation  # threshold ≈ 33.54
        assert "exceeds" in explanation.lower()
        assert "above threshold" in explanation.lower()

    def test_ne_himalaya_not_exceeded_explanation(self):
        result = check_threshold_exceedance(20.0, 72, region="ne_himalaya")
        explanation = format_explanation(result)
        assert "20.0mm" in explanation
        assert "below" in explanation.lower()

    def test_sikkim_exceeded_explanation(self):
        result = check_threshold_exceedance(50.0, 24, region="sikkim")
        explanation = format_explanation(result)
        assert "mm/day" in explanation
        assert "exceeds" in explanation.lower()

    def test_sikkim_not_exceeded_explanation(self):
        result = check_threshold_exceedance(30.0, 24, region="sikkim")
        explanation = format_explanation(result)
        assert "below" in explanation.lower()

    def test_explanation_is_string(self):
        result = check_threshold_exceedance(40.0, 72, region="ne_himalaya")
        explanation = format_explanation(result)
        assert isinstance(explanation, str)
        assert len(explanation) > 20

    def test_explanation_duration_label_days(self):
        # 168 hours = 7 days — should say "7-day"
        result = check_threshold_exceedance(100.0, 168, region="ne_himalaya")
        explanation = format_explanation(result)
        assert "7-day" in explanation

    def test_explanation_duration_label_hours(self):
        # 48 hours = 2 days exactly, so _format_duration returns "2-day"
        result = check_threshold_exceedance(20.0, 48, region="ne_himalaya")
        explanation = format_explanation(result)
        assert "2-day" in explanation

    def test_explanation_duration_label_partial_days(self):
        # 36 hours = 1.5 days — should say "1.5-day"
        result = check_threshold_exceedance(30.0, 36, region="ne_himalaya")
        explanation = format_explanation(result)
        assert "1.5-day" in explanation

    def test_explanation_duration_label_sub_day(self):
        # 12 hours — should say "12-hour"
        # Note: this is outside the NE-Himalaya domain (24 < D < 1440),
        # so we test with a valid value and verify the format logic directly.
        from apps.ml_bridge.ml.threshold_model import _format_duration

        assert _format_duration(12) == "12-hour"
        assert _format_duration(48) == "2-day"
        assert _format_duration(72) == "3-day"
        assert _format_duration(36) == "1.5-day"

    def test_explanation_margin_included(self):
        result = check_threshold_exceedance(60.0, 72, region="ne_himalaya")
        explanation = format_explanation(result)
        # Margin = 60.0 - 33.54 ≈ 26.46
        assert "26.5mm above" in explanation


# ============================================================================
# 5. Mathematical Properties
# ============================================================================


class TestMathematicalProperties:
    """Verify mathematical invariants of the threshold equations."""

    def test_ne_himalaya_monotonic_increasing(self):
        """NE-Himalaya threshold must be strictly increasing in D."""
        assert ne_himalaya_is_monotonic_increasing() is True

    def test_sikkim_monotonic_decreasing(self):
        """Sikkim I-D threshold must be strictly decreasing in D."""
        assert sikkim_is_monotonic_decreasing() is True

    def test_ne_himalaya_linear_slope(self):
        """Verify the slope is exactly 0.62 across the domain."""
        d1, d2 = 100.0, 200.0
        e1 = ne_himalaya_moisture_threshold(d1)
        e2 = ne_himalaya_moisture_threshold(d2)
        slope = (e2 - e1) / (d2 - d1)
        assert slope == pytest.approx(0.62, abs=0.0001)

    def test_ne_himalaya_intercept(self):
        """Verify the intercept is -11.10."""
        # E = -11.10 + 0.62 * D
        # At D=0 (extrapolated): E = -11.10
        d = 100.0
        e = ne_himalaya_moisture_threshold(d)
        intercept = e - 0.62 * d
        assert intercept == pytest.approx(-11.10, abs=0.0001)

    def test_sikkim_power_law_exponent(self):
        """Verify the exponent is -0.78."""
        d1, d2 = 1.0, 10.0
        i1 = sikkim_intensity_duration_threshold(d1)
        i2 = sikkim_intensity_duration_threshold(d2)
        # I2/I1 = (D2/D1)^(-0.78)
        ratio = i2 / i1
        expected_ratio = (d2 / d1) ** (-0.78)
        assert ratio == pytest.approx(expected_ratio, abs=0.0001)

    def test_sikkim_coefficient(self):
        """Verify the coefficient is 43.26."""
        d = 1.0
        i = sikkim_intensity_duration_threshold(d)
        # At D=1: I = 43.26 * 1^(-0.78) = 43.26
        assert i == pytest.approx(43.26, abs=0.0001)

    def test_ne_himalaya_positive_threshold_above_18h(self):
        """Threshold becomes positive for D > 11.10/0.62 ≈ 17.9 hours.

        Note: The valid domain is 24 < D < 1440. At D=25 (just inside
        the domain), the threshold is already positive: -11.10 + 0.62*25 = 4.4mm.
        We test here that the linear equation produces positive values
        within the valid domain.
        """
        e = ne_himalaya_moisture_threshold(25)
        assert e > 0
        assert e == pytest.approx(4.40, abs=0.01)

    def test_sikkim_threshold_always_positive(self):
        """Power law with positive coefficient is always positive for D > 0."""
        for d in [0.01, 0.1, 1, 10, 100, 1000]:
            assert sikkim_intensity_duration_threshold(d) > 0


# ============================================================================
# 6. Safe Wrappers
# ============================================================================


class TestSafeWrappers:
    """Tests for the safe wrapper functions that return None on error."""

    def test_ne_himalaya_threshold_at_valid(self):
        assert ne_himalaya_threshold_at(72) == pytest.approx(33.54, abs=0.01)

    def test_ne_himalaya_threshold_at_invalid_returns_none(self):
        assert ne_himalaya_threshold_at(24) is None
        assert ne_himalaya_threshold_at(1440) is None
        assert ne_himalaya_threshold_at(-1) is None

    def test_ne_himalaya_threshold_at_type_error_returns_none(self):
        assert ne_himalaya_threshold_at("72") is None  # type: ignore[arg-type]

    def test_sikkim_threshold_at_valid(self):
        assert sikkim_threshold_at(1) == pytest.approx(43.26, abs=0.01)

    def test_sikkim_threshold_at_invalid_returns_none(self):
        assert sikkim_threshold_at(0) is None
        assert sikkim_threshold_at(-1) is None

    def test_sikkim_threshold_at_type_error_returns_none(self):
        assert sikkim_threshold_at("7") is None  # type: ignore[arg-type]


# ============================================================================
# 7. Type Safety
# ============================================================================


class TestTypeSafety:
    """Verify functions handle incorrect types gracefully."""

    def test_ne_himalaya_string_raises_type_error(self):
        with pytest.raises(TypeError):
            ne_himalaya_moisture_threshold("72")  # type: ignore[arg-type]

    def test_ne_himalaya_none_raises_type_error(self):
        with pytest.raises(TypeError):
            ne_himalaya_moisture_threshold(None)  # type: ignore[arg-type]

    def test_ne_himalaya_bool_raises_type_error(self):
        # Python's isinstance(True, (int, float)) returns True because
        # bool is a subclass of int. So True passes the type check and
        # then fails the domain check (ValueError).
        # We verify it raises an error (either TypeError or ValueError).
        with pytest.raises((TypeError, ValueError)):
            ne_himalaya_moisture_threshold(True)  # type: ignore[arg-type]

    def test_sikkim_string_raises_type_error(self):
        with pytest.raises(TypeError):
            sikkim_intensity_duration_threshold("7")  # type: ignore[arg-type]

    def test_sikkim_none_raises_type_error(self):
        with pytest.raises(TypeError):
            sikkim_intensity_duration_threshold(None)  # type: ignore[arg-type]


# ============================================================================
# 8. Integration: Threshold → Exceedance → Explanation Pipeline
# ============================================================================


class TestThresholdToExplanationPipeline:
    """End-to-end test: threshold → exceedance → explanation.

    This simulates what the risk engine will do in production:
    1. Compute threshold
    2. Check if rainfall exceeds it
    3. Generate explanation
    4. Pass to alert system
    """

    def test_ne_himalaya_full_pipeline_exceeded(self):
        # Simulate: 72h cumulative rainfall = 80mm
        result = check_threshold_exceedance(80.0, 72, region="ne_himalaya")
        assert result.exceeded is True
        explanation = format_explanation(result)
        assert isinstance(explanation, str)
        # Explanation must contain actual values
        assert "80.0mm" in explanation
        # 72h = 3 days exactly, so _format_duration returns "3-day"
        assert "3-day" in explanation

    def test_sikkim_full_pipeline_exceeded(self):
        # Simulate: 3 days of heavy rain, 60mm total
        result = check_threshold_exceedance(60.0, 72, region="sikkim")
        # 60mm / 3 days = 20mm/day. Threshold at 3 days = 43.26 * 3^(-0.78) ≈ 18.43mm/day
        assert result.exceeded is True
        explanation = format_explanation(result)
        assert "mm/day" in explanation

    def test_ne_himalaya_full_pipeline_not_exceeded(self):
        # Simulate: 48h cumulative rainfall = 10mm
        result = check_threshold_exceedance(10.0, 48, region="ne_himalaya")
        assert result.exceeded is False
        explanation = format_explanation(result)
        assert "below" in explanation.lower()

    def test_explanation_not_hardcoded(self):
        """Verify explanations contain dynamic values, not static strings."""
        r1 = check_threshold_exceedance(50.0, 72, region="ne_himalaya")
        r2 = check_threshold_exceedance(100.0, 72, region="ne_himalaya")
        e1 = format_explanation(r1)
        e2 = format_explanation(r2)
        # Different inputs must produce different explanations
        assert e1 != e2
        # The different parts should be the actual rainfall values
        assert "50.0mm" in e1
        assert "100.0mm" in e2


# ============================================================================
# 9. Constants verification
# ============================================================================


class TestConstants:
    """Verify the module-level constants match the paper's domain."""

    def test_ne_himalaya_d_min(self):
        assert NE_HIMALAYA_D_MIN_HOURS == 24.0

    def test_ne_himalaya_d_max(self):
        assert NE_HIMALAYA_D_MAX_HOURS == 1440.0

    def test_sikkim_d_min(self):
        assert SIKKIM_D_MIN_DAYS == 0.0
