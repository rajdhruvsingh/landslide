"""Operational zone-level risk recomputation.

The threshold model in ``apps.ml_bridge.ml.threshold_model`` is the always-active
BASELINE layer of the system (published NE-Himalaya rainfall-threshold
equations). This module evaluates the latest stored weather readings for each
risk zone against those thresholds and persists the resulting exposure level
back onto the ``RiskZone`` row (``current_risk_level`` / ``last_computed_at``),
so the dashboard and summary aggregates always reflect the latest state.

The ML classifier (Phase 2+) is an optional refinement layer that sits ON TOP
of these thresholds; it does not replace them. No trained-model artifact is
required for this module to run — a missing ``.pkl/.joblib`` file never blocks
the baseline recomputation.
"""

from __future__ import annotations

import logging
from typing import Any

from django.utils import timezone

from apps.ml_bridge.ml.threshold_model import (
    ThresholdResult,
    check_threshold_exceedance,
    format_explanation,
    select_best_result,
)

logger = logging.getLogger(__name__)

_EXPLANATION_DURATION_HOURS = 48
_MAX_READINGS_PER_ZONE = 10


def evaluate_zone_from_readings(zone: Any) -> dict[str, Any] | None:
    """Evaluate a single zone against thresholds using its latest readings.

    Args:
        zone: A RiskZone instance (imported lazily inside the caller to keep
            this module importable from plain-Python contexts).

    Returns:
        Dict with ``level``, ``explanation`` and ``thresholds_checked``, or
        ``None`` when the zone has no readable (rainfall-bearing) readings.
    """
    from apps.weather.models import WeatherReading

    readings = WeatherReading.objects.filter(zone=zone).order_by("-reading_time")[
        :_MAX_READINGS_PER_ZONE
    ]

    checks: list[tuple[ThresholdResult, Any]] = []
    for reading in readings:
        if reading.rainfall_mm is None:
            continue
        result = check_threshold_exceedance(
            cumulative_rainfall_mm=reading.rainfall_mm,
            duration_hours=_EXPLANATION_DURATION_HOURS,
            region="ne_himalaya",
        )
        checks.append((result, reading))

    if not checks:
        return None

    # The "best" check drives the explanation text: an exceedance with the
    # highest positive margin, otherwise the closest-to-threshold reading.
    best = select_best_result([result for result, _ in checks])
    level = "High" if any(result.exceeded for result, _ in checks) else "Low"

    return {
        "level": level,
        "explanation": format_explanation(best[0]),
        "thresholds_checked": [
            {
                "date": reading.reading_time.isoformat()
                if reading.reading_time
                else None,
                "threshold": result.to_dict(),
            }
            for result, reading in checks
        ],
    }


def recompute_zone_risks(zone_ids: list[int] | None = None) -> dict[str, Any]:
    """Recompute baseline risk for all zones (optionally a subset).

    Args:
        zone_ids: Restrict to these RiskZone primary keys when provided.

    Returns:
        Summary dict with per-zone results, mirroring the Celery task contract.
    """
    from apps.risk_zones.models import RiskZone

    queryset = RiskZone.objects.all()
    if zone_ids:
        queryset = queryset.filter(pk__in=zone_ids)

    now = timezone.now()
    results: list[dict[str, Any]] = []

    for zone in queryset:
        evaluated = evaluate_zone_from_readings(zone)
        if evaluated is None:
            results.append(
                {
                    "zone_id": zone.pk,
                    "zone_name": zone.zone_name,
                    "updated": False,
                    "reason": "no weather readings with rainfall for this zone",
                }
            )
            continue

        zone.current_risk_level = evaluated["level"]
        zone.last_computed_at = now
        zone.save(update_fields=["current_risk_level", "last_computed_at"])
        logger.info(
            "Zone %s (%s) -> %s",
            zone.pk,
            zone.zone_name,
            evaluated["level"],
        )

        results.append(
            {
                "zone_id": zone.pk,
                "zone_name": zone.zone_name,
                "updated": True,
                "risk_level": evaluated["level"],
                "explanation": evaluated["explanation"],
            }
        )

    return {"status": "success", "zones": results}