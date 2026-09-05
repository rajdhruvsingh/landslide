"""Alert dispatch rules engine.

Determines whether an alert should be dispatched for a zone.
This module was restored after the legacy FastAPI copy was removed
during the Django migration; it is intentionally minimal until the
full threshold/rainfall heuristics are wired in.
"""

from apps.risk_zones.models import RiskZone


def should_dispatch_alert(zone_id, risk_level, readings=None):
    """Decide whether an alert should be dispatched.

    Args:
        zone_id: primary key of the RiskZone.
        risk_level: current risk string (Low/Moderate/High/Severe).
        readings: optional weather readings (reserved for future heuristics).

    Returns:
        {"should_dispatch": bool, "reason": str}
    """
    if risk_level not in ("High", "Severe"):
        return {
            "should_dispatch": False,
            "reason": f"risk level '{risk_level}' is below the dispatch threshold",
        }

    try:
        zone = RiskZone.objects.get(pk=zone_id)
    except RiskZone.DoesNotExist:
        return {"should_dispatch": False, "reason": "zone not found"}

    return {
        "should_dispatch": True,
        "reason": f"zone '{zone.zone_name}' is at {risk_level} risk",
    }