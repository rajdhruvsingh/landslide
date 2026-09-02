from django.db.models import Count, Q
from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(["GET"])
def dashboard_summary(request):
    """Aggregated stats: risk severity counts, road status, forecast."""
    from apps.risk_zones.models import RiskZone
    from apps.alerts.models import Alert
    from apps.reports.models import FieldReport

    total_zones = RiskZone.objects.count()
    risk_counts = dict(
        RiskZone.objects.values_list("current_risk_level")
        .annotate(count=Count("id"))
        .values_list("current_risk_level", "count")
    )

    active_alerts = Alert.objects.count()
    reports_pending = FieldReport.objects.filter(sync_status="pending").count()

    return Response(
        {
            "total_zones": total_zones,
            "risk_counts": {
                "Low": risk_counts.get("Low", 0),
                "Moderate": risk_counts.get("Moderate", 0),
                "High": risk_counts.get("High", 0),
                "Severe": risk_counts.get("Severe", 0),
            },
            "active_alerts": active_alerts,
            "reports_pending": reports_pending,
        }
    )
