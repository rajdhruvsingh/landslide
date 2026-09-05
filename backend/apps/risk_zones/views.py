from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, SAFE_METHODS
from rest_framework.response import Response

from apps.users.permissions import IsAdmin
from .models import RiskZone, HistoricalLandslide
from .serializers import (
    RiskZoneSerializer,
    HistoricalLandslideSerializer,
    RiskZoneHistorySerializer,
)


class RiskZoneViewSet(viewsets.ModelViewSet):
    """Risk zones are read-only for every authenticated user.

    Write endpoints exist but are restricted to district/state admins as a
    safeguard; risk data is normally written by the ML pipeline/Celery
    tasks directly through the ORM, not via the API.
    """

    queryset = RiskZone.objects.all()
    serializer_class = RiskZoneSerializer

    def get_permissions(self):
        if self.request.method in SAFE_METHODS:
            return [IsAuthenticated()]
        return [IsAdmin()]

    @action(detail=True, methods=["get"], url_path="history")
    def history(self, request, pk=None):
        zone = self.get_object()
        serializer = RiskZoneHistorySerializer(zone)
        return Response(serializer.data)

    @action(detail=True, methods=["get"], url_path="explanation")
    def explanation(self, request, pk=None):
        zone = self.get_object()
        from apps.ml_bridge.ml.threshold_model import (
            check_threshold_exceedance,
            format_explanation,
            select_best_result,
        )
        from apps.weather.models import WeatherReading

        readings = WeatherReading.objects.filter(zone=zone).order_by(
            "-reading_time"
        )[:10]
        thresholds_checked = []
        results = []
        actual_readings = []

        for r in readings:
            if r.rainfall_mm is not None:
                result = check_threshold_exceedance(
                    cumulative_rainfall_mm=r.rainfall_mm,
                    duration_hours=48,
                    region="ne_himalaya",
                )
                thresholds_checked.append(
                    {"date": r.reading_time.isoformat(), "threshold": result.to_dict()}
                )
                results.append(result)
            actual_readings.append(
                {
                    "date": r.reading_time.isoformat() if r.reading_time else None,
                    "rainfall_mm": r.rainfall_mm,
                    "soil_moisture_pct": r.soil_moisture_pct,
                }
            )

        # The headline explanation comes from the single most extreme check —
        # shared with apps.ml_bridge.risk_evaluator via select_best_result.
        explanation_text = (
            format_explanation(select_best_result(results)) if results else ""
        )

        return Response(
            {
                "zone_id": zone.pk,
                "zone_name": zone.zone_name,
                "risk_level": zone.current_risk_level,
                "explanation": explanation_text,
                "thresholds_checked": thresholds_checked,
                "actual_readings": actual_readings,
            }
        )


class HistoricalLandslideViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = HistoricalLandslide.objects.all()
    serializer_class = HistoricalLandslideSerializer
    permission_classes = [IsAuthenticated]
