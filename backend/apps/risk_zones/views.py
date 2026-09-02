from django.db.models import Count
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import RiskZone, HistoricalLandslide
from .serializers import (
    RiskZoneSerializer,
    HistoricalLandslideSerializer,
    RiskZoneHistorySerializer,
    RiskZoneExplanationSerializer,
)


class RiskZoneViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RiskZone.objects.all()
    serializer_class = RiskZoneSerializer

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
        )
        from apps.weather.models import WeatherReading

        readings = WeatherReading.objects.filter(zone=zone).order_by(
            "-reading_time"
        )[:10]
        thresholds_checked = []
        actual_readings = []

        for r in readings:
            if r.rainfall_mm is not None:
                result = check_threshold_exceedance(
                    rainfall_mm=r.rainfall_mm,
                    duration_hours=24,
                    region="ne_himalaya",
                )
                if result:
                    thresholds_checked.append(
                        {"date": r.reading_time.isoformat(), "threshold": result}
                    )
            actual_readings.append(
                {
                    "date": r.reading_time.isoformat() if r.reading_time else None,
                    "rainfall_mm": r.rainfall_mm,
                    "soil_moisture_pct": r.soil_moisture_pct,
                }
            )

        explanation_text = format_explanation(thresholds_checked)

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
