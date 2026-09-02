from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import WeatherReading
from .serializers import WeatherReadingSerializer


class WeatherReadingViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = WeatherReading.objects.select_related("zone").all()
    serializer_class = WeatherReadingSerializer


@api_view(["GET"])
def weather_forecast(request):
    """IMD-linked forecast for a zone."""
    zone_id = request.query_params.get("zone_id")
    if not zone_id:
        return Response(
            {"error": "zone_id query parameter is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    try:
        zone_id = int(zone_id)
    except (TypeError, ValueError):
        return Response(
            {"error": "zone_id must be an integer"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    readings = WeatherReading.objects.filter(zone_id=zone_id).order_by(
        "-reading_time"
    )[:7]
    forecast = [
        {
            "date": r.reading_time.date().isoformat(),
            "rainfall_mm": r.rainfall_mm,
            "soil_moisture_pct": r.soil_moisture_pct,
            "source": r.source,
        }
        for r in readings
    ]
    return Response({"zone_id": zone_id, "forecast": forecast})
