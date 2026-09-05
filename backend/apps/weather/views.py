from rest_framework import viewsets, status
from rest_framework import serializers
from rest_framework.decorators import api_view
from rest_framework.response import Response

from drf_spectacular.utils import extend_schema, inline_serializer

from .models import WeatherReading
from .serializers import WeatherReadingSerializer


class WeatherReadingViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = WeatherReading.objects.select_related("zone").all()
    serializer_class = WeatherReadingSerializer


@extend_schema(
    parameters=[
        inline_serializer(
            name="ForecastParams",
            fields={"zone_id": serializers.IntegerField()},
        )
    ],
    responses={
        200: inline_serializer(
            name="WeatherForecast",
            fields={
                "zone_id": serializers.IntegerField(),
                "forecast": serializers.ListField(
                    child=inline_serializer(
                        name="ForecastDay",
                        fields={
                            "date": serializers.CharField(),
                            "rainfall_mm": serializers.FloatField(allow_null=True),
                            "soil_moisture_pct": serializers.FloatField(
                                allow_null=True
                            ),
                            "source": serializers.CharField(),
                        },
                    )
                ),
            },
        ),
        400: None,
    },
)
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
