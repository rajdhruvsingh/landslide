from rest_framework import serializers
from .models import WeatherReading


class WeatherReadingSerializer(serializers.ModelSerializer):
    zone_name = serializers.CharField(source="zone.zone_name", read_only=True)

    class Meta:
        model = WeatherReading
        fields = "__all__"


class WeatherForecastSerializer(serializers.Serializer):
    zone_id = serializers.IntegerField()
    zone_name = serializers.CharField()
    forecast = serializers.SerializerMethodField()

    def get_forecast(self, obj):
        from apps.weather.models import WeatherReading

        readings = WeatherReading.objects.filter(zone_id=obj).order_by(
            "-reading_time"
        )[:7]
        return [
            {
                "date": r.reading_time.date().isoformat(),
                "rainfall_mm": r.rainfall_mm,
                "soil_moisture_pct": r.soil_moisture_pct,
                "source": r.source,
            }
            for r in readings
        ]
