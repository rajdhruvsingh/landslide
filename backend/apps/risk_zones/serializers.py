from rest_framework import serializers
from .models import RiskZone, HistoricalLandslide


class RiskZoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = RiskZone
        fields = "__all__"


class HistoricalLandslideSerializer(serializers.ModelSerializer):
    class Meta:
        model = HistoricalLandslide
        fields = "__all__"


class RiskZoneHistorySerializer(serializers.Serializer):
    zone_id = serializers.IntegerField()
    zone_name = serializers.CharField()
    current_risk_level = serializers.CharField()
    last_computed_at = serializers.DateTimeField()
    weather_readings = serializers.SerializerMethodField()

    def get_weather_readings(self, obj):
        from apps.weather.models import WeatherReading

        readings = WeatherReading.objects.filter(zone=obj).order_by("-reading_time")[
            :30
        ]
        return [
            {
                "reading_time": r.reading_time.isoformat(),
                "rainfall_mm": r.rainfall_mm,
                "soil_moisture_pct": r.soil_moisture_pct,
            }
            for r in readings
        ]


class RiskZoneExplanationSerializer(serializers.Serializer):
    zone_id = serializers.IntegerField()
    zone_name = serializers.CharField()
    risk_level = serializers.CharField()
    explanation = serializers.CharField()
    thresholds_checked = serializers.SerializerMethodField()
    actual_readings = serializers.SerializerMethodField()

    def get_thresholds_checked(self, obj):
        return self.context.get("thresholds_checked", [])

    def get_actual_readings(self, obj):
        return self.context.get("actual_readings", [])
