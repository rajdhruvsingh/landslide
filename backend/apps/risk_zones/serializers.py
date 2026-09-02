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
