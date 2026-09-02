from rest_framework import serializers
from .models import WeatherReading


class WeatherReadingSerializer(serializers.ModelSerializer):
    class Meta:
        model = WeatherReading
        fields = "__all__"
