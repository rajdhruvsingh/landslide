from rest_framework import viewsets
from .models import WeatherReading
from .serializers import WeatherReadingSerializer


class WeatherReadingViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = WeatherReading.objects.all()
    serializer_class = WeatherReadingSerializer
