from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import WeatherReadingViewSet, weather_forecast

router = DefaultRouter()
router.register(r"weather-readings", WeatherReadingViewSet, basename="weatherreading")

urlpatterns = [
    path("weather/forecast", weather_forecast, name="weather-forecast"),
    path("", include(router.urls)),
]
