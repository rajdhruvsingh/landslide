from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import WeatherReadingViewSet

router = DefaultRouter()
router.register(r"weather-readings", WeatherReadingViewSet)

urlpatterns = [
    path("", include(router.urls)),
]
