from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import RiskZoneViewSet, HistoricalLandslideViewSet

router = DefaultRouter()
router.register(r"risk-zones", RiskZoneViewSet)
router.register(r"historical-landslides", HistoricalLandslideViewSet)

urlpatterns = [
    path("", include(router.urls)),
]
