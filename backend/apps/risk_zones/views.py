from rest_framework import viewsets
from .models import RiskZone, HistoricalLandslide
from .serializers import RiskZoneSerializer, HistoricalLandslideSerializer


class RiskZoneViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RiskZone.objects.all()
    serializer_class = RiskZoneSerializer


class HistoricalLandslideViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = HistoricalLandslide.objects.all()
    serializer_class = HistoricalLandslideSerializer
