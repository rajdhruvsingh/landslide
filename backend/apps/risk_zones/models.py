from django.conf import settings
from django.db import models


if settings.GIS_AVAILABLE:
    from django.contrib.gis.db import models as gis_models

    _PolygonField = gis_models.PolygonField
    _PointField = gis_models.PointField
else:

    class _PolygonField(models.JSONField):
        def __init__(self, *args, srid=4326, **kwargs):
            super().__init__(*args, **kwargs)

    class _PointField(models.JSONField):
        def __init__(self, *args, srid=4326, **kwargs):
            super().__init__(*args, **kwargs)


class RiskZone(models.Model):
    zone_name = models.CharField(max_length=255, blank=True, default="")
    district = models.CharField(max_length=100, blank=True, default="")
    state = models.CharField(max_length=100, blank=True, default="")
    geom = _PolygonField(srid=4326, null=True, blank=True)
    current_risk_level = models.CharField(max_length=20, default="Low")
    last_computed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "risk_zones"

    def __str__(self):
        return f"{self.zone_name} ({self.current_risk_level})"


class HistoricalLandslide(models.Model):
    event_date = models.DateField(null=True, blank=True)
    geom = _PointField(srid=4326, null=True, blank=True)
    severity = models.CharField(max_length=20, blank=True, default="")
    source = models.CharField(max_length=50, blank=True, default="")

    class Meta:
        db_table = "historical_landslides"

    def __str__(self):
        return f"Landslide {self.event_date} ({self.severity})"
