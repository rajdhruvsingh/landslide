from django.conf import settings
from django.db import models


if settings.GIS_AVAILABLE:
    from django.contrib.gis.db import models as gis_models

    _PointField = gis_models.PointField
else:

    class _PointField(models.JSONField):
        def __init__(self, *args, srid=4326, **kwargs):
            super().__init__(*args, **kwargs)


class FieldReport(models.Model):
    REPORT_TYPES = [
        ("landslide", "Landslide"),
        ("flood", "Flood"),
        ("road_block", "Road Block"),
        ("crack", "Crack/Subsidence"),
        ("other", "Other"),
    ]

    SYNC_STATUSES = [
        ("synced", "Synced"),
        ("pending", "Pending"),
        ("conflict", "Conflict"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="field_reports",
    )
    geom = _PointField(srid=4326, null=True, blank=True)
    photo_url = models.TextField(blank=True, default="")
    video_url = models.TextField(blank=True, default="")
    description = models.TextField(blank=True, default="")
    report_type = models.CharField(max_length=50, choices=REPORT_TYPES, default="other")
    submitted_at = models.DateTimeField(auto_now_add=True)
    sync_status = models.CharField(
        max_length=20, choices=SYNC_STATUSES, default="synced"
    )

    class Meta:
        db_table = "field_reports"
        ordering = ["-submitted_at"]

    def __str__(self):
        return f"Report {self.pk} by {self.user} ({self.report_type})"
