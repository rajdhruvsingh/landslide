from django.db import models


class WeatherReading(models.Model):
    zone = models.ForeignKey(
        "risk_zones.RiskZone",
        on_delete=models.CASCADE,
        related_name="weather_readings",
    )
    station_id = models.CharField(max_length=50)
    reading_time = models.DateTimeField()
    rainfall_mm = models.FloatField(null=True, blank=True)
    soil_moisture_pct = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=50, blank=True, default="")

    class Meta:
        db_table = "weather_readings"
        ordering = ["-reading_time"]

    def __str__(self):
        return f"Weather {self.station_id} @ {self.reading_time}"
