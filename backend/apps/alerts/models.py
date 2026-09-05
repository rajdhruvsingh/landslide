from django.db import models


class Alert(models.Model):
    RISK_LEVELS = [
        ("Low", "Low"),
        ("Moderate", "Moderate"),
        ("High", "High"),
        ("Severe", "Severe"),
    ]

    CHANNELS = [
        ("sms", "SMS"),
        ("push", "Push"),
        ("both", "SMS + Push"),
    ]

    zone = models.ForeignKey(
        "risk_zones.RiskZone",
        on_delete=models.CASCADE,
        related_name="alerts",
    )
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS)
    message = models.TextField()
    language = models.CharField(max_length=10, default="en")
    channel = models.CharField(max_length=20, choices=CHANNELS, default="sms")
    dispatched_at = models.DateTimeField(auto_now_add=True)
    explanation = models.TextField(blank=True, default="")

    class Meta:
        db_table = "alerts"
        ordering = ["-dispatched_at"]

    def __str__(self):
        return f"Alert {self.pk} - {self.risk_level} for {self.zone}"
